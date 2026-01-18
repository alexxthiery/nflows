# nflows/transforms.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from flax import linen as nn

from .nets import MLP, Array, PRNGKey, validate_conditioner
import nflows.scalar_function as scalar_function
from .splines import rational_quadratic_spline


def stable_logit(p: Array) -> Array:
    """
    Numerically stable logit function: logit(p) = log(p / (1 - p)).

    Clips input to [1e-6, 1 - 1e-6] to avoid log(0) or log(inf).

    Arguments:
        p: Probability values in (0, 1).

    Returns:
        Logit of p.
    """
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)

# ===================================================================
# Linear Transform with LU-style parameterization
# ===================================================================
@dataclass
class LinearTransform:
    """
    Global linear transform with LU-style parameterization.

    We parameterize an invertible matrix W ∈ R^{dim×dim} as:

      L = tril(lower_raw, k = -1) + I        (unit-diagonal lower)
      U = triu(upper_raw, k = 1)             (zero diagonal upper)
      s = exp(log_diag)                      (positive diagonal entries)
      T = U + diag(s)                        (upper-triangular)
      W = L @ T

    For a column vector u, the forward map is:

      a = T @ u
      u' = L @ a

    For batched row-vectors x with shape (..., dim), we flatten batch
    dimensions and apply this to the last dimension.

    The Jacobian of the forward map y = x W^T has determinant det(W), and

      log |det W| = sum(log_diag).

    Inverse uses triangular solves:

      L a = u'      (forward substitution, unit diagonal)
      T u = a       (back substitution)

    Parameters (per block):
      params["lower"]:    unconstrained raw lower-tri part, shape (dim, dim)
      params["upper"]:    unconstrained raw upper-tri part, shape (dim, dim)
      params["log_diag"]: diagonal log-scales, shape (dim,)

    This yields O(dim^2) apply / inverse and O(dim) log-det, without any
    repeated matrix factorizations inside the forward pass.
    """
    dim: int

    def _reconstruct_L_U_s(self, params: Any) -> Tuple[Array, Array, Array]:
        try:
            lower_raw = jnp.asarray(params["lower"])
            upper_raw = jnp.asarray(params["upper"])
            log_diag = jnp.asarray(params["log_diag"])
        except Exception as e:
            raise KeyError(
                "LinearTransform: params must contain 'lower', 'upper', 'log_diag'"
            ) from e

        if lower_raw.shape != (self.dim, self.dim):
            raise ValueError(
                f"LinearTransform: lower must have shape ({self.dim}, {self.dim}), "
                f"got {lower_raw.shape}"
            )
        if upper_raw.shape != (self.dim, self.dim):
            raise ValueError(
                f"LinearTransform: upper must have shape ({self.dim}, {self.dim}), "
                f"got {upper_raw.shape}"
            )
        if log_diag.shape != (self.dim,):
            raise ValueError(
                f"LinearTransform: log_diag must have shape ({self.dim},), "
                f"got {log_diag.shape}"
            )

        L = jnp.tril(lower_raw, k=-1) + jnp.eye(self.dim, dtype=lower_raw.dtype)
        U = jnp.triu(upper_raw, k=1)
        s = jnp.exp(log_diag)
        return L, U, s

    def forward(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'log_diag'.
          x: input tensor of shape (..., dim).
          context: ignored (accepted for interface compatibility).

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det ∂y/∂x| = sum(log_diag), shape x.shape[:-1].
        """
        del context  # Unused in LinearTransform.
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {x.shape[-1]}"
            )

        L, U, s = self._reconstruct_L_U_s(params)
        T = U + jnp.diag(s)

        # Flatten batch dims for simpler linear algebra.
        batch_shape = x.shape[:-1]
        x_flat = x.reshape((-1, self.dim))  # (B, dim)
        u = x_flat.T                        # (dim, B)

        # Column-style forward: u' = L @ (T @ u)
        a = T @ u
        u_prime = L @ a
        y_flat = u_prime.T                  # (B, dim)
        y = y_flat.reshape(batch_shape + (self.dim,))

        # log |det W| = sum(log_diag)
        log_det_scalar = jnp.sum(jnp.log(s))  # or equivalently jnp.sum(log_diag)
        log_det_forward = jnp.broadcast_to(log_det_scalar, batch_shape)
        return y, log_det_forward

    def inverse(self, params: Any, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'log_diag'.
          y: input tensor of shape (..., dim).
          context: ignored (accepted for interface compatibility).

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det ∂x/∂y| = -sum(log_diag), shape y.shape[:-1].
        """
        del context  # Unused in LinearTransform.
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {y.shape[-1]}"
            )

        L, U, s = self._reconstruct_L_U_s(params)
        T = U + jnp.diag(s)

        batch_shape = y.shape[:-1]
        y_flat = y.reshape((-1, self.dim))  # (B, dim)
        u_prime = y_flat.T                  # (dim, B)

        # Column-style inverse:
        # 1) L a = u'   -> a
        # 2) T u = a    -> u
        a = jsp.solve_triangular(
            L, u_prime, lower=True, unit_diagonal=True
        )
        u = jsp.solve_triangular(
            T, a, lower=False
        )

        x_flat = u.T                        # (B, dim)
        x = x_flat.reshape(batch_shape + (self.dim,))

        log_det_scalar = jnp.sum(jnp.log(s))  # or sum(log_diag)
        log_det_inverse = jnp.broadcast_to(-log_det_scalar, batch_shape)
        return x, log_det_inverse

    def init_params(self, key: PRNGKey) -> dict:
        """
        Initialize parameters for this transform.

        Returns identity transform params (L=I, U=0, s=1 => W=I).

        Arguments:
            key: JAX PRNGKey (unused, included for interface consistency).

        Returns:
            Dict with keys 'lower', 'upper', 'log_diag'.
        """
        del key  # Unused for deterministic identity init.
        return {
            "lower": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "upper": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "log_diag": jnp.zeros((self.dim,), dtype=jnp.float32),
        }

    @classmethod
    def create(cls, key: PRNGKey, dim: int) -> Tuple["LinearTransform", dict]:
        """
        Factory method to create LinearTransform and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Dimensionality of the transform.

        Returns:
            Tuple of (transform, params) ready to use.

        Raises:
            ValueError: If dim <= 0.

        Example:
            >>> transform, params = LinearTransform.create(key, dim=4)
            >>> y, log_det = transform.forward(params, x)
        """
        if dim <= 0:
            raise ValueError(f"LinearTransform.create: dim must be positive, got {dim}.")

        transform = cls(dim=dim)
        params = transform.init_params(key)
        return transform, params


# ===================================================================
# Affine Coupling Layer
# ===================================================================
@dataclass
class AffineCoupling:
    """
    RealNVP-style affine coupling layer.

    This layer splits the input vector x into two parts using a binary mask m.
    Masked dimensions (where m = 1) pass through unchanged. Unmasked dimensions
    (where m = 0) are transformed using parameters produced by a conditioner
    network.
    
    It roughly works as follows:
    * Split x into x1 = x * m and x2 = x * (1 - m)
    * Use x1 as input to a conditioner network to produce shift and log_scale for transforming x2.
    * Apply the elementwise affine transform on x2: y2 = x2 * exp(log_scale) + shift.
      Note that the shift and log_scale are zeroed out on the masked dimensions.
    * Combine y1 = x1 and y2 to produce output y = y1 + y2. This only modifies the unmasked dimensions.

    Forward transformation y = T(x):
    x1 = x * m
    x2 = x * (1 - m)
    (shift, log_scale) = conditioner(x1) * (1 - m)
    y1 = x1
    y2 = (x2 * exp(log_scale) + shift) 
    y = y1 + y2
    The returned log_det is log |det ∂y/∂x|, equal to the sum of log_scale on the
    unmasked coordinates.

    Inverse transformation x = T^{-1}(y):
    y1 = y * m
    y2 = y * (1 - m)
    (shift, log_scale) = conditioner(y1) * (1 - m)
    x2 = (y2 - shift) * exp(-log_scale)
    x = y1 + x2
    The returned log_det is log |det ∂x/∂y| = -sum(log_scale).

    Parameters:
    params["mlp"]: PyTree containing the Flax parameters of the conditioner.

    All operations act along the last dimension. The mask must be one-dimensional
    with the same length as the feature dimension.
    
    Note:
    In this implementation, the conditioner network is typically initialized
    such that its output is identically zero at initialization. In that case,
    shift = 0 and log_scale = 0, so this layer is exactly the identity map
    at the start of training.
    
    Conditional flows:
      The optional `context` argument enables conditional density estimation p(x|c).
      When provided, context is concatenated to the masked input before being passed
      to the conditioner network. The conditioner MLP must be initialized with
      `context_dim` matching the size of the context vector.

      Context shape: (batch, context_dim) or (context_dim,) for a single sample.
      The same context is used for all coupling layers in a flow.

    References:
      - Dinh, Krueger, Bengio (2017). "NICE: Non-linear Independent Components Estimation"
      - Dinh, Sohl-Dickstein, Bengio (2017). "Density estimation using Real NVP"
    """
    mask: Array          # shape (dim,), values 0 or 1
    conditioner: MLP     # Flax MLP module (definition, no params inside)
    max_log_scale: float = 1.0
    max_shift: float | None = None  # Default: exp(max_log_scale)

    def __post_init__(self):
        # Ensure mask is a 1D array.
        self.mask = jnp.asarray(self.mask)
        if self.mask.ndim != 1:
            raise ValueError(
                f"AffineCoupling mask must be 1D, got shape {self.mask.shape}."
            )
        # Validate conditioner interface.
        validate_conditioner(self.conditioner, name="AffineCoupling.conditioner")

    @property
    def dim(self) -> int:
        return int(self.mask.shape[0])

    @staticmethod
    def required_out_dim(dim: int) -> int:
        """
        Return required conditioner output dimension for AffineCoupling.

        The conditioner must output shift and log_scale for each dimension,
        so out_dim = 2 * dim.

        Arguments:
            dim: Input/output dimensionality.

        Returns:
            Required output dimension for conditioner (2 * dim).
        """
        return 2 * dim

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        max_log_scale: float = 1.0,
        max_shift: float | None = None,
    ) -> Tuple["AffineCoupling", dict]:
        """
        Factory method to create AffineCoupling with properly configured MLP.

        This handles the output dimension calculation internally and initializes
        parameters, returning both the coupling and its params ready to use.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Input/output dimensionality.
            mask: Binary mask of shape (dim,). 1 = frozen, 0 = transformed.
            hidden_dim: Width of hidden layers in conditioner MLP.
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            context_dim: Context dimension (0 for unconditional).
            activation: Activation function for MLP (default: elu).
            res_scale: Residual connection scale (default: 0.1).
            max_log_scale: Bound on |log_scale| via tanh (default: 1.0).
            max_shift: Bound on |shift| via tanh (default: exp(max_log_scale)).

        Returns:
            Tuple of (coupling, params) ready to use.

        Raises:
            ValueError: If mask length doesn't match dim, or dim <= 0.

        Example:
            >>> coupling, params = AffineCoupling.create(
            ...     key, dim=4, mask=jnp.array([1, 0, 1, 0]),
            ...     hidden_dim=64, n_hidden_layers=2
            ... )
            >>> y, log_det = coupling.forward(params, x)
        """
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"AffineCoupling.create: dim must be positive, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"AffineCoupling.create: hidden_dim must be positive, got {hidden_dim}.")

        mask = jnp.asarray(mask)
        if mask.shape != (dim,):
            raise ValueError(
                f"AffineCoupling.create: mask shape {mask.shape} doesn't match (dim,) = ({dim},)."
            )

        # Create MLP with correct output dimension
        out_dim = cls.required_out_dim(dim)
        mlp = MLP(
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Create coupling
        coupling = cls(
            mask=mask,
            conditioner=mlp,
            max_log_scale=max_log_scale,
            max_shift=max_shift,
        )

        # Initialize params
        params = coupling.init_params(key)

        return coupling, params

    def _condition(self, params: dict, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Run the conditioner network and produce shift and log_scale.

        params:
          dict with key "mlp" containing the conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor of shape (..., context_dim) or (context_dim,).
        """
        if "mlp" not in params:
            raise KeyError(
                "AffineCoupling expected params to contain key 'mlp'."
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"AffineCoupling expected input with last dimension {self.dim}, "
                f"got {x.shape[-1]}."
            )

        # Use only the masked part as input to the conditioner.
        # Broadcasting: mask has shape (dim,), x has shape (..., dim).
        x_masked = x * self.mask

        # Apply the MLP. We expect output of size 2 * dim
        # which we split into shift and log_scale_raw.
        mlp_params = params["mlp"]
        out = self.conditioner.apply({"params": mlp_params}, x_masked, context)

        if out.shape[-1] != 2 * self.dim:
            raise ValueError(
                f"Conditioner output last dimension should be 2 * dim = {2 * self.dim}, "
                f"got {out.shape[-1]}."
            )

        shift, log_scale_raw = jnp.split(out, 2, axis=-1)

        # Only transform the unmasked part: zero out contributions on masked dims.
        # (1 - mask) has 1 for transformed dims, 0 otherwise.
        m_unmasked = 1.0 - self.mask

        # Bound both shift and log_scale to avoid numerical explosions.
        # Default max_shift = exp(max_log_scale) matches the maximum scale factor.
        max_shift = self.max_shift if self.max_shift is not None else jnp.exp(self.max_log_scale)
        shift = jnp.tanh(shift / max_shift) * max_shift * m_unmasked
        log_scale = jnp.tanh(log_scale_raw / self.max_log_scale) * self.max_log_scale * m_unmasked

        return shift, log_scale

    def forward(self, params: dict, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward transform: x -> y, returning (y, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det J| with shape x.shape[:-1].
        """
        shift, log_scale = self._condition(params, x, context)

        x1 = x * self.mask
        x2 = x * (1.0 - self.mask)

        y2 = x2 * jnp.exp(log_scale) + shift
        y = x1 + y2

        # Sum log_scale over transformed dimensions.
        log_det = jnp.sum(log_scale, axis=-1)
        return y, log_det

    def inverse(self, params: dict, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse transform: y -> x, returning (x, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det d x / d y| with shape y.shape[:-1].
        """
        shift, log_scale = self._condition(params, y, context)

        y1 = y * self.mask
        y2 = y * (1.0 - self.mask)

        x2 = (y2 - shift) * jnp.exp(-log_scale)
        x = y1 + x2

        # Inverse log-det is negative of forward log-det.
        log_det = -jnp.sum(log_scale, axis=-1)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int | None = None) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init to create MLP parameters. With zero-initialized final layer,
        the transform starts at identity.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension. If None, inferred from conditioner.context_dim.

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
        # Infer context_dim from conditioner if not provided
        if context_dim is None:
            context_dim = self.conditioner.context_dim

        dummy_x = jnp.zeros((1, self.dim), dtype=jnp.float32)
        dummy_context = jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]

        # Zero-init final layer for identity-start (if conditioner supports it).
        if hasattr(self.conditioner, "get_output_layer") and hasattr(self.conditioner, "set_output_layer"):
            out_layer = self.conditioner.get_output_layer(mlp_params)
            kernel = jnp.zeros_like(out_layer["kernel"])
            bias = jnp.zeros_like(out_layer["bias"])
            mlp_params = self.conditioner.set_output_layer(mlp_params, kernel, bias)

        return {"mlp": mlp_params}


# ===================================================================
# Spline Coupling Layer
# ===================================================================
@dataclass
class SplineCoupling:
    """
    RealNVP-style coupling layer with monotonic rational-quadratic splines
    (Durkan et al., 2019).
    
    It roughly works as follows:
    * Split input x into two parts using a binary mask.
    * Use the masked part as input to a conditioner network to produce spline parameters for the unmasked part.
    * Apply elementwise monotonic RQ splines on the unmasked part. The masked part remains unchanged.
    
    The spline: rational_quadratic_spline in splines.py implements the actual spline logic.
    It roughly works as follows:
    * Given K bins, the spline is defined by K widths, K heights, and K-1 internal derivatives.
    * The spline is monotonic and C^1 continuous.
    * Outside the interval [-tail_bound, tail_bound], the spline is the identity map.

    Mask semantics:
      - mask[i] == 1: dimension i is *conditioned on* (left unchanged)
      - mask[i] == 0: dimension i is *transformed* by a spline

    Conditioner:
      - A Flax module (e.g. MLP) that maps x_cond = x * mask to spline parameters.
      - Params are provided via params["mlp"].

    Parameterization per dimension (K bins):
      - widths:      K
      - heights:     K
      - derivatives: K-1  (internal knot derivatives; boundary derivatives are fixed to 1)

      => params_per_dim = 3K - 1
      => conditioner output dimension = dim * (3K - 1)

    Forward / inverse:
      - Applies elementwise monotonic RQ spline on the transformed dimensions.
      - Identity tails outside [-tail_bound, tail_bound] are handled inside splines.py.

    Conditional flows:
      The optional `context` argument enables conditional density estimation p(x|c).
      When provided, context is concatenated to the masked input before being passed
      to the conditioner network. The conditioner MLP must be initialized with
      `context_dim` matching the size of the context vector.

      Context shape: (batch, context_dim) or (context_dim,) for a single sample.
      The same context is used for all coupling layers in a flow.

    Returns:
      - output with shape (..., dim)
      - log_det with shape (...,) corresponding to forward or inverse Jacobian.
    """
    mask: Array                 # shape (dim,), values in {0, 1}
    conditioner: Any            # Flax module, called via conditioner.apply
    num_bins: int = 8
    tail_bound: float = 5.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    max_derivative: float = 10.0

    def __post_init__(self):
        self.mask = jnp.asarray(self.mask, dtype=jnp.float32)
        if self.mask.ndim != 1:
            raise ValueError(
                f"SplineCoupling: mask must be 1D, got shape {self.mask.shape}."
            )
        # Validate conditioner interface.
        validate_conditioner(self.conditioner, name="SplineCoupling.conditioner")

    @staticmethod
    def required_out_dim(dim: int, num_bins: int) -> int:
        """
        Return required conditioner output dimension for SplineCoupling.

        The conditioner must output widths (K), heights (K), and derivatives (K-1)
        for each dimension, so out_dim = dim * (3K - 1).

        Arguments:
            dim: Input/output dimensionality.
            num_bins: Number of spline bins (K).

        Returns:
            Required output dimension for conditioner: dim * (3 * num_bins - 1).
        """
        return dim * (3 * num_bins - 1)

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        max_derivative: float = 10.0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
    ) -> Tuple["SplineCoupling", dict]:
        """
        Factory method to create SplineCoupling with properly configured MLP.

        This handles the output dimension calculation internally and initializes
        parameters, returning both the coupling and its params ready to use.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Input/output dimensionality.
            mask: Binary mask of shape (dim,). 1 = frozen, 0 = transformed.
            hidden_dim: Width of hidden layers in conditioner MLP.
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            context_dim: Context dimension (0 for unconditional).
            num_bins: Number of spline bins (default: 8).
            tail_bound: Spline acts on [-B, B]; identity outside (default: 5.0).
            min_bin_width: Minimum bin width for stability (default: 1e-3).
            min_bin_height: Minimum bin height for stability (default: 1e-3).
            min_derivative: Minimum derivative for stability (default: 1e-3).
            max_derivative: Maximum derivative for stability (default: 10.0).
            activation: Activation function for MLP (default: elu).
            res_scale: Residual connection scale (default: 0.1).

        Returns:
            Tuple of (coupling, params) ready to use.

        Raises:
            ValueError: If mask length doesn't match dim, or invalid parameters.

        Example:
            >>> coupling, params = SplineCoupling.create(
            ...     key, dim=4, mask=jnp.array([1, 0, 1, 0]),
            ...     hidden_dim=64, n_hidden_layers=2, num_bins=8
            ... )
            >>> y, log_det = coupling.forward(params, x)
        """
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"SplineCoupling.create: dim must be positive, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"SplineCoupling.create: hidden_dim must be positive, got {hidden_dim}.")
        if num_bins <= 0:
            raise ValueError(f"SplineCoupling.create: num_bins must be positive, got {num_bins}.")

        mask = jnp.asarray(mask)
        if mask.shape != (dim,):
            raise ValueError(
                f"SplineCoupling.create: mask shape {mask.shape} doesn't match (dim,) = ({dim},)."
            )

        # Create MLP with correct output dimension
        out_dim = cls.required_out_dim(dim, num_bins)
        mlp = MLP(
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Create coupling
        coupling = cls(
            mask=mask,
            conditioner=mlp,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
        )

        # Initialize params
        params = coupling.init_params(key)

        return coupling, params

    def _conditioner_params(self, params: Any) -> Any:
        try:
            return params["mlp"]
        except Exception as e:
            raise KeyError("SplineCoupling expected params to contain key 'mlp'.") from e

    def _check_x(self, x: Array) -> int:
        if x.ndim < 1:
            raise ValueError(
                f"SplineCoupling expected input with at least 1 dimension, got {x.shape}."
            )
        dim = x.shape[-1]
        if self.mask.shape != (dim,):
            raise ValueError(
                f"SplineCoupling: mask shape {self.mask.shape} does not match "
                f"input dim {dim}."
            )
        if self.num_bins < 1:
            raise ValueError(
                f"SplineCoupling: num_bins must be >= 1, got {self.num_bins}."
            )
        return dim

    def _compute_spline_params(self, mlp_params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array, Array]:
        """
        Compute raw spline parameters from the conditioner and reshape them to:
          widths:      (..., dim, K)
          heights:     (..., dim, K)
          derivatives: (..., dim, K-1)

        Arguments:
          mlp_params: parameters for the conditioner network.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
        """
        dim = x.shape[-1]
        K = self.num_bins
        params_per_dim = 3 * K - 1
        expected_out_dim = dim * params_per_dim

        # Conditioner sees only the masked (conditioning) part.
        x_cond = x * self.mask

        theta = self.conditioner.apply({"params": mlp_params}, x_cond, context)  # (..., expected_out_dim)
        if theta.shape[-1] != expected_out_dim:
            raise ValueError(
                "SplineCoupling: conditioner output has wrong size. "
                f"Expected last dim {expected_out_dim}, got {theta.shape[-1]}."
            )

        theta = theta.reshape(theta.shape[:-1] + (dim, params_per_dim))  # (..., dim, 3K-1)

        widths = theta[..., :K]                 # (..., dim, K)
        heights = theta[..., K : 2 * K]         # (..., dim, K)
        derivatives = theta[..., 2 * K :]       # (..., dim, K-1)

        return widths, heights, derivatives

    def _apply_splines(self, x: Array, widths: Array, heights: Array, derivatives: Array, inverse: bool) -> Tuple[Array, Array]:
        """
        Apply scalar splines per dimension using vmap.

        Returns:
          y: (..., dim)
          logabsdet_per_dim: (..., dim)
        """
        K = self.num_bins  # for readability only

        def per_dim_fn(x_d: Array, w_d: Array, h_d: Array, d_d: Array) -> Tuple[Array, Array]:
            return rational_quadratic_spline(
                inputs=x_d,
                unnormalized_widths=w_d,
                unnormalized_heights=h_d,
                unnormalized_derivatives=d_d,
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                max_derivative=self.max_derivative,
                inverse=inverse,
            )

        # vmap over the feature dimension:
        #   x: (..., dim)          -> map axis -1
        #   widths/heights: (..., dim, K)   -> map axis -2
        #   derivatives:    (..., dim, K-1) -> map axis -2
        y, logabsdet = jax.vmap(
            per_dim_fn,
            in_axes=(-1, -2, -2, -2),
            out_axes=(-1, -1),
        )(x, widths, heights, derivatives)

        return y, logabsdet

    def forward(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward map: x -> y with log_det_forward = log|det ∂y/∂x|.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
        """
        self._check_x(x)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(mlp_params, x, context)
        y_spline, logabsdet_per_dim = self._apply_splines(
            x, widths, heights, derivatives, inverse=False
        )

        # Only transform unmasked dims; masked dims stay identity.
        inv_mask = 1.0 - self.mask
        y = x * self.mask + y_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return y, log_det

    def inverse(self, params: Any, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x with log_det_inverse = log|det ∂x/∂y|.

        Note: the conditioner depends only on the masked (unchanged) subset.
        Since masked dimensions are copied through exactly, y * mask == x * mask.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          y: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
        """
        self._check_x(y)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(mlp_params, y, context)
        x_spline, logabsdet_per_dim = self._apply_splines(
            y, widths, heights, derivatives, inverse=True
        )

        inv_mask = 1.0 - self.mask
        x = y * self.mask + x_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return x, log_det

    def _compute_identity_spline_bias(self, out_dim: int) -> Array:
        """
        Compute bias values for near-identity spline initialization.

        Sets biases such that:
          - widths/heights logits → 0 (uniform bins after softmax)
          - derivatives → ~1 (via inverse sigmoid)

        Arguments:
            out_dim: Output dimension of the conditioner.

        Returns:
            Bias array of shape (out_dim,).
        """
        dim = int(self.mask.shape[0])
        K = self.num_bins
        params_per_dim = 3 * K - 1

        new_bias = jnp.zeros((dim, params_per_dim), dtype=jnp.float32)

        # For derivatives: min + (max-min)*sigmoid(u0) ≈ 1
        lo = float(self.min_derivative)
        hi = float(self.max_derivative)
        if lo < 1.0 < hi:
            alpha = (1.0 - lo) / (hi - lo)
            u0 = stable_logit(jnp.asarray(alpha, dtype=new_bias.dtype))
            new_bias = new_bias.at[:, 2 * K:].set(u0)

        return new_bias.reshape((out_dim,))

    def _patch_dense_out(self, mlp_params: Any) -> Any:
        """
        Patch MLP final layer for near-identity spline initialization.

        Sets kernel to zero and biases for identity spline.

        Arguments:
            mlp_params: MLP parameter dict.

        Returns:
            Patched MLP parameters.

        Raises:
            RuntimeError: If conditioner lacks get_output_layer/set_output_layer methods.
        """
        if not (hasattr(self.conditioner, "get_output_layer") and
                hasattr(self.conditioner, "set_output_layer")):
            raise RuntimeError(
                "SplineCoupling._patch_dense_out: conditioner must implement "
                "get_output_layer() and set_output_layer() methods."
            )

        out_layer = self.conditioner.get_output_layer(mlp_params)
        new_kernel = jnp.zeros_like(out_layer["kernel"])
        new_bias = self._compute_identity_spline_bias(out_layer["bias"].shape[0])

        return self.conditioner.set_output_layer(mlp_params, new_kernel, new_bias)

    def init_params(self, key: PRNGKey, context_dim: int | None = None) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init + patches final layer for near-identity spline init.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension. If None, inferred from conditioner.context_dim.

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
        # Infer context_dim from conditioner if not provided
        if context_dim is None:
            context_dim = self.conditioner.context_dim

        dim = int(self.mask.shape[0])
        dummy_x = jnp.zeros((1, dim), dtype=jnp.float32)
        dummy_context = jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]
        mlp_params = self._patch_dense_out(mlp_params)
        return {"mlp": mlp_params}


# ===================================================================
# Permutation Transform: Fixed permutation of dimensions
# ===================================================================
@dataclass
class Permutation:
    """
    Permutation of feature dimensions.

    Forward transform y = T(x):
    y[..., i] = x[..., perm[i]]
    Inverse transform x = T⁻¹(y):
    x[..., perm[i]] = y[..., i]

    Both the forward and inverse Jacobians are permutation matrices with unit
    determinant. The returned log_det is therefore always zero.

    perm must be a one-dimensional integer array of shape (dim,). An inverse
    permutation is precomputed at construction.
    """
    perm: Array  # integer indices, shape (dim,)

    def __post_init__(self):
        self.perm = jnp.asarray(self.perm)
        if self.perm.ndim != 1:
            raise ValueError(
                f"Permutation perm must be 1D, got shape {self.perm.shape}."
            )
        if not jnp.issubdtype(self.perm.dtype, jnp.integer):
            raise TypeError(
                f"Permutation perm must be integer dtype, got {self.perm.dtype}."
            )

        dim = self.perm.shape[0]
        # Precompute inverse permutation: inv_perm[perm[i]] = i
        inv_perm = jnp.empty_like(self.perm)
        inv_perm = inv_perm.at[self.perm].set(jnp.arange(dim))
        self._inv_perm = inv_perm

    @property
    def dim(self) -> int:
        return int(self.perm.shape[0])

    def forward(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward permutation: x -> y.

        params:
          ignored (kept for interface compatibility).
        x:
          input tensor of shape (..., dim).
        context:
          ignored (accepted for interface compatibility).
        """
        del context  # Unused in Permutation.
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Permutation expected input with last dimension {self.dim}, "
                f"got {x.shape[-1]}."
            )

        y = x[..., self.perm]
        log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        return y, log_det

    def inverse(self, params: Any, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse permutation: y -> x.

        params:
          ignored.
        y:
          input tensor of shape (..., dim).
        context:
          ignored (accepted for interface compatibility).
        """
        del context  # Unused in Permutation.
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"Permutation expected input with last dimension {self.dim}, "
                f"got {y.shape[-1]}."
            )

        x = y[..., self._inv_perm]
        log_det = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        return x, log_det

    def init_params(self, key: PRNGKey) -> dict:
        """
        Initialize parameters for this transform.

        Permutation has no learnable parameters.

        Arguments:
            key: JAX PRNGKey (unused).

        Returns:
            Empty dict.
        """
        del key  # Unused.
        return {}

    @classmethod
    def create(cls, key: PRNGKey, perm: Array) -> Tuple["Permutation", dict]:
        """
        Factory method to create Permutation and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization (unused, for consistency).
            perm: Permutation indices of shape (dim,).

        Returns:
            Tuple of (transform, params) ready to use.

        Example:
            >>> perm = jnp.array([3, 2, 1, 0])  # reverse permutation
            >>> transform, params = Permutation.create(key, perm=perm)
            >>> y, log_det = transform.forward(params, x)
        """
        del key  # Unused for Permutation.
        transform = cls(perm=perm)
        params = transform.init_params(None)  # type: ignore
        return transform, params


# ===================================================================
# Composite Transform: Sequential composition of multiple transforms
# ===================================================================
@dataclass
class CompositeTransform:
    """
    Sequential composition of multiple transforms.

    Given transforms T_1, T_2, ..., T_n, this object represents the composite
    mapping:
    T(x) = T_n(... T_2(T_1(x)) ...)

    Each block must implement forward(params, x) and inverse(params, y), returning
    the output and the corresponding log-Jacobian determinant.

    Forward propagation:
    y = x
    log_det_total = sum_i log |det ∂T_i/∂(input_i)|
    where the blocks are applied in their listed order.

    Inverse propagation:
    x = y
    log_det_total = sum_i log |det ∂T_i⁻¹/∂(output_i)|
    where the blocks are applied in reverse order.

    Parameters must be a sequence whose length matches that of blocks, where the
    i-th entry contains the parameter PyTree for the i-th transform.
    """
    blocks: List[Any]  # list of AffineCoupling, Permutation, etc

    def forward(self, params: Sequence[Any], x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward composition: x -> y, applying blocks in order.

        params:
          sequence of parameter objects, one per block.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: sum of all block log-dets, shape x.shape[:-1].
        """
        if len(params) != len(self.blocks):
            raise ValueError(
                f"CompositeTransform expected {len(self.blocks)} param sets, "
                f"got {len(params)}."
            )

        y = x
        # Use float64 for log-det accumulation to avoid precision loss in deep flows.
        # Only use float64 if JAX x64 mode is enabled, otherwise fall back silently.
        use_f64 = jax.config.read("jax_enable_x64")
        accum_dtype = jnp.float64 if use_f64 else x.dtype
        log_det_total = jnp.zeros(x.shape[:-1], dtype=accum_dtype)

        for block, p in zip(self.blocks, params):
            y, log_det = block.forward(p, y, context)
            log_det_total = log_det_total + log_det.astype(accum_dtype)

        return y, log_det_total.astype(x.dtype)

    def inverse(self, params: Sequence[Any], y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse composition: y -> x, applying blocks in reverse order.

        params:
          sequence of parameter objects, one per block (same order as forward).
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: sum of all block inverse log-dets, shape y.shape[:-1].
        """
        if len(params) != len(self.blocks):
            raise ValueError(
                f"CompositeTransform expected {len(self.blocks)} param sets, "
                f"got {len(params)}."
            )

        x = y
        # Use float64 for log-det accumulation to avoid precision loss in deep flows.
        # Only use float64 if JAX x64 mode is enabled, otherwise fall back silently.
        use_f64 = jax.config.read("jax_enable_x64")
        accum_dtype = jnp.float64 if use_f64 else y.dtype
        log_det_total = jnp.zeros(y.shape[:-1], dtype=accum_dtype)

        # Reverse both blocks and parameter sequence.
        for block, p in zip(reversed(self.blocks), reversed(params)):
            x, log_det = block.inverse(p, x, context)
            log_det_total = log_det_total + log_det.astype(accum_dtype)

        return x, log_det_total.astype(y.dtype)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> list:
        """
        Initialize parameters for all blocks in this composite transform.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            List of parameter dicts, one per block.
        """
        keys = jax.random.split(key, len(self.blocks))
        params = []
        for k, block in zip(keys, self.blocks):
            if hasattr(block, "init_params"):
                sig = inspect.signature(block.init_params)
                if "context_dim" in sig.parameters:
                    p = block.init_params(k, context_dim=context_dim)
                else:
                    p = block.init_params(k)
                params.append(p)
            else:
                params.append({})
        return params


# ===================================================================
# LOFT Transform: Coordinate-wise log-soft extension
# ===================================================================
@dataclass
class LoftTransform:
    """
    Coordinate-wise LOFT (log soft extension) transform. It is used to
    stabilize training of normalizing flows in high-dimensional settings.
    This prevents numerical issues arising from extremely small or large
    log-densities in high dimensions by modifying the tails of the
    transformation to be logarithmic instead of linear beyond a threshold.

    Parameters
    ----------
    dim : int
        Feature dimension (size of the last axis).
    tau : float
        Positive threshold where the behavior transitions from linear to
        logarithmic tails.

    Notes
    -----
    - This transform is strictly monotone and C^1 for tau > 0.
    - params is currently unused, kept only for interface compatibility.
      If you later want a learnable tau, you can route it through params.
      
    References
    ----------
    "STABLE TRAINING OF NORMALIZING FLOWS FOR HIGH-DIMENSIONAL VARIATIONAL INFERENCE" by DANIEL ANDRADE
    """
    dim: int
    tau: float

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError(
                f"LoftTransform: dim must be positive, got {self.dim}."
            )
        if self.tau <= 0.0:
            raise ValueError(
                f"LoftTransform: tau must be strictly positive, got {self.tau}."
            )

    def forward(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        x : Array
            Input tensor of shape (..., dim).
        context : Array | None
            Ignored (accepted for interface compatibility).

        Returns
        -------
        y : Array
            Transformed tensor of shape (..., dim).
        log_det_forward : Array
            log |det ∂y/∂x|, shape x.shape[:-1].
        """
        del context  # Unused in LoftTransform.
        x = jnp.asarray(x)

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LoftTransform: expected input last dim {self.dim}, "
                f"got {x.shape[-1]}."
            )

        # Forward LOFT (elementwise) from loft.py
        y = scalar_function.loft(x, self.tau)

        # loft_logabsdet_fn(x, tau) has same shape as x and contains
        # elementwise log |g'(x_i)|.
        log_abs_jac = scalar_function.loft_log_abs_det_jac(x, self.tau)
        # Sum over feature dimension to get per-sample log-det.
        log_det_forward = jnp.sum(log_abs_jac, axis=-1)

        return y, log_det_forward

    def inverse(self, params: Any, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        y : Array
            Input tensor of shape (..., dim).
        context : Array | None
            Ignored (accepted for interface compatibility).

        Returns
        -------
        x : Array
            Inverse-transformed tensor of shape (..., dim).
        log_det_inverse : Array
            log |det ∂x/∂y|, shape y.shape[:-1].
        """
        del context  # Unused in LoftTransform.
        y = jnp.asarray(y)

        if y.shape[-1] != self.dim:
            raise ValueError(
                f"LoftTransform: expected input last dim {self.dim}, "
                f"got {y.shape[-1]}."
            )

        # Inverse LOFT (elementwise) from loft.py
        x = scalar_function.loft_inv(y, self.tau)

        # Forward log |g'(x)| at the recovered x.
        log_abs_jac_x = scalar_function.loft_log_abs_det_jac(x, self.tau)

        # For the inverse, log |det ∂x/∂y| = - log |det ∂y/∂x|.
        log_det_inverse = -jnp.sum(log_abs_jac_x, axis=-1)

        return x, log_det_inverse

    def init_params(self, key: PRNGKey) -> dict:
        """
        Initialize parameters for this transform.

        LoftTransform has no learnable parameters.

        Arguments:
            key: JAX PRNGKey (unused).

        Returns:
            Empty dict.
        """
        del key  # Unused.
        return {}

    @classmethod
    def create(
        cls, key: PRNGKey, dim: int, tau: float = 1000.0
    ) -> Tuple["LoftTransform", dict]:
        """
        Factory method to create LoftTransform and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization (unused, for consistency).
            dim: Dimensionality of the transform.
            tau: Threshold parameter for LOFT transition (default: 1000.0).

        Returns:
            Tuple of (transform, params) ready to use.

        Raises:
            ValueError: If dim <= 0 or tau <= 0.

        Example:
            >>> transform, params = LoftTransform.create(key, dim=4, tau=5.0)
            >>> y, log_det = transform.forward(params, x)
        """
        if dim <= 0:
            raise ValueError(f"LoftTransform.create: dim must be positive, got {dim}.")
        if tau <= 0:
            raise ValueError(f"LoftTransform.create: tau must be positive, got {tau}.")

        del key  # Unused for LoftTransform.
        transform = cls(dim=dim, tau=tau)
        params = transform.init_params(None)  # type: ignore
        return transform, params