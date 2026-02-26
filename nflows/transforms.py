# nflows/transforms.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from flax import linen as nn

from .nets import MLP, Array, PRNGKey, validate_conditioner
import nflows.scalar_function as scalar_function
from .splines import rational_quadratic_spline


def _compute_gate_value(identity_gate, context):
    """
    Compute gate value from context, handling batching via vmap.

    When identity_gate(context) = 0, the transform should be the identity.
    When identity_gate(context) = 1, the transform acts normally.

    Arguments:
        identity_gate: Callable that maps context -> scalar, or None.
        context: Context tensor of shape (context_dim,) or (batch, context_dim), or None.

    Returns:
        Gate value array of shape () or (batch,), or None if identity_gate is None.

    Raises:
        ValueError: If identity_gate returns non-scalar output.
    """
    if identity_gate is None or context is None:
        return None

    # Handle single sample vs batch
    if context.ndim == 1:
        g_val = identity_gate(context)
    else:
        g_val = jax.vmap(identity_gate)(context)

    g_val = jnp.asarray(g_val)

    # Validate: should be scalar per sample
    if context.ndim == 1 and g_val.ndim > 0:
        raise ValueError(
            f"identity_gate must return scalar, got shape {g_val.shape}"
        )
    if context.ndim > 1 and g_val.ndim > 1:
        raise ValueError(
            f"identity_gate must return scalar per sample, got shape {g_val.shape}"
        )

    return g_val


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
      s = softplus(raw_diag + delta)         (positive diagonal entries)
      T = U + diag(s)                        (upper-triangular)
      W = L @ T

    Where delta is either 0 (unconditional) or produced by a conditioner
    network that takes context as input (conditional).

    For a column vector u, the forward map is:

      a = T @ u
      u' = L @ a

    For batched row-vectors x with shape (..., dim), we flatten batch
    dimensions and apply this to the last dimension.

    The Jacobian of the forward map y = x W^T has determinant det(W), and

      log |det W| = sum(log(s)) = sum(log(softplus(raw_diag + delta))).

    Inverse uses triangular solves:

      L a = u'      (forward substitution, unit diagonal)
      T u = a       (back substitution)

    Parameters (per block):
      params["lower"]:    unconstrained raw lower-tri part, shape (dim, dim)
      params["upper"]:    unconstrained raw upper-tri part, shape (dim, dim)
      params["raw_diag"]: unconstrained diagonal params, shape (dim,)
      params["mlp"]:      conditioner params (only if context_dim > 0)

    Conditional flows:
      When context_dim > 0, a conditioner MLP maps context to delta_diag,
      which is added to raw_diag before softplus. The MLP is initialized
      with zero output, so the transform starts at identity.

    This yields O(dim^2) apply / inverse and O(dim) log-det, without any
    repeated matrix factorizations inside the forward pass.
    """
    dim: int
    conditioner: MLP | None = None  # None if context_dim=0
    context_dim: int = 0

    def _get_raw_params(self, params: Any) -> Tuple[Array, Array, Array]:
        """Extract and validate raw parameters from params dict."""
        try:
            lower_raw = jnp.asarray(params["lower"])
            upper_raw = jnp.asarray(params["upper"])
            raw_diag = jnp.asarray(params["raw_diag"])
        except Exception as e:
            raise KeyError(
                "LinearTransform: params must contain 'lower', 'upper', 'raw_diag'"
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
        if raw_diag.shape != (self.dim,):
            raise ValueError(
                f"LinearTransform: raw_diag must have shape ({self.dim},), "
                f"got {raw_diag.shape}"
            )

        return lower_raw, upper_raw, raw_diag

    def _compute_diagonal(
        self,
        params: Any,
        raw_diag: Array,
        context: Array | None,
    ) -> Array:
        """
        Compute diagonal scaling s from raw_diag and optional context.

        If conditioner exists and context is provided, adds delta from conditioner.
        Uses softplus for numerical stability.

        Returns:
            s: positive diagonal scaling, shape (dim,) or (batch, dim) if batched context.
        """
        if self.conditioner is not None and context is not None:
            # Conditioner maps context -> delta_diag
            mlp_params = params["mlp"]
            # MLP with x_dim=context_dim, context_dim=0: pass context as x
            delta = self.conditioner.apply({"params": mlp_params}, context, None)
            # delta shape: (batch, dim) or (dim,)
            s = jax.nn.softplus(raw_diag + delta)
        else:
            s = jax.nn.softplus(raw_diag)
        return s

    def _forward_batched_gate(
        self,
        x: Array,
        lower_raw: Array,
        upper_raw: Array,
        s: Array,
        g_value: Array,
        batch_shape: tuple,
    ) -> Tuple[Array, Array]:
        """Forward pass with per-sample gating via vmap.

        When g_value is batched, each sample needs its own L, U matrices
        constructed with its gate value. This is slower than the shared-matrix
        path but correctly handles per-sample identity interpolation.
        """
        # Gate the diagonal: s_gated = 1 + g * (s - 1)
        g_diag = g_value[:, None]  # (B, 1)
        if s.ndim == 1:
            # s is (dim,) - broadcast to (B, dim)
            s_gated = 1.0 - g_diag + g_diag * s
        else:
            # s is already (B, dim)
            s_gated = 1.0 - g_diag + g_diag * s

        dim = self.dim
        dtype = lower_raw.dtype

        def forward_single(x_i, g_i, s_i):
            # Build gated LU factors: when g=0, L=I and U=0, so W=I (identity).
            # When g=1, we get the full learned transform.
            L_i = jnp.tril(g_i * lower_raw, k=-1) + jnp.eye(dim, dtype=dtype)
            U_i = jnp.triu(g_i * upper_raw, k=1)
            T_i = U_i + jnp.diag(s_i)
            y_i = L_i @ T_i @ x_i
            log_det_i = jnp.sum(jnp.log(s_i))
            return y_i, log_det_i

        # Flatten batch dims for vmap, then reshape back.
        x_flat = x.reshape((-1, dim))
        g_flat = g_value.reshape((-1,))
        s_flat = s_gated.reshape((-1, dim))
        y_flat, log_det_flat = jax.vmap(forward_single)(x_flat, g_flat, s_flat)
        y = y_flat.reshape(batch_shape + (dim,))
        log_det_forward = log_det_flat.reshape(batch_shape)
        return y, log_det_forward

    def _inverse_batched_gate(
        self,
        y: Array,
        lower_raw: Array,
        upper_raw: Array,
        s: Array,
        g_value: Array,
        batch_shape: tuple,
    ) -> Tuple[Array, Array]:
        """Inverse pass with per-sample gating via vmap.

        When g_value is batched, each sample needs its own L, U matrices
        constructed with its gate value. This is slower than the shared-matrix
        path but correctly handles per-sample identity interpolation.
        """
        # Gate the diagonal: s_gated = 1 + g * (s - 1)
        g_diag = g_value[:, None]  # (B, 1)
        if s.ndim == 1:
            # s is (dim,) - broadcast to (B, dim)
            s_gated = 1.0 - g_diag + g_diag * s
        else:
            # s is already (B, dim)
            s_gated = 1.0 - g_diag + g_diag * s

        dim = self.dim
        dtype = lower_raw.dtype

        def inverse_single(y_i, g_i, s_i):
            # Build gated LU factors (same as forward).
            L_i = jnp.tril(g_i * lower_raw, k=-1) + jnp.eye(dim, dtype=dtype)
            U_i = jnp.triu(g_i * upper_raw, k=1)
            T_i = U_i + jnp.diag(s_i)
            # Solve L @ T @ x = y via two triangular solves.
            a_i = jsp.solve_triangular(L_i, y_i, lower=True, unit_diagonal=True)
            x_i = jsp.solve_triangular(T_i, a_i, lower=False)
            log_det_i = -jnp.sum(jnp.log(s_i))
            return x_i, log_det_i

        # Flatten batch dims for vmap, then reshape back.
        y_flat = y.reshape((-1, dim))
        g_flat = g_value.reshape((-1,))
        s_flat = s_gated.reshape((-1, dim))
        x_flat, log_det_flat = jax.vmap(inverse_single)(y_flat, g_flat, s_flat)
        x = x_flat.reshape(batch_shape + (dim,))
        log_det_inverse = log_det_flat.reshape(batch_shape)
        return x, log_det_inverse

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'raw_diag', and optionally 'mlp'.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det ∂y/∂x| = sum(log(s)), shape x.shape[:-1].
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {x.shape[-1]}"
            )

        # Get raw parameters with validation
        lower_raw, upper_raw, raw_diag = self._get_raw_params(params)
        batch_shape = x.shape[:-1]

        # Compute diagonal scaling (context-dependent if conditioner exists)
        s = self._compute_diagonal(params, raw_diag, context)  # shape (dim,) or (batch, dim)

        # Batched gate requires per-sample L, U - use dedicated vmap path
        if g_value is not None and g_value.ndim > 0:
            return self._forward_batched_gate(
                x, lower_raw, upper_raw, s, g_value, batch_shape
            )

        # Fast path: shared L, U (possibly scaled by scalar gate)
        if g_value is not None:
            # Scalar gate - scale L, U and interpolate s
            lower_raw = g_value * lower_raw
            upper_raw = g_value * upper_raw
            s = 1.0 - g_value + g_value * s

        # Reconstruct L, U
        L = jnp.tril(lower_raw, k=-1) + jnp.eye(self.dim, dtype=lower_raw.dtype)
        U = jnp.triu(upper_raw, k=1)

        # Handle batched s (when context is batched)
        if s.ndim == 1:
            # s is (dim,) - shared across batch
            T = U + jnp.diag(s)
            x_flat = x.reshape((-1, self.dim))  # (B, dim)
            u = x_flat.T                        # (dim, B)
            a = T @ u
            u_prime = L @ a
            y_flat = u_prime.T                  # (B, dim)
            y = y_flat.reshape(batch_shape + (self.dim,))
            # log |det W| = sum(log(s))
            log_det_scalar = jnp.sum(jnp.log(s))
            log_det_forward = jnp.broadcast_to(log_det_scalar, batch_shape)
        else:
            # s is (batch, dim) - different per sample, use vmap
            def forward_single(x_i, s_i):
                T_i = U + jnp.diag(s_i)
                a_i = T_i @ x_i
                y_i = L @ a_i
                log_det_i = jnp.sum(jnp.log(s_i))
                return y_i, log_det_i

            x_flat = x.reshape((-1, self.dim))
            s_flat = s.reshape((-1, self.dim))
            y_flat, log_det_flat = jax.vmap(forward_single)(x_flat, s_flat)
            y = y_flat.reshape(batch_shape + (self.dim,))
            log_det_forward = log_det_flat.reshape(batch_shape)

        return y, log_det_forward

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'raw_diag', and optionally 'mlp'.
          y: input tensor of shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det ∂x/∂y| = -sum(log(s)), shape y.shape[:-1].
        """
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {y.shape[-1]}"
            )

        # Get raw parameters with validation
        lower_raw, upper_raw, raw_diag = self._get_raw_params(params)
        batch_shape = y.shape[:-1]

        # Compute diagonal scaling (context-dependent if conditioner exists)
        s = self._compute_diagonal(params, raw_diag, context)  # shape (dim,) or (batch, dim)

        # Batched gate requires per-sample L, U - use dedicated vmap path
        if g_value is not None and g_value.ndim > 0:
            return self._inverse_batched_gate(
                y, lower_raw, upper_raw, s, g_value, batch_shape
            )

        # Fast path: shared L, U (possibly scaled by scalar gate)
        if g_value is not None:
            # Scalar gate - scale L, U and interpolate s
            lower_raw = g_value * lower_raw
            upper_raw = g_value * upper_raw
            s = 1.0 - g_value + g_value * s

        # Reconstruct L, U
        L = jnp.tril(lower_raw, k=-1) + jnp.eye(self.dim, dtype=lower_raw.dtype)
        U = jnp.triu(upper_raw, k=1)

        # Handle batched s (when context is batched)
        if s.ndim == 1:
            # s is (dim,) - shared across batch
            T = U + jnp.diag(s)
            y_flat = y.reshape((-1, self.dim))  # (B, dim)
            u_prime = y_flat.T                  # (dim, B)

            # Column-style inverse:
            # 1) L a = u'   -> a
            # 2) T u = a    -> u
            a = jsp.solve_triangular(L, u_prime, lower=True, unit_diagonal=True)
            u = jsp.solve_triangular(T, a, lower=False)

            x_flat = u.T
            x = x_flat.reshape(batch_shape + (self.dim,))
            log_det_scalar = jnp.sum(jnp.log(s))
            log_det_inverse = jnp.broadcast_to(-log_det_scalar, batch_shape)
        else:
            # s is (batch, dim) - different per sample, use vmap
            def inverse_single(y_i, s_i):
                T_i = U + jnp.diag(s_i)
                a_i = jsp.solve_triangular(L, y_i, lower=True, unit_diagonal=True)
                x_i = jsp.solve_triangular(T_i, a_i, lower=False)
                log_det_i = -jnp.sum(jnp.log(s_i))
                return x_i, log_det_i

            y_flat = y.reshape((-1, self.dim))
            s_flat = s.reshape((-1, self.dim))
            x_flat, log_det_flat = jax.vmap(inverse_single)(y_flat, s_flat)
            x = x_flat.reshape(batch_shape + (self.dim,))
            log_det_inverse = log_det_flat.reshape(batch_shape)

        return x, log_det_inverse

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Returns identity transform params (L=I, U=0, s=1 => W=I).
        For softplus parametrization, raw_diag is initialized so softplus(raw_diag) = 1.

        Arguments:
            key: JAX PRNGKey for conditioner initialization.
            context_dim: Context dimension (must match self.context_dim).

        Returns:
            Dict with keys 'lower', 'upper', 'raw_diag', and 'mlp' if context_dim > 0.
        """
        # softplus(x) = 1 when x = log(e - 1) ≈ 0.541
        raw_diag_init = jnp.full((self.dim,), jnp.log(jnp.e - 1), dtype=jnp.float32)

        params = {
            "lower": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "upper": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "raw_diag": raw_diag_init,
        }

        # Initialize conditioner if present
        if self.conditioner is not None:
            dummy_context = jnp.zeros((1, self.context_dim), dtype=jnp.float32)
            variables = self.conditioner.init(key, dummy_context, None)
            mlp_params = variables["params"]

            # Zero-init output layer so delta=0 at init => identity transform
            if hasattr(self.conditioner, "get_output_layer") and hasattr(self.conditioner, "set_output_layer"):
                out_layer = self.conditioner.get_output_layer(mlp_params)
                kernel = jnp.zeros_like(out_layer["kernel"])
                bias = jnp.zeros_like(out_layer["bias"])
                mlp_params = self.conditioner.set_output_layer(mlp_params, kernel, bias)

            params["mlp"] = mlp_params

        return params

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        activation: Callable[[Array], Array] = nn.tanh,
        res_scale: float = 0.1,
    ) -> Tuple["LinearTransform", dict]:
        """
        Factory method to create LinearTransform and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Dimensionality of the transform.
            context_dim: Context dimension (0 for unconditional).
            hidden_dim: Width of hidden layers in conditioner MLP (if context_dim > 0).
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            activation: Activation function for conditioner MLP.
            res_scale: Residual connection scale for conditioner MLP.

        Returns:
            Tuple of (transform, params) ready to use.

        Raises:
            ValueError: If dim <= 0 or context_dim < 0.

        Example:
            >>> # Unconditional
            >>> transform, params = LinearTransform.create(key, dim=4)
            >>> y, log_det = transform.forward(params, x)

            >>> # Conditional on context
            >>> transform, params = LinearTransform.create(
            ...     key, dim=4, context_dim=8, hidden_dim=64, n_hidden_layers=2
            ... )
            >>> y, log_det = transform.forward(params, x, context)
        """
        if dim <= 0:
            raise ValueError(f"LinearTransform.create: dim must be positive, got {dim}.")
        if context_dim < 0:
            raise ValueError(f"LinearTransform.create: context_dim must be non-negative, got {context_dim}.")

        # Create conditioner if context_dim > 0
        conditioner = None
        if context_dim > 0:
            if hidden_dim <= 0:
                raise ValueError(f"LinearTransform.create: hidden_dim must be positive, got {hidden_dim}.")
            # MLP with x_dim=context_dim, context_dim=0: context goes in x slot
            conditioner = MLP(
                x_dim=context_dim,
                context_dim=0,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                out_dim=dim,  # output delta for each diagonal entry
                activation=activation,
                res_scale=res_scale,
            )

        transform = cls(dim=dim, conditioner=conditioner, context_dim=context_dim)
        params = transform.init_params(key, context_dim=context_dim)
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
        params = coupling.init_params(key, context_dim=context_dim)

        return coupling, params

    def _condition(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Run the conditioner network and produce shift and log_scale.

        params:
          dict with key "mlp" containing the conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor of shape (..., context_dim) or (context_dim,).
        g_value:
          optional gate value from identity_gate(context). When g_value=0, the
          transform should be identity, so shift and log_scale are zeroed.
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

        # Apply identity gate: when g_value=0, shift=0 and log_scale=0 => identity.
        if g_value is not None:
            g = g_value[..., None]  # broadcast to (..., 1) for element-wise multiply
            shift = g * shift
            log_scale = g * log_scale

        return shift, log_scale

    def forward(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward transform: x -> y, returning (y, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det J| with shape x.shape[:-1].
        """
        shift, log_scale = self._condition(params, x, context, g_value=g_value)

        x1 = x * self.mask
        x2 = x * (1.0 - self.mask)

        y2 = x2 * jnp.exp(log_scale) + shift
        y = x1 + y2

        # Sum log_scale over transformed dimensions.
        log_det = jnp.sum(log_scale, axis=-1)
        return y, log_det

    def inverse(
        self,
        params: dict,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse transform: y -> x, returning (x, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det d x / d y| with shape y.shape[:-1].
        """
        shift, log_scale = self._condition(params, y, context, g_value=g_value)

        y1 = y * self.mask
        y2 = y * (1.0 - self.mask)

        x2 = (y2 - shift) * jnp.exp(-log_scale)
        x = y1 + x2

        # Inverse log-det is negative of forward log-det.
        log_det = -jnp.sum(log_scale, axis=-1)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init to create MLP parameters. With zero-initialized final layer,
        the transform starts at identity.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
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

        # Warn if identity-like initialization is not possible.
        lo, hi = float(self.min_derivative), float(self.max_derivative)
        if not (lo < 1.0 < hi):
            warnings.warn(
                f"SplineCoupling: derivative range [{lo}, {hi}] excludes 1.0; "
                "identity-like initialization not possible, using midpoint derivative.",
                stacklevel=2
            )

        # Precompute the identity derivative logit for gating.
        # When gated to zero, derivatives should interpolate to this value
        # so that the spline derivative equals 1 (identity behavior).
        if lo < 1.0 < hi:
            alpha = (1.0 - lo) / (hi - lo)
            self._identity_deriv_logit = stable_logit(jnp.array(alpha))
        else:
            # If 1.0 is outside the range, use midpoint (cannot achieve identity)
            self._identity_deriv_logit = jnp.array(0.0)

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
        params = coupling.init_params(key, context_dim=context_dim)

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

    def _compute_spline_params(
        self,
        mlp_params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Compute raw spline parameters from the conditioner and reshape them to:
          widths:      (..., dim, K)
          heights:     (..., dim, K)
          derivatives: (..., dim, K-1)

        Arguments:
          mlp_params: parameters for the conditioner network.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value. When g_value=0, spline params are set
              to produce identity transform (uniform bins, derivative=1).
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

        # Apply identity gate: when g_value=0, interpolate to identity spline params.
        # Identity spline: widths=heights=0 (uniform bins after softmax), derivatives → 1.
        if g_value is not None:
            # g_value shape: () or (batch,). Broadcast to (..., 1, 1) for element-wise.
            g = g_value[..., None, None]  # (..., 1, 1)

            # widths/heights → 0 when g → 0 (uniform bins)
            widths = g * widths
            heights = g * heights

            # derivatives: interpolate from identity_deriv_logit (gives d=1) to learned value
            identity_d = self._identity_deriv_logit
            derivatives = (1 - g) * identity_d + g * derivatives

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

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward map: x -> y with log_det_forward = log|det ∂y/∂x|.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.
        """
        self._check_x(x)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(
            mlp_params, x, context, g_value=g_value
        )
        y_spline, logabsdet_per_dim = self._apply_splines(
            x, widths, heights, derivatives, inverse=False
        )

        # Only transform unmasked dims; masked dims stay identity.
        inv_mask = 1.0 - self.mask
        y = x * self.mask + y_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return y, log_det

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x with log_det_inverse = log|det ∂x/∂y|.

        Note: the conditioner depends only on the masked (unchanged) subset.
        Since masked dimensions are copied through exactly, y * mask == x * mask.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          y: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.
        """
        self._check_x(y)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(
            mlp_params, y, context, g_value=g_value
        )
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

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init + patches final layer for near-identity spline init.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
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

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Permutation has no learnable parameters.

        Arguments:
            key: JAX PRNGKey (unused).
            context_dim: Context dimension (unused, included for interface consistency).

        Returns:
            Empty dict.
        """
        del key, context_dim  # Unused.
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
def _block_supports_gvalue(block: Any) -> bool:
    """Check if a transform block supports the g_value parameter."""
    return isinstance(block, (AffineCoupling, SplineCoupling, LinearTransform, LoftTransform))


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

    def forward(
        self,
        params: Sequence[Any],
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward composition: x -> y, applying blocks in order.

        params:
          sequence of parameter objects, one per block.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.
        g_value:
          optional gate value for identity_gate. Passed to all sub-blocks that
          support it (couplings, linear). When g_value=0, returns identity.

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
            # Pass g_value to blocks that support it (check for keyword argument)
            if g_value is not None and _block_supports_gvalue(block):
                y, log_det = block.forward(p, y, context, g_value=g_value)
            else:
                y, log_det = block.forward(p, y, context)
            log_det_total = log_det_total + log_det.astype(accum_dtype)

        return y, log_det_total.astype(x.dtype)

    def inverse(
        self,
        params: Sequence[Any],
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse composition: y -> x, applying blocks in reverse order.

        params:
          sequence of parameter objects, one per block (same order as forward).
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.
        g_value:
          optional gate value for identity_gate. Passed to all sub-blocks that
          support it. When g_value=0, returns identity.

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
            # Pass g_value to blocks that support it
            if g_value is not None and _block_supports_gvalue(block):
                x, log_det = block.inverse(p, x, context, g_value=g_value)
            else:
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
                p = block.init_params(k, context_dim=context_dim)
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

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
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
        g_value : Array | None
            Gate value for identity gating. Shape x.shape[:-1].
            When g=0, returns identity. When g=1, returns full LOFT.

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

        # Forward LOFT (elementwise)
        y_loft = scalar_function.loft(x, self.tau)
        # Elementwise log |loft'(x_i)|
        log_abs_jac = scalar_function.loft_log_abs_det_jac(x, self.tau)

        if g_value is not None:
            g = g_value[..., None]  # (..., 1) for broadcasting over dim
            # Gated forward: y = (1-g)*x + g*loft(x)
            y = (1.0 - g) * x + g * y_loft
            # dy/dx element-wise = (1-g) + g*loft'(x)
            # loft'(x) = exp(log_abs_jac) element-wise
            loft_deriv = jnp.exp(log_abs_jac)
            gated_deriv = (1.0 - g) + g * loft_deriv
            log_det_forward = jnp.sum(jnp.log(jnp.abs(gated_deriv)), axis=-1)
        else:
            y = y_loft
            log_det_forward = jnp.sum(log_abs_jac, axis=-1)

        return y, log_det_forward

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
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
        g_value : Array | None
            Gate value for identity gating. Must match the value used in forward.

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

        if g_value is not None:
            g = g_value[..., None]  # (..., 1)
            # Solve y = (1-g)*x + g*loft(x, tau) for x via Newton iteration.
            # f(x) = (1-g)*x + g*loft(x) - y = 0
            # f'(x) = (1-g) + g*loft'(x)
            # Newton: x_{n+1} = x_n - f(x_n)/f'(x_n)
            x = y  # initial guess (exact when g=0)
            for _ in range(10):
                loft_x = scalar_function.loft(x, self.tau)
                log_jac = scalar_function.loft_log_abs_det_jac(x, self.tau)
                loft_deriv = jnp.exp(log_jac)
                f_val = (1.0 - g) * x + g * loft_x - y
                f_deriv = (1.0 - g) + g * loft_deriv
                x = x - f_val / f_deriv

            # Compute log-det at the converged x
            log_jac_x = scalar_function.loft_log_abs_det_jac(x, self.tau)
            loft_deriv_x = jnp.exp(log_jac_x)
            gated_deriv = (1.0 - g) + g * loft_deriv_x
            log_det_inverse = -jnp.sum(jnp.log(jnp.abs(gated_deriv)), axis=-1)
        else:
            x = scalar_function.loft_inv(y, self.tau)
            log_abs_jac_x = scalar_function.loft_log_abs_det_jac(x, self.tau)
            log_det_inverse = -jnp.sum(log_abs_jac_x, axis=-1)

        return x, log_det_inverse

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        LoftTransform has no learnable parameters.

        Arguments:
            key: JAX PRNGKey (unused).
            context_dim: Context dimension (unused, included for interface consistency).

        Returns:
            Empty dict.
        """
        del key, context_dim  # Unused.
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