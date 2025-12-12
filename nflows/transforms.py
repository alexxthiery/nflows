# nflows/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

from .nets import MLP, Array
import nflows.scalar_function as scalar_function
from .splines import rational_quadratic_spline

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

    def forward(self, params: Any, x: Array) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'log_diag'.
          x: input tensor of shape (..., dim).

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det ∂y/∂x| = sum(log_diag), shape x.shape[:-1].
        """
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

    def inverse(self, params: Any, y: Array) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'log_diag'.
          y: input tensor of shape (..., dim).

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det ∂x/∂y| = -sum(log_diag), shape y.shape[:-1].
        """
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
    (shift, log_scale) = conditioner(x1)
    y1 = x1
    y2 = x2 * exp(log_scale) + shift
    y = y1 + y2
    The returned log_det is log |det ∂y/∂x|, equal to the sum of log_scale on the
    unmasked coordinates.

    Inverse transformation x = T^{-1}(y):
    y1 = y * m
    y2 = y * (1 - m)
    (shift, log_scale) = conditioner(y1)
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
    
    References:
      - Dinh, Krueger, Bengio (2017). "NICE: Non-linear Independent Components Estimation"
      - Dinh, Sohl-Dickstein, Bengio (2017). "Density estimation using Real NVP"
    """
    mask: Array          # shape (dim,), values 0 or 1
    conditioner: MLP     # Flax MLP module (definition, no params inside)
    max_log_scale: float = 1.0

    def __post_init__(self):
        # Ensure mask is a 1D array.
        self.mask = jnp.asarray(self.mask)
        if self.mask.ndim != 1:
            raise ValueError(
                f"AffineCoupling mask must be 1D, got shape {self.mask.shape}."
            )

    @property
    def dim(self) -> int:
        return int(self.mask.shape[0])

    def _condition(self, params: dict, x: Array) -> Tuple[Array, Array]:
        """
        Run the conditioner network and produce shift and log_scale.

        params:
          dict with key "mlp" containing the conditioner parameters.
        x:
          input tensor of shape (..., dim).
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
        out = self.conditioner.apply({"params": mlp_params}, x_masked)

        if out.shape[-1] != 2 * self.dim:
            raise ValueError(
                f"Conditioner output last dimension should be 2 * dim = {2 * self.dim}, "
                f"got {out.shape[-1]}."
            )

        shift, log_scale_raw = jnp.split(out, 2, axis=-1)

        # Only transform the unmasked part: zero out contributions on masked dims.
        # (1 - mask) has 1 for transformed dims, 0 otherwise.
        m_unmasked = 1.0 - self.mask

        shift = shift * m_unmasked
        # Bound log_scale to avoid numerical explosions.
        log_scale = jnp.tanh(log_scale_raw / self.max_log_scale) * self.max_log_scale * m_unmasked

        return shift, log_scale

    def forward(self, params: dict, x: Array) -> Tuple[Array, Array]:
        """
        Forward transform: x -> y, returning (y, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        x:
          input tensor of shape (..., dim).

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det J| with shape x.shape[:-1].
        """
        shift, log_scale = self._condition(params, x)

        x1 = x * self.mask
        x2 = x * (1.0 - self.mask)

        y2 = x2 * jnp.exp(log_scale) + shift
        y = x1 + y2

        # Sum log_scale over transformed dimensions.
        log_det = jnp.sum(log_scale, axis=-1)
        return y, log_det

    def inverse(self, params: dict, y: Array) -> Tuple[Array, Array]:
        """
        Inverse transform: y -> x, returning (x, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        y:
          input tensor of shape (..., dim).

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det d x / d y| with shape y.shape[:-1].
        """
        shift, log_scale = self._condition(params, y)

        y1 = y * self.mask
        y2 = y * (1.0 - self.mask)

        x2 = (y2 - shift) * jnp.exp(-log_scale)
        x = y1 + x2

        # Inverse log-det is negative of forward log-det.
        log_det = -jnp.sum(log_scale, axis=-1)
        return x, log_det


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

    def _compute_spline_params(self, mlp_params: Any, x: Array) -> Tuple[Array, Array, Array]:
        """
        Compute raw spline parameters from the conditioner and reshape them to:
          widths:      (..., dim, K)
          heights:     (..., dim, K)
          derivatives: (..., dim, K-1)
        """
        dim = x.shape[-1]
        K = self.num_bins
        params_per_dim = 3 * K - 1
        expected_out_dim = dim * params_per_dim

        # Conditioner sees only the masked (conditioning) part.
        x_cond = x * self.mask

        theta = self.conditioner.apply({"params": mlp_params}, x_cond)  # (..., expected_out_dim)
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

    def forward(self, params: Any, x: Array) -> Tuple[Array, Array]:
        """
        Forward map: x -> y with log_det_forward = log|det ∂y/∂x|.
        """
        self._check_x(x)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(mlp_params, x)
        y_spline, logabsdet_per_dim = self._apply_splines(
            x, widths, heights, derivatives, inverse=False
        )

        # Only transform unmasked dims; masked dims stay identity.
        inv_mask = 1.0 - self.mask
        y = x * self.mask + y_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return y, log_det

    def inverse(self, params: Any, y: Array) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x with log_det_inverse = log|det ∂x/∂y|.

        Note: the conditioner depends only on the masked (unchanged) subset.
        Since masked dimensions are copied through exactly, y * mask == x * mask.
        """
        self._check_x(y)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(mlp_params, y)
        x_spline, logabsdet_per_dim = self._apply_splines(
            y, widths, heights, derivatives, inverse=True
        )

        inv_mask = 1.0 - self.mask
        x = y * self.mask + x_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return x, log_det


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

    def forward(self, params: Any, x: Array) -> Tuple[Array, Array]:
        """
        Forward permutation: x -> y.

        params:
          ignored (kept for interface compatibility).
        x:
          input tensor of shape (..., dim).
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Permutation expected input with last dimension {self.dim}, "
                f"got {x.shape[-1]}."
            )

        y = x[..., self.perm]
        log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        return y, log_det

    def inverse(self, params: Any, y: Array) -> Tuple[Array, Array]:
        """
        Inverse permutation: y -> x.

        params:
          ignored.
        y:
          input tensor of shape (..., dim).
        """
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"Permutation expected input with last dimension {self.dim}, "
                f"got {y.shape[-1]}."
            )

        x = y[..., self._inv_perm]
        log_det = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        return x, log_det


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

    def forward(self, params: Sequence[Any], x: Array) -> Tuple[Array, Array]:
        """
        Forward composition: x -> y, applying blocks in order.

        params:
          sequence of parameter objects, one per block.
        x:
          input tensor of shape (..., dim).

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
        log_det_total = jnp.zeros(x.shape[:-1], dtype=x.dtype)

        for block, p in zip(self.blocks, params):
            y, log_det = block.forward(p, y)
            log_det_total = log_det_total + log_det

        return y, log_det_total

    def inverse(self, params: Sequence[Any], y: Array) -> Tuple[Array, Array]:
        """
        Inverse composition: y -> x, applying blocks in reverse order.

        params:
          sequence of parameter objects, one per block (same order as forward).
        y:
          input tensor of shape (..., dim).

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
        log_det_total = jnp.zeros(y.shape[:-1], dtype=y.dtype)

        # Reverse both blocks and parameter sequence.
        for block, p in zip(reversed(self.blocks), reversed(params)):
            x, log_det = block.inverse(p, x)
            log_det_total = log_det_total + log_det

        return x, log_det_total


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

    def forward(self, params: Any, x: Array) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        x : Array
            Input tensor of shape (..., dim).

        Returns
        -------
        y : Array
            Transformed tensor of shape (..., dim).
        log_det_forward : Array
            log |det ∂y/∂x|, shape x.shape[:-1].
        """
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

    def inverse(self, params: Any, y: Array) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        y : Array
            Input tensor of shape (..., dim).

        Returns
        -------
        x : Array
            Inverse-transformed tensor of shape (..., dim).
        log_det_inverse : Array
            log |det ∂x/∂y|, shape y.shape[:-1].
        """
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