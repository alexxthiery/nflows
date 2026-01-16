# nflows/builders.py
from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from .nets import MLP, init_mlp, Array, PRNGKey
from .distributions import StandardNormal, DiagNormal
from .transforms import AffineCoupling, Permutation, CompositeTransform, LinearTransform, LoftTransform, SplineCoupling
from .flows import Flow


# ====================================================================
# Mask coverage analysis
# ====================================================================
def analyze_mask_coverage(blocks, dim: int) -> None:
    """
    Check how often each *original* dimension is transformed by coupling-like
    blocks, taking permutations into account.

    A coupling-like block is any block with a 1D `mask` attribute of shape (dim,),
    where mask=1 means frozen and mask=0 means transformed.

    Raises a ValueError if any dimension is never transformed.
    """
    current_pos = jnp.arange(dim)
    transformed_counts = jnp.zeros(dim, dtype=jnp.int32)

    for block in blocks:
        mask = getattr(block, "mask", None)

        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != (dim,):
                raise ValueError(
                    f"Coupling mask shape {mask.shape} incompatible with dim={dim} "
                    f"for block type {type(block).__name__}."
                )

            transformed_positions = jnp.where(mask == 0)[0]
            transformed_original = current_pos[transformed_positions]
            transformed_counts = transformed_counts.at[transformed_original].add(1)

        elif isinstance(block, Permutation):
            current_pos = current_pos[block.perm]

        else:
            # ignore other block types (LinearTransform, LoftTransform, etc.)
            pass

    never = jnp.where(transformed_counts == 0)[0]
    if never.size > 0:
        raise ValueError(
            "\n" + "=" * 80 + "\n"
            f"Mask/permutation schedule leaves some dims never transformed: \n"
            f"indices {never.tolist()}, \n"
            f"counts={transformed_counts.tolist()} \n"
            f"Possible cause: permutation layer and parity masks cancel out. \n"
            f"\n ** Use the option: use_permutation=False or change the number of layers. ** \n"
            "\n" + "=" * 80
        )

# ====================================================================
# Mask construction utility
# ====================================================================
def _make_alternating_mask(dim: int, parity: int) -> Array:
    """
    Build a binary mask of length dim with alternating 1/0 pattern.

    parity = 0: mask = [1, 0, 1, 0, ...]
    parity = 1: mask = [0, 1, 0, 1, ...]
    """
    idx = jnp.arange(dim)
    # (idx + parity) % 2 == 0 -> positions set to 1
    mask = ((idx + parity) % 2 == 0).astype(jnp.float32)
    return mask


# ====================================================================
# RealNVP builder
# ====================================================================
def build_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    *,
    context_dim: int = 0,
    max_log_scale: float = 5.0,
    res_scale: float = 0.1,
    use_permutation: bool = False,
    use_linear: bool = False,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
) -> Tuple[Flow, Any]:
    """
    Construct a RealNVP-style flow with affine coupling layers.

    Architecture:
      - Base distribution p_base(z):
          * by default: StandardNormal(dim) with no trainable parameters.
          * if trainable_base=True and base_dist is None:
                DiagNormal(dim) with trainable loc and log_scale.
          * if base_dist is provided explicitly:
                use that object and base_params as given.
      - Transform: CompositeTransform of
          [AffineCoupling, (Permutation), AffineCoupling, (Permutation), ...]

    Parameters structure:
      params["base"]      -> parameters for the base distribution (PyTree)
      params["transform"] -> list of per-block params, same length as transform.blocks
        - for AffineCoupling blocks: {"mlp": mlp_params}
        - for Permutation blocks: {}

    Arguments:
      key: JAX PRNGKey used to initialize all MLPs.
      dim: Dimensionality of the data / latent space.
      num_layers: Number of affine coupling layers.
      hidden_dim: Width of hidden layers in conditioner MLPs.
      n_hidden_layers: Number of residual blocks in conditioner MLPs.
      context_dim: Dimensionality of the conditioning variable. If 0 (default),
                   the flow is unconditional. Context is concatenated to the
                   conditioner input.
      max_log_scale: Bound on |log_scale| via tanh to stabilize training.
      res_scale: Scale factor for residual connections in conditioner MLPs.
      use_permutation: If True, insert a fixed reverse permutation between
                       successive coupling layers.
      use_linear: If True, add a global LinearTransform at the start of the flow.

      trainable_base: If True and base_dist is None, use a DiagNormal base with
                      trainable loc and log_scale. Ignored if base_dist is given.

      base_dist: Optional explicit base distribution object, e.g. DiagNormal(dim).
                 If provided, this takes precedence over trainable_base.

      base_params: Optional initial parameters for base_dist. If None and
                   base_dist is provided, defaults to {}. For DiagNormal, you
                   typically want:
                       {"loc": zeros(dim), "log_scale": zeros(dim)}.
      activation: Activation function for conditioner MLPs.
      loft_tau: Threshold parameter for LOFT (stabilizing transform) in the affine coupling.

    Returns:
      flow: Flow object (definition only, no parameters inside).
      params: PyTree of parameters for the flow (including base and transform).
    """
    if dim <= 0:
        raise ValueError(f"build_realnvp: dim must be positive, got {dim}.")
    if num_layers <= 0:
        raise ValueError(
            f"build_realnvp: num_layers must be positive, got {num_layers}."
        )
    if context_dim < 0:
        raise ValueError(
            f"build_realnvp: context_dim must be non-negative, got {context_dim}."
        )

    # --------------------------------------------------------------
    # Base distribution and its parameters
    # --------------------------------------------------------------
    if base_dist is not None:
        # User provided an explicit base distribution.
        # If no params are given, default to an empty dict.
        base_params_resolved = {} if base_params is None else base_params
        base = base_dist
    elif trainable_base:
        # Convenience: trainable diagonal Gaussian base.
        base = DiagNormal(dim=dim)
        base_params_resolved = {
            "loc": jnp.zeros((dim,), dtype=jnp.float32),
            "log_scale": jnp.zeros((dim,), dtype=jnp.float32),
        }
    else:
        # Default: standard normal with no trainable parameters.
        base = StandardNormal(dim=dim)
        base_params_resolved = {}

    # --------------------------------------------------------------
    # Transform: stack of affine couplings (and optional permutations)
    # --------------------------------------------------------------
    keys = jax.random.split(key, num_layers)

    blocks = []
    block_params = []

    # Optional global linear layer at the beginning of the transform.
    if use_linear:
        lin_block = LinearTransform(dim=dim)
        # Identity initialization: L = I, T = I => W = I.
        lower_raw = jnp.zeros((dim, dim), dtype=jnp.float32)
        upper_raw = jnp.zeros((dim, dim), dtype=jnp.float32)
        log_diag = jnp.zeros((dim,), dtype=jnp.float32)

        blocks.append(lin_block)
        block_params.append(
            {"lower": lower_raw, "upper": upper_raw, "log_diag": log_diag}
        )

    parity = 0  # start with [1, 0, 1, 0, ...]
    for layer_idx in range(num_layers):
        mask = _make_alternating_mask(dim, parity)
        parity = 1 - parity  # flip parity for next layer

        # Initialize conditioner MLP.
        mlp, mlp_params = init_mlp(
            keys[layer_idx],
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=2 * dim,
            activation=activation,
            res_scale=res_scale,
        )

        coupling = AffineCoupling(
            mask=mask,
            conditioner=mlp,
            max_log_scale=max_log_scale,
        )

        blocks.append(coupling)
        block_params.append({"mlp": mlp_params})

        # Optionally insert a permutation between coupling layers,
        # but not after the last one.
        if use_permutation and layer_idx != num_layers - 1:
            perm = jnp.arange(dim - 1, -1, -1)  # simple fixed reverse permutation
            perm_block = Permutation(perm=perm)
            blocks.append(perm_block)
            block_params.append({})  # no parameters for permutation
    
    # add the final LOFT transform
    loft_block = LoftTransform(
        dim=dim,
        tau=loft_tau,
    )
    blocks.append(loft_block)
    block_params.append({})  # no parameters for LOFT

    transform = CompositeTransform(blocks=blocks)
    
    # Analyze mask coverage to catch potential issues.
    analyze_mask_coverage(blocks, dim)

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }

    flow = Flow(base_dist=base, transform=transform)
    return flow, params


# ====================================================================
# Spline RealNVP builder
# ====================================================================
# def _inv_softplus(y: Array) -> Array:
#     # Numerically stable inverse of softplus for y > 0.
#     # softplus(x) = log(1 + exp(x))  =>  x = log(exp(y) - 1)
#     y = jnp.maximum(y, 1e-6)
#     return jnp.log(jnp.expm1(y))


# def _patch_spline_conditioner_dense_out(
#     mlp_params: Any,
#     *,
#     dim: int,
#     num_bins: int,
#     min_derivative: float,
# ) -> Any:
#     """
#     Make the spline conditioner start near identity by patching the final Dense layer:
#       - kernel -> 0
#       - bias widths/heights logits -> 0  (uniform bins after softmax)
#       - bias derivatives -> inv_softplus(1 - min_derivative) so that
#             min_derivative + softplus(bias) ≈ 1  => initial slopes ≈ 1
#     Assumes the final layer is named 'dense_out' (as in nflows/nets.py).
    
#     This is often important for stabilizing training of spline flows and makes
#     them behave more like standard affine coupling layers at the start of training.
#     """
#     if "dense_out" not in mlp_params:
#         raise KeyError(
#             "_patch_spline_conditioner_dense_out: expected mlp_params to contain 'dense_out'. "
#             "Make sure your MLP final layer is named 'dense_out'."
#         )

#     dense_out = mlp_params["dense_out"]
#     if "kernel" not in dense_out or "bias" not in dense_out:
#         raise KeyError(
#             "_patch_spline_conditioner_dense_out: expected dense_out to contain 'kernel' and 'bias'."
#         )

#     K = num_bins
#     params_per_dim = 3 * K - 1

#     bias = dense_out["bias"]
#     if bias.shape != (dim * params_per_dim,):
#         raise ValueError(
#             "_patch_spline_conditioner_dense_out: unexpected dense_out bias shape. "
#             f"Expected {(dim * params_per_dim,)}, got {bias.shape}."
#         )

#     # New kernel: all zeros => NN outputs controlled purely by bias at init.
#     new_kernel = jnp.zeros_like(dense_out["kernel"])

#     # New bias: widths/heights logits = 0; derivatives chosen to yield slopes ~ 1.
#     new_bias = jnp.zeros_like(bias).reshape((dim, params_per_dim))

#     # Derivatives live in the last (K-1) entries of each per-dim block.
#     # We want: min_derivative + softplus(raw_d) ≈ 1.
#     target = 1.0 - float(min_derivative)
#     raw_d0 = _inv_softplus(jnp.asarray(target, dtype=new_bias.dtype))

#     new_bias = new_bias.at[:, 2 * K :].set(raw_d0)  # set all derivative logits

#     new_bias = new_bias.reshape((dim * params_per_dim,))

#     # Rebuild mlp_params with patched dense_out.
#     new_dense_out = dict(dense_out)
#     new_dense_out["kernel"] = new_kernel
#     new_dense_out["bias"] = new_bias

#     new_mlp_params = dict(mlp_params)
#     new_mlp_params["dense_out"] = new_dense_out
#     return new_mlp_params

def _logit(p: Array) -> Array:
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _patch_spline_conditioner_dense_out(
    mlp_params: Any,
    *,
    dim: int,
    num_bins: int,
    min_derivative: float,
    max_derivative: float,
) -> Any:
    """
    Make the spline conditioner start near identity by patching the final Dense layer:
      - kernel -> 0
      - bias widths/heights logits -> 0  (uniform bins after softmax)
      - bias derivatives -> chosen so initial internal derivatives are ~1.

    This assumes:
      - spline parameter layout per dimension is [widths(K), heights(K), derivatives(K-1)]
      - final MLP layer is named 'dense_out'
      - derivatives are parameterized either as:
          (A) min + softplus(u)                    if max_derivative is None
          (B) min + (max-min)*sigmoid(u)           if max_derivative is not None

    If you change the derivative parameterization in splines.py, update this function.
    """
    if "dense_out" not in mlp_params:
        raise KeyError(
            "_patch_spline_conditioner_dense_out: expected mlp_params to contain 'dense_out'. "
            "Make sure your MLP final layer is named 'dense_out'."
        )

    dense_out = mlp_params["dense_out"]
    if "kernel" not in dense_out or "bias" not in dense_out:
        raise KeyError(
            "_patch_spline_conditioner_dense_out: expected dense_out to contain 'kernel' and 'bias'."
        )

    K = num_bins
    params_per_dim = 3 * K - 1

    bias = dense_out["bias"]
    if bias.shape != (dim * params_per_dim,):
        raise ValueError(
            "_patch_spline_conditioner_dense_out: unexpected dense_out bias shape. "
            f"Expected {(dim * params_per_dim,)}, got {bias.shape}."
        )

    new_kernel = jnp.zeros_like(dense_out["kernel"])
    new_bias = jnp.zeros_like(bias).reshape((dim, params_per_dim))

    # We want: min_derivative + (max-min)*sigmoid(u0) ≈ 1
    lo = float(min_derivative)
    hi = float(max_derivative)
    if not (lo < 1.0 < hi):
        raise ValueError(
            "_patch_spline_conditioner_dense_out: to initialize derivatives ~1 with "
            "sigmoid-bounded parameterization, require min_derivative < 1 < max_derivative. "
            f"Got min_derivative={lo}, max_derivative={hi}."
        )
    alpha = (1.0 - lo) / (hi - lo)  # desired sigmoid output
    u0 = _logit(jnp.asarray(alpha, dtype=new_bias.dtype))

    new_bias = new_bias.at[:, 2 * K :].set(u0)
    new_bias = new_bias.reshape((dim * params_per_dim,))

    new_dense_out = dict(dense_out)
    new_dense_out["kernel"] = new_kernel
    new_dense_out["bias"] = new_bias

    new_mlp_params = dict(mlp_params)
    new_mlp_params["dense_out"] = new_dense_out
    return new_mlp_params


def build_spline_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    *,
    context_dim: int = 0,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-2,
    min_bin_height: float = 1e-2,
    min_derivative: float = 1e-2,
    max_derivative: float = 10.0,
    res_scale: float = 0.1,
    use_permutation: bool = False,
    use_linear: bool = False,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
) -> Tuple[Flow, Any]:
    """
    Construct a RealNVP-style flow with monotonic rational-quadratic spline coupling layers
    (Durkan et al., 2019).

    Architecture:
      - Base distribution p_base(z):
          * by default: StandardNormal(dim) with no trainable parameters.
          * if trainable_base=True and base_dist is None:
                DiagNormal(dim) with trainable loc and log_scale.
          * if base_dist is provided explicitly:
                use that object and base_params as given.
      - Transform: CompositeTransform of
          (optional) LinearTransform,
          [SplineCoupling, (Permutation), SplineCoupling, (Permutation), ...],
          (final) LoftTransform (stabilizing tail / bounding layer).

    Parameters structure:
      params["base"]      -> parameters for the base distribution (PyTree)
      params["transform"] -> list of per-block params, same length as transform.blocks
        - for SplineCoupling blocks: {"mlp": mlp_params}
        - for Permutation blocks: {}
        - for LinearTransform block: {"lower": ..., "upper": ..., "log_diag": ...}
        - for LoftTransform block: {}

    Spline coupling parameterization:
      For K bins, each dimension uses:
        - widths:      K
        - heights:     K
        - derivatives: K-1 (internal knot derivatives; boundary derivatives fixed to 1)
      => params_per_dim = 3K - 1
      => conditioner output dim = dim * (3K - 1)

    Arguments:
      key: JAX PRNGKey used to initialize all MLPs.
      dim: Dimensionality of the data / latent space.
      num_layers: Number of spline coupling layers.
      hidden_dim: Width of hidden layers in conditioner MLPs.
      n_hidden_layers: Number of residual blocks in conditioner MLPs.
      context_dim: Dimensionality of the conditioning variable. If 0 (default),
                   the flow is unconditional. Context is concatenated to the
                   conditioner input.

      num_bins: Number of spline bins (K).
      tail_bound: Spline acts on [-B, B]; outside, the transform is linear with slope 1.
      min_bin_width/min_bin_height/min_derivative: Stability floors used by the spline.
      res_scale: Scale factor for residual connections in conditioner MLPs.

      use_permutation: If True, insert a fixed reverse permutation between successive couplings.
      use_linear: If True, add a global LinearTransform at the start of the flow.

      trainable_base: If True and base_dist is None, use a DiagNormal base with trainable params.
      base_dist: Optional explicit base distribution object; takes precedence over trainable_base.
      base_params: Optional initial parameters for base_dist. If None and base_dist is provided,
                   defaults to {}. For DiagNormal, typical init is:
                      {"loc": zeros(dim), "log_scale": zeros(dim)}.

      activation: Activation function for conditioner MLPs.
      loft_tau: Threshold parameter for LOFT (stabilizing transform) appended at the end.

    Returns:
      flow: Flow object (definition only, no parameters inside).
      params: PyTree of parameters for the flow (including base and transform).
    """
    if dim <= 0:
        raise ValueError(f"build_spline_realnvp: dim must be positive, got {dim}.")
    if num_layers <= 0:
        raise ValueError(
            f"build_spline_realnvp: num_layers must be positive, got {num_layers}."
        )
    if context_dim < 0:
        raise ValueError(
            f"build_spline_realnvp: context_dim must be non-negative, got {context_dim}."
        )
    if num_bins <= 0:
        raise ValueError(
            f"build_spline_realnvp: num_bins must be positive, got {num_bins}."
        )
    if min_bin_width * num_bins >= 1.0:
        raise ValueError(
            "build_spline_realnvp: min_bin_width * num_bins must be < 1. "
            f"Got {min_bin_width} * {num_bins} = {min_bin_width * num_bins}."
        )
    if min_bin_height * num_bins >= 1.0:
        raise ValueError(
            "build_spline_realnvp: min_bin_height * num_bins must be < 1. "
            f"Got {min_bin_height} * {num_bins} = {min_bin_height * num_bins}."
        )

    # --------------------------------------------------------------
    # Base distribution and its parameters (same logic as build_realnvp)
    # --------------------------------------------------------------
    if base_dist is not None:
        base_params_resolved = {} if base_params is None else base_params
        base = base_dist
    elif trainable_base:
        base = DiagNormal(dim=dim)
        base_params_resolved = {
            "loc": jnp.zeros((dim,), dtype=jnp.float32),
            "log_scale": jnp.zeros((dim,), dtype=jnp.float32),
        }
    else:
        base = StandardNormal(dim=dim)
        base_params_resolved = {}

    # --------------------------------------------------------------
    # Transform: stack of spline couplings (and optional permutations)
    # --------------------------------------------------------------
    keys = jax.random.split(key, num_layers)

    blocks = []
    block_params = []

    # Optional global linear layer at the beginning of the transform.
    if use_linear:
        lin_block = LinearTransform(dim=dim)
        lower_raw = jnp.zeros((dim, dim), dtype=jnp.float32)
        upper_raw = jnp.zeros((dim, dim), dtype=jnp.float32)
        log_diag = jnp.zeros((dim,), dtype=jnp.float32)

        blocks.append(lin_block)
        block_params.append(
            {"lower": lower_raw, "upper": upper_raw, "log_diag": log_diag}
        )

    params_per_dim = 3 * num_bins - 1
    out_dim = dim * params_per_dim

    parity = 0  # start with [1, 0, 1, 0, ...]
    for layer_idx in range(num_layers):
        mask = _make_alternating_mask(dim, parity)
        parity = 1 - parity

        # Initialize conditioner MLP.
        mlp, mlp_params = init_mlp(
            keys[layer_idx],
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Patch final layer init so each spline coupling starts near identity.
        # Important for stabilizing training at the start of optimization.
        mlp_params = _patch_spline_conditioner_dense_out(
            mlp_params,
            dim=dim,
            num_bins=num_bins,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
        )

        coupling = SplineCoupling(
            mask=mask,
            conditioner=mlp,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
        )

        blocks.append(coupling)
        block_params.append({"mlp": mlp_params})

        if use_permutation and layer_idx != num_layers - 1:
            perm = jnp.arange(dim - 1, -1, -1)
            perm_block = Permutation(perm=perm)
            blocks.append(perm_block)
            block_params.append({})

    # Final LOFT transform (same as build_realnvp)
    # It is used for stabilizing training in high-dimensional settings.
    loft_block = LoftTransform(dim=dim, tau=loft_tau)
    blocks.append(loft_block)
    block_params.append({})

    transform = CompositeTransform(blocks=blocks)

    # Diagnostics: analyze mask coverage to make sure all dims are transformed.
    # Issues can arise due to permutations canceling out masks.
    analyze_mask_coverage(blocks, dim)

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }

    flow = Flow(base_dist=base, transform=transform)
    return flow, params