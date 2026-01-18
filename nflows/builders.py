# nflows/builders.py
from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from .nets import init_resnet, Array, PRNGKey
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
    context_extractor_hidden_dim: int = 0,
    context_extractor_n_layers: int = 2,
    context_feature_dim: int | None = None,
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
      context_extractor_hidden_dim: Hidden dimension for context feature extractor.
                   If > 0, a ResNet is used to extract features from context before
                   concatenation. If 0 (default), raw context is used directly.
      context_extractor_n_layers: Number of residual blocks in context extractor (default: 2).
      context_feature_dim: Output dimension of context extractor. If None (default),
                   uses context_dim (i.e., same dimension as input context).
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
    # Context feature extractor (optional)
    # --------------------------------------------------------------
    use_context_extractor = context_extractor_hidden_dim > 0

    if use_context_extractor:
        if context_dim == 0:
            raise ValueError(
                "build_realnvp: context_extractor_hidden_dim > 0 requires context_dim > 0."
            )
        # Determine effective context dimension for coupling MLPs.
        effective_context_dim = context_feature_dim if context_feature_dim is not None else context_dim

        # Split off a key for the feature extractor.
        key, fe_key = jax.random.split(key)
        feature_extractor, fe_params = init_resnet(
            fe_key,
            in_dim=context_dim,
            hidden_dim=context_extractor_hidden_dim,
            out_dim=effective_context_dim,
            n_hidden_layers=context_extractor_n_layers,
            activation=activation,
            res_scale=res_scale,
        )
    else:
        effective_context_dim = context_dim
        feature_extractor = None
        fe_params = None

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
        key, lin_key = jax.random.split(key)
        lin_block, lin_params = LinearTransform.create(lin_key, dim=dim)
        blocks.append(lin_block)
        block_params.append(lin_params)

    parity = 0  # start with [1, 0, 1, 0, ...]
    for layer_idx in range(num_layers):
        mask = _make_alternating_mask(dim, parity)
        parity = 1 - parity  # flip parity for next layer

        coupling, coupling_params = AffineCoupling.create(
            keys[layer_idx],
            dim=dim,
            mask=mask,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            context_dim=effective_context_dim,
            activation=activation,
            res_scale=res_scale,
            max_log_scale=max_log_scale,
        )
        blocks.append(coupling)
        block_params.append(coupling_params)

        # Optionally insert a permutation between coupling layers,
        # but not after the last one.
        if use_permutation and layer_idx != num_layers - 1:
            perm = jnp.arange(dim - 1, -1, -1)  # simple fixed reverse permutation
            perm_block, perm_params = Permutation.create(keys[layer_idx], perm=perm)
            blocks.append(perm_block)
            block_params.append(perm_params)

    # add the final LOFT transform
    key, loft_key = jax.random.split(key)
    loft_block, loft_params = LoftTransform.create(loft_key, dim=dim, tau=loft_tau)
    blocks.append(loft_block)
    block_params.append(loft_params)

    transform = CompositeTransform(blocks=blocks)
    
    # Analyze mask coverage to catch potential issues.
    analyze_mask_coverage(blocks, dim)

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }

    # Add feature extractor params if used.
    if use_context_extractor:
        params["feature_extractor"] = fe_params

    flow = Flow(base_dist=base, transform=transform, feature_extractor=feature_extractor)
    return flow, params

def build_spline_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    *,
    context_dim: int = 0,
    context_extractor_hidden_dim: int = 0,
    context_extractor_n_layers: int = 2,
    context_feature_dim: int | None = None,
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
      context_extractor_hidden_dim: Hidden dimension for context feature extractor.
                   If > 0, a ResNet is used to extract features from context before
                   concatenation. If 0 (default), raw context is used directly.
      context_extractor_n_layers: Number of residual blocks in context extractor (default: 2).
      context_feature_dim: Output dimension of context extractor. If None (default),
                   uses context_dim (i.e., same dimension as input context).

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
    # Context feature extractor (optional)
    # --------------------------------------------------------------
    use_context_extractor = context_extractor_hidden_dim > 0

    if use_context_extractor:
        if context_dim == 0:
            raise ValueError(
                "build_spline_realnvp: context_extractor_hidden_dim > 0 requires context_dim > 0."
            )
        # Determine effective context dimension for coupling MLPs.
        effective_context_dim = context_feature_dim if context_feature_dim is not None else context_dim

        # Split off a key for the feature extractor.
        key, fe_key = jax.random.split(key)
        feature_extractor, fe_params = init_resnet(
            fe_key,
            in_dim=context_dim,
            hidden_dim=context_extractor_hidden_dim,
            out_dim=effective_context_dim,
            n_hidden_layers=context_extractor_n_layers,
            activation=activation,
            res_scale=res_scale,
        )
    else:
        effective_context_dim = context_dim
        feature_extractor = None
        fe_params = None

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
        key, lin_key = jax.random.split(key)
        lin_block, lin_params = LinearTransform.create(lin_key, dim=dim)
        blocks.append(lin_block)
        block_params.append(lin_params)

    parity = 0  # start with [1, 0, 1, 0, ...]
    for layer_idx in range(num_layers):
        mask = _make_alternating_mask(dim, parity)
        parity = 1 - parity

        coupling, coupling_params = SplineCoupling.create(
            keys[layer_idx],
            dim=dim,
            mask=mask,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            context_dim=effective_context_dim,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            activation=activation,
            res_scale=res_scale,
        )
        blocks.append(coupling)
        block_params.append(coupling_params)

        if use_permutation and layer_idx != num_layers - 1:
            perm = jnp.arange(dim - 1, -1, -1)
            perm_block, perm_params = Permutation.create(keys[layer_idx], perm=perm)
            blocks.append(perm_block)
            block_params.append(perm_params)

    # Final LOFT transform (same as build_realnvp)
    # It is used for stabilizing training in high-dimensional settings.
    key, loft_key = jax.random.split(key)
    loft_block, loft_params = LoftTransform.create(loft_key, dim=dim, tau=loft_tau)
    blocks.append(loft_block)
    block_params.append(loft_params)

    transform = CompositeTransform(blocks=blocks)

    # Diagnostics: analyze mask coverage to make sure all dims are transformed.
    # Issues can arise due to permutations canceling out masks.
    analyze_mask_coverage(blocks, dim)

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }

    # Add feature extractor params if used.
    if use_context_extractor:
        params["feature_extractor"] = fe_params

    flow = Flow(base_dist=base, transform=transform, feature_extractor=feature_extractor)
    return flow, params