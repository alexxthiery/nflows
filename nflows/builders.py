# nflows/builders.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp

from .nets import init_resnet, Array, PRNGKey
from .distributions import StandardNormal, DiagNormal
from .transforms import AffineCoupling, Permutation, CompositeTransform, LinearTransform, LoftTransform, SplineCoupling
from .flows import Bijection, Flow


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
def make_alternating_mask(dim: int, parity: int) -> Array:
    """
    Build a binary mask of length dim with alternating 1/0 pattern.

    In coupling layers, mask=1 means the dimension is frozen (passed through),
    and mask=0 means the dimension is transformed.

    Args:
        dim: Length of the mask (must be positive).
        parity: Which pattern to use (must be 0 or 1).
            parity=0: mask = [1, 0, 1, 0, ...] (even indices frozen)
            parity=1: mask = [0, 1, 0, 1, ...] (odd indices frozen)

    Returns:
        Binary mask array of shape (dim,) with dtype float32.

    Raises:
        ValueError: If dim <= 0 or parity not in {0, 1}.

    Example:
        >>> make_alternating_mask(4, parity=0)
        Array([1., 0., 1., 0.], dtype=float32)
        >>> make_alternating_mask(4, parity=1)
        Array([0., 1., 0., 1.], dtype=float32)
    """
    if dim <= 0:
        raise ValueError(f"make_alternating_mask: dim must be positive, got {dim}.")
    if parity not in (0, 1):
        raise ValueError(f"make_alternating_mask: parity must be 0 or 1, got {parity}.")

    idx = jnp.arange(dim)
    mask = ((idx + parity) % 2 == 0).astype(jnp.float32)
    return mask


# Keep private alias for backward compatibility within module
_make_alternating_mask = make_alternating_mask


# ====================================================================
# Feature extractor creation
# ====================================================================
def create_feature_extractor(
    key: PRNGKey,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int = 2,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    res_scale: float = 0.1,
) -> Tuple[Any, Any]:
    """
    Create a context feature extractor network.

    The feature extractor is a ResNet that transforms raw context into learned
    features before they are used by coupling layers. This is useful when raw
    context features are high-dimensional or not directly informative.

    Args:
        key: JAX PRNGKey for initialization.
        in_dim: Input dimension (raw context dimension).
        hidden_dim: Width of hidden layers in the ResNet.
        out_dim: Output dimension (effective context dimension for couplings).
        n_layers: Number of residual blocks (default: 2).
        activation: Activation function (default: tanh).
        res_scale: Scale factor for residual connections (default: 0.1).

    Returns:
        feature_extractor: A Flax ResNet module.
        params: Parameters for the feature extractor.

    Raises:
        ValueError: If any dimension is not positive or n_layers < 1.

    Example:
        >>> fe, fe_params = create_feature_extractor(key, in_dim=8, hidden_dim=32, out_dim=16)
        >>> # Then use out_dim as context_dim for couplings:
        >>> coupling, c_params = AffineCoupling.create(..., context_dim=16)
    """
    if in_dim <= 0:
        raise ValueError(f"create_feature_extractor: in_dim must be positive, got {in_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"create_feature_extractor: hidden_dim must be positive, got {hidden_dim}.")
    if out_dim <= 0:
        raise ValueError(f"create_feature_extractor: out_dim must be positive, got {out_dim}.")
    if n_layers < 1:
        raise ValueError(f"create_feature_extractor: n_layers must be >= 1, got {n_layers}.")

    feature_extractor, params = init_resnet(
        key,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_layers,
        activation=activation,
        res_scale=res_scale,
        zero_init_output=False,
    )
    return feature_extractor, params


# ====================================================================
# Shared helpers
# ====================================================================
def _resolve_base_distribution(dim: int, trainable_base: bool, base_dist, base_params):
    """
    Resolve the base distribution and its parameters.

    Returns:
        (base, base_params_resolved): The distribution object and its parameters.
    """
    if base_dist is not None:
        return base_dist, (base_dist.init_params() if base_params is None else base_params)
    elif trainable_base:
        base = DiagNormal(dim=dim)
        return base, base.init_params()
    else:
        base = StandardNormal(dim=dim)
        return base, base.init_params()


def _init_context_extractor(
    key: PRNGKey,
    context_dim: int,
    extractor_hidden_dim: int,
    extractor_n_layers: int,
    feature_dim: int | None,
    activation: Callable[[Array], Array],
    res_scale: float,
):
    """
    Initialize context feature extractor if extractor_hidden_dim > 0.

    Returns:
        (feature_extractor, fe_params, effective_context_dim):
            - feature_extractor: ResNet callable or None
            - fe_params: Parameters for the extractor or None
            - effective_context_dim: Dimension of context fed to coupling layers
    """
    if extractor_hidden_dim <= 0:
        return None, None, context_dim

    effective_dim = feature_dim if feature_dim is not None else context_dim
    feature_extractor, fe_params = init_resnet(
        key,
        in_dim=context_dim,
        hidden_dim=extractor_hidden_dim,
        out_dim=effective_dim,
        n_hidden_layers=extractor_n_layers,
        activation=activation,
        res_scale=res_scale,
    )
    return feature_extractor, fe_params, effective_dim


# ====================================================================
# Flow/Bijection assembly from blocks
# ====================================================================
def _validate_blocks_and_params(
    blocks_and_params: List[Tuple[Any, Any]],
    function_name: str,
) -> Tuple[List[Any], List[Any]]:
    """
    Validate and unzip a list of (block, params) tuples.

    Args:
        blocks_and_params: List of (transform, params) tuples.
        function_name: Name of calling function for error messages.

    Returns:
        (blocks, block_params): Unzipped lists.

    Raises:
        ValueError: If validation fails.
    """
    if not blocks_and_params:
        raise ValueError(
            f"{function_name}: blocks_and_params cannot be empty. "
            "Provide at least one (transform, params) tuple."
        )

    blocks = []
    block_params = []

    for i, item in enumerate(blocks_and_params):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"{function_name}: Element {i} must be a (transform, params) tuple, "
                f"got {type(item).__name__} with {len(item) if hasattr(item, '__len__') else 'N/A'} elements."
            )
        block, params = item
        blocks.append(block)
        block_params.append(params)

    return blocks, block_params


def _infer_dim_from_blocks(blocks: List[Any]) -> int | None:
    """
    Infer dimension from coupling-like blocks by checking mask shape.

    Returns:
        Inferred dimension, or None if no coupling blocks found.
    """
    for block in blocks:
        mask = getattr(block, "mask", None)
        if mask is not None:
            return int(jnp.asarray(mask).shape[0])
    return None


def _check_dimension_consistency(blocks: List[Any], function_name: str) -> int | None:
    """
    Check that all coupling-like blocks have consistent dimensions.

    Returns:
        The consistent dimension, or None if no coupling blocks.

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    dims = []
    for i, block in enumerate(blocks):
        mask = getattr(block, "mask", None)
        if mask is not None:
            dim = int(jnp.asarray(mask).shape[0])
            dims.append((i, type(block).__name__, dim))

    if not dims:
        return None

    first_dim = dims[0][2]
    inconsistent = [(i, name, d) for i, name, d in dims if d != first_dim]

    if inconsistent:
        raise ValueError(
            f"{function_name}: Inconsistent dimensions across coupling blocks. "
            f"First coupling (index {dims[0][0]}, {dims[0][1]}) has dim={first_dim}, "
            f"but found: {[(f'index {i}, {name}, dim={d}') for i, name, d in inconsistent]}."
        )

    return first_dim


def assemble_bijection(
    blocks_and_params: List[Tuple[Any, Any]],
    feature_extractor: Any = None,
    feature_extractor_params: Any = None,
    validate: bool = True,
) -> Tuple[Bijection, Dict]:
    """
    Assemble a Bijection from a list of (transform, params) tuples.

    This function provides a mid-level API for building custom flow architectures
    by manually composing transform blocks. It handles the bookkeeping of aligning
    blocks with their parameters and constructing the correct params dict structure.

    Args:
        blocks_and_params: List of (transform, params) tuples, as returned by
            Transform.create() factory methods. Example:
                [
                    AffineCoupling.create(key1, dim=4, mask=mask0, ...),
                    SplineCoupling.create(key2, dim=4, mask=mask1, ...),
                    LinearTransform.create(key3, dim=4),
                ]
        feature_extractor: Optional context feature extractor (Flax module).
            If provided, raw context will be transformed before being passed
            to coupling layers.
        feature_extractor_params: Parameters for the feature extractor.
            Required if feature_extractor is provided.
        validate: If True (default), perform validation:
            - Check blocks_and_params is non-empty
            - Check each element is a 2-tuple
            - Check dimension consistency across coupling blocks
            - Run mask coverage analysis

    Returns:
        bijection: Bijection object with the composed transform.
        params: Dict with structure:
            {"transform": [block_params...]}
            or {"transform": [...], "feature_extractor": fe_params}

    Raises:
        ValueError: If validation fails.

    Example:
        >>> from nflows.builders import make_alternating_mask, assemble_bijection
        >>> from nflows.transforms import AffineCoupling, SplineCoupling, LoftTransform
        >>>
        >>> keys = jax.random.split(key, 4)
        >>> mask0 = make_alternating_mask(4, parity=0)
        >>> mask1 = make_alternating_mask(4, parity=1)
        >>>
        >>> blocks_and_params = [
        ...     AffineCoupling.create(keys[0], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2),
        ...     AffineCoupling.create(keys[1], dim=4, mask=mask1, hidden_dim=64, n_hidden_layers=2),
        ...     SplineCoupling.create(keys[2], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2, num_bins=8),
        ...     LoftTransform.create(keys[3], dim=4),
        ... ]
        >>>
        >>> bijection, params = assemble_bijection(blocks_and_params)
        >>> y, log_det = bijection.forward(params, x)
    """
    # Validate feature extractor arguments
    if feature_extractor is not None and feature_extractor_params is None:
        raise ValueError(
            "assemble_bijection: feature_extractor_params is required when "
            "feature_extractor is provided."
        )
    if feature_extractor is None and feature_extractor_params is not None:
        raise ValueError(
            "assemble_bijection: feature_extractor_params provided but "
            "feature_extractor is None."
        )

    # Validate and unzip blocks
    blocks, block_params = _validate_blocks_and_params(
        blocks_and_params, "assemble_bijection"
    )

    if validate:
        # Check dimension consistency
        dim = _check_dimension_consistency(blocks, "assemble_bijection")

        # Run mask coverage analysis if we have coupling blocks
        if dim is not None:
            analyze_mask_coverage(blocks, dim)

    # Assemble
    transform = CompositeTransform(blocks=blocks)

    params: Dict[str, Any] = {"transform": block_params}
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor_params

    bijection = Bijection(transform=transform, feature_extractor=feature_extractor)
    return bijection, params


def assemble_flow(
    blocks_and_params: List[Tuple[Any, Any]],
    base: Any,
    base_params: Any = None,
    feature_extractor: Any = None,
    feature_extractor_params: Any = None,
    validate: bool = True,
) -> Tuple[Flow, Dict]:
    """
    Assemble a Flow from blocks, base distribution, and optional feature extractor.

    This function provides a mid-level API for building custom flow architectures.
    It combines manually composed transform blocks with a base distribution to
    create a full normalizing flow.

    Args:
        blocks_and_params: List of (transform, params) tuples, as returned by
            Transform.create() factory methods.
        base: Base distribution object (e.g., StandardNormal(dim), DiagNormal(dim)).
            Must have log_prob(), sample(), and init_params() methods.
        base_params: Parameters for the base distribution. If None, calls
            base.init_params() to get default parameters.
        feature_extractor: Optional context feature extractor (Flax module).
        feature_extractor_params: Parameters for the feature extractor.
            Required if feature_extractor is provided.
        validate: If True (default), perform validation checks.

    Returns:
        flow: Flow object with base distribution and composed transform.
        params: Dict with structure:
            {"base": base_params, "transform": [block_params...]}
            or {"base": ..., "transform": [...], "feature_extractor": fe_params}

    Raises:
        ValueError: If validation fails or base is None.

    Example:
        >>> from nflows.builders import make_alternating_mask, assemble_flow
        >>> from nflows.transforms import AffineCoupling, LoftTransform
        >>> from nflows.distributions import StandardNormal
        >>>
        >>> keys = jax.random.split(key, 3)
        >>> mask0 = make_alternating_mask(4, parity=0)
        >>> mask1 = make_alternating_mask(4, parity=1)
        >>>
        >>> blocks_and_params = [
        ...     AffineCoupling.create(keys[0], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2),
        ...     AffineCoupling.create(keys[1], dim=4, mask=mask1, hidden_dim=64, n_hidden_layers=2),
        ...     LoftTransform.create(keys[2], dim=4),
        ... ]
        >>>
        >>> flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=4))
        >>> log_prob = flow.log_prob(params, x)
        >>> samples = flow.sample(params, key, shape=(100,))
    """
    if base is None:
        raise ValueError(
            "assemble_flow: base distribution is required. "
            "Use assemble_bijection() if you don't need a base distribution."
        )

    # Validate feature extractor arguments
    if feature_extractor is not None and feature_extractor_params is None:
        raise ValueError(
            "assemble_flow: feature_extractor_params is required when "
            "feature_extractor is provided."
        )
    if feature_extractor is None and feature_extractor_params is not None:
        raise ValueError(
            "assemble_flow: feature_extractor_params provided but "
            "feature_extractor is None."
        )

    # Validate and unzip blocks
    blocks, block_params = _validate_blocks_and_params(
        blocks_and_params, "assemble_flow"
    )

    if validate:
        # Check dimension consistency
        dim = _check_dimension_consistency(blocks, "assemble_flow")

        # Run mask coverage analysis if we have coupling blocks
        if dim is not None:
            analyze_mask_coverage(blocks, dim)

    # Resolve base params
    if base_params is None:
        base_params = base.init_params()

    # Assemble
    transform = CompositeTransform(blocks=blocks)

    params: Dict[str, Any] = {
        "base": base_params,
        "transform": block_params,
    }
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor_params

    flow = Flow(base_dist=base, transform=transform, feature_extractor=feature_extractor)
    return flow, params


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
    use_loft: bool = True,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
    return_transform_only: bool = False,
) -> Tuple[Flow | Bijection, Any]:
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
      use_loft: If True (default), append a LoftTransform at the end for tail stabilization.

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
      return_transform_only: If True, return a Bijection (transform + optional feature
                   extractor) instead of a full Flow. Useful when you only need the
                   invertible map with tractable Jacobian, without a base distribution.

    Returns:
      If return_transform_only=False (default):
        flow: Flow object (definition only, no parameters inside).
        params: PyTree with keys "base", "transform", and optionally "feature_extractor".
      If return_transform_only=True:
        bijection: Bijection object (transform + optional feature extractor).
        params: PyTree with keys "transform" and optionally "feature_extractor".
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

    # Context feature extractor (optional)
    if context_extractor_hidden_dim > 0 and context_dim == 0:
        raise ValueError(
            "build_realnvp: context_extractor_hidden_dim > 0 requires context_dim > 0."
        )
    key, fe_key = jax.random.split(key)
    feature_extractor, fe_params, effective_context_dim = _init_context_extractor(
        fe_key, context_dim, context_extractor_hidden_dim, context_extractor_n_layers,
        context_feature_dim, activation, res_scale,
    )

    # Base distribution (skip if only returning transform)
    if not return_transform_only:
        base, base_params_resolved = _resolve_base_distribution(
            dim, trainable_base, base_dist, base_params
        )

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

    # Optional final LOFT transform for tail stabilization
    if use_loft:
        key, loft_key = jax.random.split(key)
        loft_block, loft_params = LoftTransform.create(loft_key, dim=dim, tau=loft_tau)
        blocks.append(loft_block)
        block_params.append(loft_params)

    transform = CompositeTransform(blocks=blocks)

    # Analyze mask coverage to catch potential issues.
    analyze_mask_coverage(blocks, dim)

    # Build params and return object based on return_transform_only
    if return_transform_only:
        params = {"transform": block_params}
        if feature_extractor is not None:
            params["feature_extractor"] = fe_params
        bijection = Bijection(transform=transform, feature_extractor=feature_extractor)
        return bijection, params

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }
    if feature_extractor is not None:
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
    use_loft: bool = True,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
    return_transform_only: bool = False,
) -> Tuple[Flow | Bijection, Any]:
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
      use_loft: If True (default), append a LoftTransform at the end for tail stabilization.

      trainable_base: If True and base_dist is None, use a DiagNormal base with trainable params.
      base_dist: Optional explicit base distribution object; takes precedence over trainable_base.
      base_params: Optional initial parameters for base_dist. If None and base_dist is provided,
                   defaults to {}. For DiagNormal, typical init is:
                      {"loc": zeros(dim), "log_scale": zeros(dim)}.

      activation: Activation function for conditioner MLPs.
      loft_tau: Threshold parameter for LOFT (stabilizing transform) appended at the end.
      return_transform_only: If True, return a Bijection (transform + optional feature
                   extractor) instead of a full Flow. Useful when you only need the
                   invertible map with tractable Jacobian, without a base distribution.

    Returns:
      If return_transform_only=False (default):
        flow: Flow object (definition only, no parameters inside).
        params: PyTree with keys "base", "transform", and optionally "feature_extractor".
      If return_transform_only=True:
        bijection: Bijection object (transform + optional feature extractor).
        params: PyTree with keys "transform" and optionally "feature_extractor".
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

    # Context feature extractor (optional)
    if context_extractor_hidden_dim > 0 and context_dim == 0:
        raise ValueError(
            "build_spline_realnvp: context_extractor_hidden_dim > 0 requires context_dim > 0."
        )
    key, fe_key = jax.random.split(key)
    feature_extractor, fe_params, effective_context_dim = _init_context_extractor(
        fe_key, context_dim, context_extractor_hidden_dim, context_extractor_n_layers,
        context_feature_dim, activation, res_scale,
    )

    # Base distribution (skip if only returning transform)
    if not return_transform_only:
        base, base_params_resolved = _resolve_base_distribution(
            dim, trainable_base, base_dist, base_params
        )

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

    # Optional final LOFT transform for tail stabilization
    if use_loft:
        key, loft_key = jax.random.split(key)
        loft_block, loft_params = LoftTransform.create(loft_key, dim=dim, tau=loft_tau)
        blocks.append(loft_block)
        block_params.append(loft_params)

    transform = CompositeTransform(blocks=blocks)

    # Diagnostics: analyze mask coverage to make sure all dims are transformed.
    # Issues can arise due to permutations canceling out masks.
    analyze_mask_coverage(blocks, dim)

    # Build params and return object based on return_transform_only
    if return_transform_only:
        params = {"transform": block_params}
        if feature_extractor is not None:
            params["feature_extractor"] = fe_params
        bijection = Bijection(transform=transform, feature_extractor=feature_extractor)
        return bijection, params

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }
    if feature_extractor is not None:
        params["feature_extractor"] = fe_params

    flow = Flow(base_dist=base, transform=transform, feature_extractor=feature_extractor)
    return flow, params