# nflows/builders.py
from __future__ import annotations

from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp

from .nets import MLP, init_mlp, Array, PRNGKey
from .distributions import StandardNormal, DiagNormal
from .transforms import AffineCoupling, Permutation, CompositeTransform, LinearTransform
from .flows import Flow


import jax.numpy as jnp
from .transforms import AffineCoupling, Permutation, CompositeTransform


def analyze_mask_coverage(blocks, dim: int) -> None:
    """
    Check how often each *original* dimension is transformed by AffineCoupling
    blocks, taking permutations into account.

    Raises a ValueError if any dimension is never transformed.
    
    Example:
        dim = 2
        use_permutation = True
    In this case, with alternating masks and permutations cancel out. This means 
    that one of the two dimensions is never transformed. And it is very hard to detect!
    """
    # current_pos[i] = original dim index currently sitting at position i
    current_pos = jnp.arange(dim)
    transformed_counts = jnp.zeros(dim, dtype=jnp.int32)

    for block in blocks:
        if isinstance(block, AffineCoupling):
            # mask = 1: frozen; mask = 0: transformed
            mask = block.mask
            if mask.shape[0] != dim:
                raise ValueError(
                    # multiline err
                    f"AffineCoupling mask shape {mask.shape} incompatible with dim={dim}."
                )

            transformed_positions = jnp.where(mask == 0)[0]
            # Which original dims are at those positions?
            transformed_original = current_pos[transformed_positions]
            transformed_counts = transformed_counts.at[transformed_original].add(1)

        elif isinstance(block, Permutation):
            # Update "current positions": y[..., i] = x[..., perm[i]]
            # So new_current_pos[i] = old_current_pos[perm[i]]
            current_pos = current_pos[block.perm]

        else:
            # ignore other block types for now, or add special handling later
            pass

    # Any dimension never transformed?
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
    hidden_sizes: Sequence[int],
    max_log_scale: float = 5.0,
    use_permutation: bool = False,
    use_linear: bool = False,
    *,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
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
      hidden_sizes: Hidden layer sizes for all conditioner MLPs.
      max_log_scale: Bound on |log_scale| via tanh to stabilize training.
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

        # Conditioner MLP: input dim = dim, output dim = 2 * dim.
        mlp, mlp_params = init_mlp(
            keys[layer_idx],
            in_dim=dim,
            hidden_sizes=hidden_sizes,
            out_dim=2 * dim,
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

    transform = CompositeTransform(blocks=blocks)
    
    # Analyze mask coverage to catch potential issues.
    analyze_mask_coverage(blocks, dim)

    params = {
        "base": base_params_resolved,
        "transform": block_params,
    }

    flow = Flow(base_dist=base, transform=transform)
    return flow, params