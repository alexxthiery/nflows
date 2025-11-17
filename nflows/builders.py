# nflows/builders.py
from __future__ import annotations

from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp

from .nets import MLP, init_mlp, Array, PRNGKey
from .distributions import StandardNormal
from .transforms import AffineCoupling, Permutation, CompositeTransform
from .flows import Flow


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


def build_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    max_log_scale: float = 5.0,
    use_permutation: bool = True,
) -> Tuple[Flow, Any]:
    """
    Construct a RealNVP-style flow with affine coupling layers.

    Architecture:
      - Base distribution: StandardNormal(dim)
      - Transform: CompositeTransform of
          [AffineCoupling, (Permutation), AffineCoupling, (Permutation), ...]

    Each coupling layer:
      - Uses a binary mask to split the features into "condition" and "transform" halves.
      - Uses an MLP conditioner with input dim = dim and output dim = 2 * dim.
      - Interprets the conditioner output as (shift, log_scale) for all dims,
        but only applies them on the unmasked dims.

    Parameters structure:
      params["base"]      -> parameters for base distribution (unused, set to {})
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

    Returns:
      flow: Flow object (definition only, no parameters inside).
      params: PyTree of parameters for the flow.
    """
    if dim <= 0:
        raise ValueError(f"build_realnvp: dim must be positive, got {dim}.")
    if num_layers <= 0:
        raise ValueError(
            f"build_realnvp: num_layers must be positive, got {num_layers}."
        )

    # Initialize base distribution (no trainable params for StandardNormal).
    base_dist = StandardNormal(dim=dim)
    base_params = {}

    # Split key for each coupling layer's MLP initialization.
    keys = jax.random.split(key, num_layers)

    blocks = []
    block_params = []

    # Alternate masks across layers to mix dimensions.
    parity = 0  # start with [1, 0, 1, 0, ...]
    for layer_idx in range(num_layers):
        mask = _make_alternating_mask(dim, parity)
        parity = 1 - parity  # flip parity for next layer

        # Conditioner MLP: input dim = dim, output dim = 2 * dim
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

    # Parameters for the whole flow.
    params = {
        "base": base_params,
        "transform": block_params,
    }

    flow = Flow(base_dist=base_dist, transform=transform)
    return flow, params