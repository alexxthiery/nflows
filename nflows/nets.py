# nflows/nets.py
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


Array = jnp.ndarray
PRNGKey = jax.Array  # type alias for JAX random keys


class MLP(nn.Module):
    """
    Simple fully connected network used as a conditioner in flow layers.

    Assumptions:
      - Inputs have shape (..., in_dim).
      - Last axis is the feature dimension.
      - Network is purely feedforward, no dropout or batch statistics.
    """
    in_dim: int
    hidden_sizes: Sequence[int]
    out_dim: int
    activation: Callable[[Array], Array] = nn.tanh
    use_bias: bool = True
    kernel_init: Callable[..., Array] = nn.initializers.lecun_normal()
    bias_init: Callable[..., Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Defensive shape check: last dimension must match declared in_dim.
        if x.ndim < 1:
            raise ValueError(
                f"MLP expected input with at least 1 dimension, got shape {x.shape}."
            )
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"MLP expected last dimension {self.in_dim}, got {x.shape[-1]}."
            )

        h = x
        for i, size in enumerate(self.hidden_sizes):
            h = nn.Dense(
                features=size,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"dense_{i}",
            )(h)
            h = self.activation(h)

        # Final linear layer without activation.
        h = nn.Dense(
            features=self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="dense_out",
        )(h)
        return h


def init_mlp(
    key: PRNGKey,
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: Callable[[Array], Array] = nn.tanh,
) -> Tuple[MLP, dict]:
    """
    Construct an MLP module and initialize its parameters.

    Arguments:
      key: JAX PRNGKey used for parameter initialization.
      in_dim: Size of the input feature dimension.
      hidden_sizes: Sizes of the hidden layers.
      out_dim: Size of the output feature dimension.
      activation: Activation function applied after each hidden layer.

    Returns:
      mlp: A Flax MLP module (definition only, no params inside).
      params: A PyTree of parameters for this MLP (suitable for mlp.apply).
    """

    mlp = MLP(
        in_dim=in_dim,
        hidden_sizes=tuple(hidden_sizes),
        out_dim=out_dim,
        activation=activation,
    )
    B = 1  # Dummy batch size for initialization.
    dummy_x = jnp.zeros((B, in_dim), dtype=jnp.float32)
    variables = mlp.init(key, dummy_x)

    # We only keep the "params" collection. No batch_stats etc here.
    params = variables.get("params", {})
    return mlp, params