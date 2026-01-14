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
    activation: Callable[[Array], Array] = nn.elu
    use_bias: bool = True
    kernel_init: Callable[..., Array] = nn.initializers.lecun_normal()
    bias_init: Callable[..., Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        """
        Forward pass through the MLP.

        Arguments:
            x: Input tensor of shape (..., x_dim).
            context: Optional context tensor of shape (..., context_dim) or (context_dim,).
                     If provided, it is concatenated to x before processing, so
                     in_dim should equal x_dim + context_dim. If None, x is used
                     directly and in_dim should equal x_dim.

        Returns:
            Output tensor of shape (..., out_dim).
        """
        # Concatenate context to input if provided.
        if context is not None:
            # Broadcast context to match x batch dimensions if needed.
            if context.ndim < x.ndim:
                # context has shape (context_dim,), broadcast to (..., context_dim)
                context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))
            h = jnp.concatenate([x, context], axis=-1)
        else:
            h = x

        # Defensive shape check: last dimension must match declared in_dim.
        if h.ndim < 1:
            raise ValueError(
                f"MLP expected input with at least 1 dimension, got shape {h.shape}."
            )
        if h.shape[-1] != self.in_dim:
            raise ValueError(
                f"MLP expected last dimension {self.in_dim}, got {h.shape[-1]}."
            )
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
    activation: Callable[[Array], Array] = nn.elu,
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

    Notes:
      The final linear layer ("dense_out") is explicitly zero-initialized
      (kernel and bias set to zero). This makes the initial output of the
      conditioner identically zero, so any flow that interprets the output
      as shift/log_scale starts exactly at the identity transform.
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

    # Flax returns a FrozenDict for "params". We copy it to a mutable dict so
    # we can override the final layer weights and bias.
    raw_params = variables.get("params", {})
    params = dict(raw_params)  # shallow copy of top-level mapping

    # ------------------------------------------------------------------
    # Zero-initialize the final Dense layer:
    # Rationale: for flow conditioners, this makes the initial output zero,
    # so the flow starts as the identity transform. This is known to greatly
    # stabilize training since the flow does not start with extreme transforms.
    # ------------------------------------------------------------------
    zero_init_final_layer = True
    if zero_init_final_layer:
        if "dense_out" not in params:
            raise KeyError(
                "init_mlp expected a 'dense_out' parameter collection in the MLP; "
                "check the MLP implementation if this error occurs."
            )

        # Zero-initialize final layer: kernel and bias.
        dense_out = dict(params["dense_out"])
        dense_out["kernel"] = jnp.zeros_like(dense_out["kernel"])
        dense_out["bias"] = jnp.zeros_like(dense_out["bias"])
        params["dense_out"] = dense_out

    return mlp, params