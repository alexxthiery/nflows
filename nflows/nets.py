# nflows/nets.py
from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


Array = jnp.ndarray
PRNGKey = jax.Array  # type alias for JAX random keys


class ResNet(nn.Module):
    """
    Residual MLP block: input → hidden → residual blocks → output.

    A reusable residual network architecture that can be used standalone
    or as a building block within other modules (e.g., MLP, feature extractors).

    Architecture:
      1. Input projection: x → Dense(hidden_dim) → h
      2. Residual trunk: h = h + res_scale * F(h) for each layer
         where F = Dense → activation → Dense
      3. Output projection: Dense(out_dim)

    Attributes:
        hidden_dim: Width of residual hidden stream.
        out_dim: Output dimensionality.
        n_hidden_layers: Number of residual blocks.
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
    """

    hidden_dim: int
    out_dim: int
    n_hidden_layers: int = 2
    activation: Callable[[Array], Array] = nn.elu
    res_scale: float = 0.1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through the residual network.

        Arguments:
            x: Input tensor of shape (..., in_dim). Input dimension is inferred.

        Returns:
            Output tensor of shape (..., out_dim).
        """
        # 1. Input projection.
        h = nn.Dense(self.hidden_dim, name="dense_in")(x)

        # 2. Residual trunk.
        for i in range(self.n_hidden_layers):
            r = nn.Dense(self.hidden_dim, name=f"res_{i}_dense0")(h)
            r = self.activation(r)
            r = nn.Dense(self.hidden_dim, name=f"res_{i}_dense1")(r)
            h = h + self.res_scale * r

        # 3. Output projection.
        out = nn.Dense(self.out_dim, name="dense_out")(h)
        return out


class MLP(nn.Module):
    """
    Residual MLP conditioner for flow layers.

    Wraps a ResNet with context handling (validation, broadcasting, concatenation).
    The underlying ResNet is stored under the "net" submodule.

    The final layer ("net/dense_out") should be zero-initialized externally
    (via init_mlp) to start the flow at identity.

    Attributes:
        x_dim: Dimensionality of input x.
        context_dim: Dimensionality of context (0 for unconditional).
        hidden_dim: Width of residual hidden stream.
        n_hidden_layers: Number of residual blocks.
        out_dim: Output dimensionality (e.g., coupling parameters).
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
    """
    x_dim: int
    context_dim: int = 0
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    out_dim: int = 1
    activation: Callable[[Array], Array] = nn.elu
    res_scale: float = 0.1

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        """
        Forward pass through the residual MLP.

        Arguments:
            x: Input tensor of shape (..., x_dim).
            context: Optional context tensor:
                - None (unconditional)
                - shape (context_dim,) for shared context
                - shape (..., context_dim) for per-sample context

        Returns:
            Output tensor of shape (..., out_dim).
        """
        # Shape check for x.
        if x.shape[-1] != self.x_dim:
            raise ValueError(
                f"MLP expected x with last dimension {self.x_dim}, got {x.shape[-1]}."
            )

        # Context handling.
        if context is not None and self.context_dim > 0:
            # Check context feature dimension.
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"MLP expected context with last dimension {self.context_dim}, "
                    f"got {context.shape[-1]}."
                )
            # Broadcast shared context to batch dimensions.
            if context.ndim == 1:
                context = jnp.broadcast_to(context, x.shape[:-1] + (self.context_dim,))
            elif context.shape[:-1] != x.shape[:-1]:
                raise ValueError(
                    f"Context batch shape {context.shape[:-1]} doesn't match "
                    f"x batch shape {x.shape[:-1]}."
                )
            inp = jnp.concatenate([x, context], axis=-1)
        else:
            inp = x

        # Delegate to ResNet for the actual computation.
        net = ResNet(
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            res_scale=self.res_scale,
            name="net",
        )
        return net(inp)


def init_mlp(
    key: PRNGKey,
    x_dim: int,
    context_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    out_dim: int,
    activation: Callable[[Array], Array] = nn.elu,
    res_scale: float = 0.1,
) -> Tuple[MLP, dict]:
    """
    Construct a residual MLP module and initialize its parameters.

    Arguments:
        key: JAX PRNGKey used for parameter initialization.
        x_dim: Dimensionality of input x.
        context_dim: Dimensionality of context (0 for unconditional).
        hidden_dim: Width of residual hidden stream.
        n_hidden_layers: Number of residual blocks.
        out_dim: Output dimensionality.
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).

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
        x_dim=x_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        out_dim=out_dim,
        activation=activation,
        res_scale=res_scale,
    )

    # Dummy inputs for initialization.
    B = 1
    dummy_x = jnp.zeros((B, x_dim), dtype=jnp.float32)
    dummy_context = jnp.zeros((B, context_dim), dtype=jnp.float32) if context_dim > 0 else None
    variables = mlp.init(key, dummy_x, dummy_context)

    # Copy params to mutable dict for zero-init of final layer.
    raw_params = variables.get("params", {})
    params = dict(raw_params)

    # Zero-initialize the final Dense layer for identity-start flows.
    # Params are nested under "net" (the ResNet submodule).
    if "net" not in params or "dense_out" not in params["net"]:
        raise KeyError(
            "init_mlp expected a 'net/dense_out' parameter collection in the MLP; "
            "check the MLP implementation if this error occurs."
        )
    net_params = dict(params["net"])
    dense_out = dict(net_params["dense_out"])
    dense_out["kernel"] = jnp.zeros_like(dense_out["kernel"])
    dense_out["bias"] = jnp.zeros_like(dense_out["bias"])
    net_params["dense_out"] = dense_out
    params["net"] = net_params

    return mlp, params


def init_resnet(
    key: PRNGKey,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden_layers: int = 2,
    activation: Callable[[Array], Array] = nn.elu,
    res_scale: float = 0.1,
    zero_init_output: bool = False,
) -> Tuple[ResNet, dict]:
    """
    Construct a ResNet module and initialize its parameters.

    Arguments:
        key: JAX PRNGKey used for parameter initialization.
        in_dim: Dimensionality of input (used for dummy input during init).
        hidden_dim: Width of residual hidden stream.
        out_dim: Output dimensionality.
        n_hidden_layers: Number of residual blocks (default: 2).
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
        zero_init_output: If True, zero-initialize the output layer (default: False).

    Returns:
        resnet: A Flax ResNet module (definition only, no params inside).
        params: A PyTree of parameters for this ResNet (suitable for resnet.apply).
    """
    resnet = ResNet(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        res_scale=res_scale,
    )

    # Dummy input for initialization.
    dummy_x = jnp.zeros((1, in_dim), dtype=jnp.float32)
    variables = resnet.init(key, dummy_x)

    raw_params = variables.get("params", {})
    params = dict(raw_params)

    # Optionally zero-initialize the output layer.
    if zero_init_output:
        if "dense_out" not in params:
            raise KeyError(
                "init_resnet expected a 'dense_out' parameter collection; "
                "check the ResNet implementation if this error occurs."
            )
        dense_out = dict(params["dense_out"])
        dense_out["kernel"] = jnp.zeros_like(dense_out["kernel"])
        dense_out["bias"] = jnp.zeros_like(dense_out["bias"])
        params["dense_out"] = dense_out

    return resnet, params