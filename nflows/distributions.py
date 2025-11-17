# nflows/distributions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.Array  # JAX random key alias


# ----------------------------------------------------------------------
# Standard Normal
# ----------------------------------------------------------------------
@dataclass
class StandardNormal:
    """
    Isotropic Gaussian N(0, I) in R^dim.

    x is expected to have shape (..., dim).

    params is ignored for this distribution; included only for API uniformity.
    """
    dim: int

    def log_prob(self, params: Any, x: Array) -> Array:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"StandardNormal: expected last dim {self.dim}, got {x.shape[-1]}"
            )

        quad = jnp.sum(x * x, axis=-1)
        log_norm = 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return -0.5 * quad - log_norm

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        return jax.random.normal(key, shape=shape + (self.dim,))


# ----------------------------------------------------------------------
# Diagonal Gaussian
# ----------------------------------------------------------------------
@dataclass
class DiagNormal:
    """
    Diagonal-covariance Gaussian N(loc, diag(scale^2)) in R^dim.

    Required params leaves:
      params["loc"]       shape (dim,)
      params["log_scale"] shape (dim,)

    Both log_prob and sample broadcast batch dimensions naturally.
    """
    dim: int

    def _extract_params(self, params: Any) -> Tuple[Array, Array]:
        try:
            loc = jnp.asarray(params["loc"])
            log_scale = jnp.asarray(params["log_scale"])
        except Exception as e:
            raise KeyError(
                "DiagNormal expected params to contain 'loc' and 'log_scale'"
            ) from e

        if loc.shape != (self.dim,):
            raise ValueError(
                f"DiagNormal: loc must have shape ({self.dim},), got {loc.shape}"
            )
        if log_scale.shape != (self.dim,):
            raise ValueError(
                f"DiagNormal: log_scale must have shape ({self.dim},), got {log_scale.shape}"
            )

        return loc, log_scale

    def log_prob(self, params: Any, x: Array) -> Array:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"DiagNormal: expected last dim {self.dim}, got {x.shape[-1]}"
            )

        loc, log_scale = self._extract_params(params)
        scale = jnp.exp(log_scale)

        z = (x - loc) / scale
        quad = jnp.sum(z * z, axis=-1)
        log_norm = 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        log_det = jnp.sum(log_scale)

        return -0.5 * quad - log_norm - log_det

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        loc, log_scale = self._extract_params(params)
        scale = jnp.exp(log_scale)

        eps = jax.random.normal(key, shape=shape + (self.dim,))
        return loc + eps * scale