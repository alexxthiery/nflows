# nflows/flows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.Array  # JAX random key alias


@dataclass
class Flow:
    """
    Normalizing flow distribution.

    A flow is defined by:
      - a base distribution p_base(z) with methods:
          log_prob(base_params, z)
          sample(base_params, key, shape)
      - an invertible transform T with methods:
          forward(transform_params, z) -> (x, log |det ∂x/∂z|)
          inverse(transform_params, x) -> (z, log |det ∂z/∂x|)

    Parameters for the whole flow are passed explicitly as a PyTree `params`
    with the following convention:

      params["base"]      -> parameters for the base distribution (or {} / None)
      params["transform"] -> parameters for the transform

    This class only stores the *definitions* of base_dist and transform.
    It does not store any trainable parameters.
    """
    base_dist: Any
    transform: Any

    # --------------------------------------------------------------
    # Low-level: forward / inverse maps
    # --------------------------------------------------------------
    def forward(self, params: Any, z: Array) -> Tuple[Array, Array]:
        """
        Forward map: latent z -> x.

        Arguments:
          params: PyTree with at least key "transform".
          z: latent samples, shape (..., dim).

        Returns:
          x: transformed samples, shape (..., dim).
          log_det: log |det ∂x/∂z|, shape (...,).
        """
        transform_params = params["transform"]
        x, log_det = self.transform.forward(transform_params, z)
        return x, log_det

    def inverse(self, params: Any, x: Array) -> Tuple[Array, Array]:
        """
        Inverse map: x -> latent z.

        Arguments:
          params: PyTree with at least key "transform".
          x: samples in data space, shape (..., dim).

        Returns:
          z: latent samples, shape (..., dim).
          log_det: log |det ∂z/∂x|, shape (...,).
        """
        transform_params = params["transform"]
        z, log_det = self.transform.inverse(transform_params, x)
        return z, log_det

    # --------------------------------------------------------------
    # Distribution interface: log_prob / sample
    # --------------------------------------------------------------
    def log_prob(self, params: Any, x: Array) -> Array:
        """
        Log-density of the flow at x.

        Uses the change-of-variables formula:
          z, log_det_inv = inverse(params, x)
          log q(x) = log p_base(z) + log_det_inv

        Arguments:
          params: PyTree with keys "base" and "transform".
          x: samples in data space, shape (..., dim).

        Returns:
          log_prob: array with shape (...,).
        """
        base_params = params["base"]
        z, log_det_inv = self.inverse(params, x)
        base_log_prob = self.base_dist.log_prob(base_params, z)
        return base_log_prob + log_det_inv

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        """
        Draw samples from the flow distribution.

        Samples z from the base distribution and then applies the forward map.

        Arguments:
          params: PyTree with keys "base" and "transform".
          key: JAX PRNGKey.
          shape: batch shape for samples (excluding event dimension).

        Returns:
          x: samples in data space, shape (*shape, dim).
        """
        base_params = params["base"]
        # We only need a single key here, but split for possible future use.
        key_base, _ = jax.random.split(key)
        z = self.base_dist.sample(base_params, key_base, shape)
        x, _ = self.forward(params, z)
        return x
      
    def sample_and_log_prob(
        self, params: Any, key: PRNGKey, shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        """
        Draw samples from the flow and compute their log-density q(x) in a single pass.

        This function is an optimized alternative to:
            x = self.sample(params, key, shape)
            log_q = self.log_prob(params, x)
        which would perform one forward transformation (for sampling) and
        one inverse transformation (to compute z and log-det), even though
        all required quantities are available from a single forward pass.

        Operation:
          1. Sample latent variables z ~ p_base(z).
          2. Apply the forward flow transformation:
                x, log_det_fwd = T(z)
            where log_det_fwd is log |det ∂x/∂z|.
          3. Use the change-of-variables formula:
                log q(x) = log p_base(z) - log_det_fwd.

        Why it is useful:
          In many VI objectives (e.g. reverse KL, ELBO expectations, AIS,
          score-function estimators), expectations are taken under q(x):
                E_{x ~ q} [ f(x, log q(x)) ].
          When sampling from q, one never needs the inverse map T⁻¹(x).
          Using the forward Jacobian reduces computation, improves numerical
          stability, and avoids unnecessary evaluation of the conditioner.
          This is the correct and minimal way to obtain (x, log q(x)) when
          x is generated by the flow itself.

        Arguments:
          params: PyTree with keys "base" and "transform".
          key: JAX PRNGKey for sampling.
          shape: batch shape for samples (excluding event dimension).

        Returns:
          x: samples from q(x), shape (*shape, dim).
          log_q: log q(x) evaluated at these samples, shape (*shape,).
        """
        base_params = params["base"]
        transform_params = params["transform"]

        key_z, _ = jax.random.split(key)
        z = self.base_dist.sample(base_params, key_z, shape)
        x, log_det_fwd = self.transform.forward(transform_params, z)
        base_log = self.base_dist.log_prob(base_params, z)
        log_q = base_log - log_det_fwd
        return x, log_q