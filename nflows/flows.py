# nflows/flows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.Array  # JAX random key alias


@dataclass
class Bijection:
    """
    Invertible transformation with tractable Jacobian, optionally conditioned on context.

    This is a lightweight wrapper that composes a transform with an optional
    context feature extractor. Unlike Flow, it does not define a probability
    distribution—just the forward/inverse maps.

    Use cases:
      - Change of variables in integration
      - Learned coordinate transformations
      - Composing with custom base distributions
      - Reparameterization tricks

    Parameters structure:
      params["transform"]         -> parameters for the transform
      params["feature_extractor"] -> parameters for context feature extractor (if used)

    This class only stores the *definitions* of transform and feature_extractor.
    It does not store any trainable parameters.
    """
    transform: Any
    feature_extractor: Any = None

    def _extract_context_features(self, params: Any, context: Array | None) -> Array | None:
        """
        Extract features from context using the feature extractor if available.

        Arguments:
          params: PyTree that may contain "feature_extractor" key.
          context: raw context tensor, or None.

        Returns:
          Extracted context features, or original context if no extractor.
        """
        if self.feature_extractor is None or context is None:
            return context
        fe_params = params["feature_extractor"]
        return self.feature_extractor.apply({"params": fe_params}, context)

    def forward(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward map: x -> y.

        Arguments:
          params: PyTree with key "transform" (and optionally "feature_extractor").
          x: input samples, shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          y: transformed samples, shape (..., dim).
          log_det: log |det ∂y/∂x|, shape (...,).
        """
        context_features = self._extract_context_features(params, context)
        transform_params = params["transform"]
        return self.transform.forward(transform_params, x, context_features)

    def inverse(self, params: Any, y: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x.

        Arguments:
          params: PyTree with key "transform" (and optionally "feature_extractor").
          y: transformed samples, shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          x: original samples, shape (..., dim).
          log_det: log |det ∂x/∂y|, shape (...,).
        """
        context_features = self._extract_context_features(params, context)
        transform_params = params["transform"]
        return self.transform.inverse(transform_params, y, context_features)


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
      params["feature_extractor"] -> parameters for context feature extractor (optional)

    Optionally, a feature_extractor can be provided to learn a representation
    of the context before it is used in the flow. When provided, context is
    first passed through the feature extractor, and the resulting features
    are used in place of raw context throughout the flow.

    This class only stores the *definitions* of base_dist, transform, and
    feature_extractor. It does not store any trainable parameters.
    """
    base_dist: Any
    transform: Any
    feature_extractor: Any = None

    # --------------------------------------------------------------
    # Context feature extraction
    # --------------------------------------------------------------
    def _extract_context_features(self, params: Any, context: Array | None) -> Array | None:
        """
        Extract features from context using the feature extractor if available.

        Arguments:
          params: PyTree that may contain "feature_extractor" key.
          context: raw context tensor, or None.

        Returns:
          Extracted context features, or original context if no extractor.
        """
        if self.feature_extractor is None or context is None:
            return context
        fe_params = params["feature_extractor"]
        return self.feature_extractor.apply({"params": fe_params}, context)

    # --------------------------------------------------------------
    # Low-level: forward / inverse maps
    # --------------------------------------------------------------
    def forward(self, params: Any, z: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Forward map: latent z -> x.

        Arguments:
          params: PyTree with at least key "transform".
          z: latent samples, shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          x: transformed samples, shape (..., dim).
          log_det: log |det ∂x/∂z|, shape (...,).
        """
        context_features = self._extract_context_features(params, context)
        transform_params = params["transform"]
        x, log_det = self.transform.forward(transform_params, z, context_features)
        return x, log_det

    def inverse(self, params: Any, x: Array, context: Array | None = None) -> Tuple[Array, Array]:
        """
        Inverse map: x -> latent z.

        Arguments:
          params: PyTree with at least key "transform".
          x: samples in data space, shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          z: latent samples, shape (..., dim).
          log_det: log |det ∂z/∂x|, shape (...,).
        """
        context_features = self._extract_context_features(params, context)
        transform_params = params["transform"]
        z, log_det = self.transform.inverse(transform_params, x, context_features)
        return z, log_det

    # --------------------------------------------------------------
    # Distribution interface: log_prob / sample
    # --------------------------------------------------------------
    def log_prob(self, params: Any, x: Array, context: Array | None = None) -> Array:
        """
        Log-density of the flow at x.

        Uses the change-of-variables formula:
          z, log_det_inv = inverse(params, x, context)
          log q(x | context) = log p_base(z) + log_det_inv

        Arguments:
          params: PyTree with keys "base" and "transform".
          x: samples in data space, shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          log_prob: array with shape (...,).
        """
        base_params = params["base"]
        z, log_det_inv = self.inverse(params, x, context)
        base_log_prob = self.base_dist.log_prob(base_params, z)
        return base_log_prob + log_det_inv

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...], context: Array | None = None) -> Array:
        """
        Draw samples from the flow distribution.

        Samples z from the base distribution and then applies the forward map.

        Arguments:
          params: PyTree with keys "base" and "transform".
          key: JAX PRNGKey.
          shape: batch shape for samples (excluding event dimension).
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          x: samples in data space, shape (*shape, dim).
        """
        base_params = params["base"]
        # We only need a single key here, but split for possible future use.
        key_base, _ = jax.random.split(key)
        z = self.base_dist.sample(base_params, key_base, shape)
        x, _ = self.forward(params, z, context)
        return x

    def sample_and_log_prob(
        self, params: Any, key: PRNGKey, shape: Tuple[int, ...], context: Array | None = None
    ) -> Tuple[Array, Array]:
        """
        Draw samples from the flow and compute their log-density q(x) in a single pass.

        This function is an optimized alternative to:
            x = self.sample(params, key, shape, context)
            log_q = self.log_prob(params, x, context)
        which would perform one forward transformation (for sampling) and
        one inverse transformation (to compute z and log-det), even though
        all required quantities are available from a single forward pass.

        Operation:
          1. Sample latent variables z ~ p_base(z).
          2. Apply the forward flow transformation:
                x, log_det_fwd = T(z; context)
            where log_det_fwd is log |det ∂x/∂z|.
          3. Use the change-of-variables formula:
                log q(x | context) = log p_base(z) - log_det_fwd.

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
          context: optional conditioning tensor, shape (..., context_dim).

        Returns:
          x: samples from q(x | context), shape (*shape, dim).
          log_q: log q(x | context) evaluated at these samples, shape (*shape,).
        """
        context_features = self._extract_context_features(params, context)
        base_params = params["base"]
        transform_params = params["transform"]

        key_z, _ = jax.random.split(key)
        z = self.base_dist.sample(base_params, key_z, shape)
        x, log_det_fwd = self.transform.forward(transform_params, z, context_features)
        base_log = self.base_dist.log_prob(base_params, z)
        log_q = base_log - log_det_fwd
        return x, log_q