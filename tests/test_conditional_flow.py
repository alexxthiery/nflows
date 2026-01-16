# tests/test_conditional_flow.py
"""
Pytest tests for conditional normalizing flows.

Run with:
    PYTHONPATH=. pytest tests/test_conditional_flow.py -v
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflows.builders import build_realnvp, build_spline_realnvp


# ---------------------------------------------------------------------------
# Fixtures - module-scoped to avoid JIT recompilation overhead
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def dim():
    return 4


@pytest.fixture(scope="module")
def context_dim():
    return 2


@pytest.fixture(scope="module")
def realnvp_flow(key, dim, context_dim):
    """Conditional RealNVP flow."""
    flow, params = build_realnvp(
        key, dim=dim, num_layers=2, hidden_dim=16, n_hidden_layers=1,
        context_dim=context_dim
    )
    return flow, params


@pytest.fixture(scope="module")
def realnvp_flow_uncond(key, dim):
    """Unconditional RealNVP flow."""
    flow, params = build_realnvp(
        key, dim=dim, num_layers=2, hidden_dim=16, n_hidden_layers=1
    )
    return flow, params


@pytest.fixture(scope="module")
def spline_flow(key, dim, context_dim):
    """Conditional Spline flow."""
    flow, params = build_spline_realnvp(
        key, dim=dim, num_layers=2, hidden_dim=16, n_hidden_layers=1,
        context_dim=context_dim, num_bins=8
    )
    return flow, params


@pytest.fixture(scope="module")
def spline_flow_uncond(key, dim):
    """Unconditional Spline flow."""
    flow, params = build_spline_realnvp(
        key, dim=dim, num_layers=2, hidden_dim=16, n_hidden_layers=1,
        num_bins=8
    )
    return flow, params


def perturb_params(p, key):
    """Perturb params to break zero-init symmetry."""
    if isinstance(p, dict):
        return {k: perturb_params(v, jax.random.fold_in(key, hash(k) % 2**31)) for k, v in p.items()}
    elif isinstance(p, list):
        return [perturb_params(v, jax.random.fold_in(key, i)) for i, v in enumerate(p)]
    elif hasattr(p, 'shape'):
        return p + 0.1 * jax.random.normal(key, p.shape)
    else:
        return p


# ---------------------------------------------------------------------------
# RealNVP Tests
# ---------------------------------------------------------------------------
class TestRealNVPConditional:
    """Tests for conditional RealNVP flow."""

    def test_invertibility_inverse_forward(self, realnvp_flow, key, dim, context_dim):
        """forward(inverse(x, c), c) ≈ x"""
        flow, params = realnvp_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-5, f"inverse->forward error: {error}"

    def test_invertibility_forward_inverse(self, realnvp_flow, key, dim, context_dim):
        """inverse(forward(z, c), c) ≈ z"""
        flow, params = realnvp_flow
        z = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        x, _ = flow.forward(params, z, context=context)
        z_reconstructed, _ = flow.inverse(params, x, context=context)

        error = jnp.abs(z - z_reconstructed).max()
        assert error < 1e-5, f"forward->inverse error: {error}"

    def test_log_det_consistency(self, realnvp_flow, key, dim, context_dim):
        """log_det_forward = -log_det_inverse"""
        flow, params = realnvp_flow
        z = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        x, log_det_fwd = flow.forward(params, z, context=context)
        _, log_det_inv = flow.inverse(params, x, context=context)

        error = jnp.abs(log_det_fwd + log_det_inv).max()
        assert error < 1e-5, f"log_det inconsistency: {error}"

    def test_context_affects_forward(self, realnvp_flow, key, dim, context_dim):
        """Different contexts produce different forward outputs."""
        flow, params = realnvp_flow
        params_perturbed = perturb_params(params, key)

        z = jax.random.normal(key, (50, dim))
        context1 = jnp.zeros((50, context_dim))
        context2 = jnp.ones((50, context_dim))

        x1, _ = flow.forward(params_perturbed, z, context=context1)
        x2, _ = flow.forward(params_perturbed, z, context=context2)

        diff = jnp.abs(x1 - x2).mean()
        assert diff > 1e-3, f"Context has no effect on forward: diff={diff}"

    def test_context_affects_log_prob(self, realnvp_flow, key, dim, context_dim):
        """Different contexts produce different log_prob."""
        flow, params = realnvp_flow
        params_perturbed = perturb_params(params, key)

        x = jax.random.normal(key, (50, dim))
        context1 = jnp.zeros((50, context_dim))
        context2 = jnp.ones((50, context_dim))

        lp1 = flow.log_prob(params_perturbed, x, context=context1)
        lp2 = flow.log_prob(params_perturbed, x, context=context2)

        diff = jnp.abs(lp1 - lp2).mean()
        assert diff > 1e-3, f"Context has no effect on log_prob: diff={diff}"

    def test_sample_and_log_prob_consistency(self, realnvp_flow, key, dim, context_dim):
        """sample_and_log_prob matches separate sample + log_prob."""
        flow, params = realnvp_flow
        context = jax.random.normal(key, (20, context_dim))

        samples, log_prob_combined = flow.sample_and_log_prob(params, key, (20,), context=context)
        log_prob_separate = flow.log_prob(params, samples, context=context)

        error = jnp.abs(log_prob_combined - log_prob_separate).max()
        assert error < 1e-5, f"sample_and_log_prob inconsistency: {error}"

    def test_gradient_flows_through_context(self, realnvp_flow, key, dim, context_dim):
        """Gradients w.r.t. context are non-zero."""
        flow, params = realnvp_flow
        params_perturbed = perturb_params(params, key)

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        def loss_fn(context):
            return flow.log_prob(params_perturbed, x, context=context).sum()

        grad_context = jax.grad(loss_fn)(context)
        grad_norm = jnp.linalg.norm(grad_context)

        assert grad_norm > 0, "No gradient flow through context"

    def test_context_broadcasting_per_sample(self, realnvp_flow, key, dim, context_dim):
        """Per-sample context (batch, context_dim) works."""
        flow, params = realnvp_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        log_prob = flow.log_prob(params, x, context=context)

        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

    def test_context_broadcasting_shared(self, realnvp_flow, key, dim, context_dim):
        """Shared context (context_dim,) broadcasts correctly."""
        flow, params = realnvp_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (context_dim,))

        log_prob = flow.log_prob(params, x, context=context)

        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

class TestRealNVPUnconditional:
    """Tests for unconditional RealNVP (context=None)."""

    def test_sample_works(self, realnvp_flow_uncond, key, dim):
        flow, params = realnvp_flow_uncond
        samples = flow.sample(params, key, (10,))
        assert samples.shape == (10, dim)

    def test_log_prob_works(self, realnvp_flow_uncond, key, dim):
        flow, params = realnvp_flow_uncond
        x = jax.random.normal(key, (10, dim))
        log_prob = flow.log_prob(params, x)
        assert log_prob.shape == (10,)
        assert not jnp.isnan(log_prob).any()

    def test_invertibility(self, realnvp_flow_uncond, key, dim):
        flow, params = realnvp_flow_uncond
        x = jax.random.normal(key, (10, dim))

        z, _ = flow.inverse(params, x)
        x_back, _ = flow.forward(params, z)

        error = jnp.abs(x - x_back).max()
        assert error < 1e-5


# ---------------------------------------------------------------------------
# Spline Flow Tests
# ---------------------------------------------------------------------------
class TestSplineConditional:
    """Tests for conditional Spline flow."""

    def test_invertibility_inverse_forward(self, spline_flow, key, dim, context_dim):
        """forward(inverse(x, c), c) ≈ x"""
        flow, params = spline_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-4, f"inverse->forward error: {error}"

    def test_invertibility_forward_inverse(self, spline_flow, key, dim, context_dim):
        """inverse(forward(z, c), c) ≈ z"""
        flow, params = spline_flow
        z = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        x, _ = flow.forward(params, z, context=context)
        z_reconstructed, _ = flow.inverse(params, x, context=context)

        error = jnp.abs(z - z_reconstructed).max()
        assert error < 1e-4, f"forward->inverse error: {error}"

    def test_log_det_consistency(self, spline_flow, key, dim, context_dim):
        """log_det_forward = -log_det_inverse"""
        flow, params = spline_flow
        z = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        x, log_det_fwd = flow.forward(params, z, context=context)
        _, log_det_inv = flow.inverse(params, x, context=context)

        error = jnp.abs(log_det_fwd + log_det_inv).max()
        assert error < 1e-4, f"log_det inconsistency: {error}"

    def test_context_affects_forward(self, spline_flow, key, dim, context_dim):
        """Different contexts produce different forward outputs."""
        flow, params = spline_flow
        params_perturbed = perturb_params(params, key)

        z = jax.random.normal(key, (50, dim))
        context1 = jnp.zeros((50, context_dim))
        context2 = jnp.ones((50, context_dim))

        x1, _ = flow.forward(params_perturbed, z, context=context1)
        x2, _ = flow.forward(params_perturbed, z, context=context2)

        diff = jnp.abs(x1 - x2).mean()
        assert diff > 1e-3, f"Context has no effect on forward: diff={diff}"

    def test_context_affects_log_prob(self, spline_flow, key, dim, context_dim):
        """Different contexts produce different log_prob."""
        flow, params = spline_flow
        params_perturbed = perturb_params(params, key)

        x = jax.random.normal(key, (50, dim))
        context1 = jnp.zeros((50, context_dim))
        context2 = jnp.ones((50, context_dim))

        lp1 = flow.log_prob(params_perturbed, x, context=context1)
        lp2 = flow.log_prob(params_perturbed, x, context=context2)

        diff = jnp.abs(lp1 - lp2).mean()
        assert diff > 1e-3, f"Context has no effect on log_prob: diff={diff}"

    def test_sample_and_log_prob_consistency(self, spline_flow, key, dim, context_dim):
        """sample_and_log_prob matches separate sample + log_prob."""
        flow, params = spline_flow
        context = jax.random.normal(key, (20, context_dim))

        samples, log_prob_combined = flow.sample_and_log_prob(params, key, (20,), context=context)
        log_prob_separate = flow.log_prob(params, samples, context=context)

        error = jnp.abs(log_prob_combined - log_prob_separate).max()
        assert error < 1e-4, f"sample_and_log_prob inconsistency: {error}"

    def test_gradient_flows_through_context(self, spline_flow, key, dim, context_dim):
        """Gradients w.r.t. context are non-zero."""
        flow, params = spline_flow
        params_perturbed = perturb_params(params, key)

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        def loss_fn(context):
            return flow.log_prob(params_perturbed, x, context=context).sum()

        grad_context = jax.grad(loss_fn)(context)
        grad_norm = jnp.linalg.norm(grad_context)

        assert grad_norm > 0, "No gradient flow through context"

    def test_context_broadcasting_per_sample(self, spline_flow, key, dim, context_dim):
        """Per-sample context (batch, context_dim) works."""
        flow, params = spline_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        log_prob = flow.log_prob(params, x, context=context)

        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

    def test_context_broadcasting_shared(self, spline_flow, key, dim, context_dim):
        """Shared context (context_dim,) broadcasts correctly."""
        flow, params = spline_flow
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (context_dim,))

        log_prob = flow.log_prob(params, x, context=context)

        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

class TestSplineUnconditional:
    """Tests for unconditional Spline flow (context=None)."""

    def test_sample_works(self, spline_flow_uncond, key, dim):
        flow, params = spline_flow_uncond
        samples = flow.sample(params, key, (10,))
        assert samples.shape == (10, dim)

    def test_log_prob_works(self, spline_flow_uncond, key, dim):
        flow, params = spline_flow_uncond
        x = jax.random.normal(key, (10, dim))
        log_prob = flow.log_prob(params, x)
        assert log_prob.shape == (10,)
        assert not jnp.isnan(log_prob).any()

    def test_invertibility(self, spline_flow_uncond, key, dim):
        flow, params = spline_flow_uncond
        x = jax.random.normal(key, (10, dim))

        z, _ = flow.inverse(params, x)
        x_back, _ = flow.forward(params, z)

        error = jnp.abs(x - x_back).max()
        assert error < 1e-4


# ---------------------------------------------------------------------------
# Context Feature Extractor Tests
# ---------------------------------------------------------------------------
class TestContextFeatureExtractor:
    """Tests for flows with context feature extractor."""

    @pytest.fixture(scope="class")
    def realnvp_with_extractor(self, key, dim, context_dim):
        """RealNVP flow with context feature extractor."""
        flow, params = build_realnvp(
            key,
            dim=dim,
            num_layers=2,
            hidden_dim=16,
            n_hidden_layers=1,
            context_dim=context_dim,
            context_extractor_hidden_dim=32,
            context_extractor_n_layers=2,
        )
        return flow, params

    @pytest.fixture(scope="class")
    def realnvp_with_extractor_diff_dim(self, key, dim, context_dim):
        """RealNVP with extractor outputting different dimension."""
        flow, params = build_realnvp(
            key,
            dim=dim,
            num_layers=2,
            hidden_dim=16,
            n_hidden_layers=1,
            context_dim=context_dim,
            context_extractor_hidden_dim=32,
            context_extractor_n_layers=2,
            context_feature_dim=8,  # Different from context_dim
        )
        return flow, params

    @pytest.fixture(scope="class")
    def spline_with_extractor(self, key, dim, context_dim):
        """Spline flow with context feature extractor."""
        flow, params = build_spline_realnvp(
            key,
            dim=dim,
            num_layers=2,
            hidden_dim=16,
            n_hidden_layers=1,
            context_dim=context_dim,
            context_extractor_hidden_dim=32,
            context_extractor_n_layers=2,
            num_bins=8,
        )
        return flow, params

    def test_has_feature_extractor_params(self, realnvp_with_extractor):
        """Params include feature_extractor when enabled."""
        flow, params = realnvp_with_extractor
        assert flow.feature_extractor is not None
        assert "feature_extractor" in params

    def test_realnvp_invertibility(self, realnvp_with_extractor, key, dim, context_dim):
        """Invertibility holds with feature extractor."""
        flow, params = realnvp_with_extractor
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-5, f"inverse->forward error with extractor: {error}"

    def test_spline_invertibility(self, spline_with_extractor, key, dim, context_dim):
        """Invertibility holds with feature extractor for spline flows."""
        flow, params = spline_with_extractor
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-4, f"inverse->forward error with extractor: {error}"

    def test_extractor_changes_output(self, realnvp_flow, realnvp_with_extractor, key, dim, context_dim):
        """Feature extractor produces different outputs than raw context."""
        flow_no_ext, params_no_ext = realnvp_flow
        flow_with_ext, params_with_ext = realnvp_with_extractor

        # Perturb both to break symmetry
        params_no_ext = perturb_params(params_no_ext, key)
        params_with_ext = perturb_params(params_with_ext, jax.random.fold_in(key, 1))

        z = jax.random.normal(key, (50, dim))
        context = jax.random.normal(key, (50, context_dim))

        x_no_ext, _ = flow_no_ext.forward(params_no_ext, z, context=context)
        x_with_ext, _ = flow_with_ext.forward(params_with_ext, z, context=context)

        # Different flows should produce different outputs
        diff = jnp.abs(x_no_ext - x_with_ext).mean()
        assert diff > 1e-3, f"Flows with/without extractor should differ: diff={diff}"

    def test_gradient_flows_through_extractor(self, realnvp_with_extractor, key, dim, context_dim):
        """Gradients flow through the feature extractor params."""
        flow, params = realnvp_with_extractor
        params_perturbed = perturb_params(params, key)

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        def loss_fn(params):
            return flow.log_prob(params, x, context=context).sum()

        grads = jax.grad(loss_fn)(params_perturbed)

        # Check that feature extractor params have non-zero gradients
        fe_grads = grads["feature_extractor"]
        grad_norm = sum(
            jnp.linalg.norm(v) for v in jax.tree_util.tree_leaves(fe_grads)
        )
        assert grad_norm > 0, "No gradient flow through feature extractor"

    def test_different_context_feature_dim(self, realnvp_with_extractor_diff_dim, key, dim, context_dim):
        """Flow works with context_feature_dim != context_dim."""
        flow, params = realnvp_with_extractor_diff_dim
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (20, context_dim))

        # Should run without error
        log_prob = flow.log_prob(params, x, context=context)
        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

        # Sampling should work (use matching batch size for context)
        sample_context = jax.random.normal(key, (10, context_dim))
        samples = flow.sample(params, key, (10,), context=sample_context)
        assert samples.shape == (10, dim)

    def test_sample_and_log_prob_with_extractor(self, realnvp_with_extractor, key, dim, context_dim):
        """sample_and_log_prob works with feature extractor."""
        flow, params = realnvp_with_extractor
        context = jax.random.normal(key, (20, context_dim))

        samples, log_prob_combined = flow.sample_and_log_prob(params, key, (20,), context=context)
        log_prob_separate = flow.log_prob(params, samples, context=context)

        error = jnp.abs(log_prob_combined - log_prob_separate).max()
        assert error < 1e-5, f"sample_and_log_prob inconsistency: {error}"

    def test_context_broadcasting_shared_with_extractor(self, realnvp_with_extractor, key, dim, context_dim):
        """Shared context broadcasting works with feature extractor."""
        flow, params = realnvp_with_extractor
        x = jax.random.normal(key, (20, dim))
        context = jax.random.normal(key, (context_dim,))  # Shared

        log_prob = flow.log_prob(params, x, context=context)
        assert log_prob.shape == (20,)
        assert not jnp.isnan(log_prob).any()

    def test_no_extractor_backward_compatible(self, realnvp_flow):
        """Flows without extractor have no feature_extractor field in params."""
        flow, params = realnvp_flow
        assert flow.feature_extractor is None
        assert "feature_extractor" not in params
