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
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def dim():
    return 4


@pytest.fixture
def context_dim():
    return 2


@pytest.fixture
def realnvp_flow(key, dim, context_dim):
    """Conditional RealNVP flow."""
    flow, params = build_realnvp(
        key, dim=dim, num_layers=4, hidden_sizes=[32, 32],
        context_dim=context_dim
    )
    return flow, params


@pytest.fixture
def realnvp_flow_uncond(key, dim):
    """Unconditional RealNVP flow."""
    flow, params = build_realnvp(
        key, dim=dim, num_layers=4, hidden_sizes=[32, 32]
    )
    return flow, params


@pytest.fixture
def spline_flow(key, dim, context_dim):
    """Conditional Spline flow."""
    flow, params = build_spline_realnvp(
        key, dim=dim, num_layers=4, hidden_sizes=[32, 32],
        context_dim=context_dim, num_bins=8
    )
    return flow, params


@pytest.fixture
def spline_flow_uncond(key, dim):
    """Unconditional Spline flow."""
    flow, params = build_spline_realnvp(
        key, dim=dim, num_layers=4, hidden_sizes=[32, 32],
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
        x = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-5, f"inverse->forward error: {error}"

    def test_invertibility_forward_inverse(self, realnvp_flow, key, dim, context_dim):
        """inverse(forward(z, c), c) ≈ z"""
        flow, params = realnvp_flow
        z = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

        x, _ = flow.forward(params, z, context=context)
        z_reconstructed, _ = flow.inverse(params, x, context=context)

        error = jnp.abs(z - z_reconstructed).max()
        assert error < 1e-5, f"forward->inverse error: {error}"

    def test_log_det_consistency(self, realnvp_flow, key, dim, context_dim):
        """log_det_forward = -log_det_inverse"""
        flow, params = realnvp_flow
        z = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

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
        context = jax.random.normal(key, (100, context_dim))

        samples, log_prob_combined = flow.sample_and_log_prob(params, key, (100,), context=context)
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

    def test_jit_forward(self, realnvp_flow, key, dim, context_dim):
        """forward JIT-compiles."""
        flow, params = realnvp_flow
        z = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        forward_jit = jax.jit(lambda p, z, c: flow.forward(p, z, context=c))
        x, log_det = forward_jit(params, z, context)

        assert x.shape == (10, dim)
        assert log_det.shape == (10,)

    def test_jit_inverse(self, realnvp_flow, key, dim, context_dim):
        """inverse JIT-compiles."""
        flow, params = realnvp_flow
        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        inverse_jit = jax.jit(lambda p, x, c: flow.inverse(p, x, context=c))
        z, log_det = inverse_jit(params, x, context)

        assert z.shape == (10, dim)
        assert log_det.shape == (10,)

    def test_jit_log_prob(self, realnvp_flow, key, dim, context_dim):
        """log_prob JIT-compiles."""
        flow, params = realnvp_flow
        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        log_prob_jit = jax.jit(lambda p, x, c: flow.log_prob(p, x, context=c))
        lp = log_prob_jit(params, x, context)

        assert lp.shape == (10,)

    def test_jit_sample(self, realnvp_flow, key, dim, context_dim):
        """sample JIT-compiles."""
        flow, params = realnvp_flow
        context = jax.random.normal(key, (10, context_dim))

        sample_jit = jax.jit(lambda p, k, c: flow.sample(p, k, (10,), context=c))
        samples = sample_jit(params, key, context)

        assert samples.shape == (10, dim)


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
        x = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

        z, _ = flow.inverse(params, x, context=context)
        x_reconstructed, _ = flow.forward(params, z, context=context)

        error = jnp.abs(x - x_reconstructed).max()
        assert error < 1e-4, f"inverse->forward error: {error}"

    def test_invertibility_forward_inverse(self, spline_flow, key, dim, context_dim):
        """inverse(forward(z, c), c) ≈ z"""
        flow, params = spline_flow
        z = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

        x, _ = flow.forward(params, z, context=context)
        z_reconstructed, _ = flow.inverse(params, x, context=context)

        error = jnp.abs(z - z_reconstructed).max()
        assert error < 1e-4, f"forward->inverse error: {error}"

    def test_log_det_consistency(self, spline_flow, key, dim, context_dim):
        """log_det_forward = -log_det_inverse"""
        flow, params = spline_flow
        z = jax.random.normal(key, (100, dim))
        context = jax.random.normal(key, (100, context_dim))

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
        context = jax.random.normal(key, (100, context_dim))

        samples, log_prob_combined = flow.sample_and_log_prob(params, key, (100,), context=context)
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

    def test_jit_log_prob(self, spline_flow, key, dim, context_dim):
        """log_prob JIT-compiles."""
        flow, params = spline_flow
        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        log_prob_jit = jax.jit(lambda p, x, c: flow.log_prob(p, x, context=c))
        lp = log_prob_jit(params, x, context)

        assert lp.shape == (10,)


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
