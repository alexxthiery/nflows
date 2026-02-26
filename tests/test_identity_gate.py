# tests/test_identity_gate.py
"""Tests for identity_gate functionality in transforms and flows."""
import pytest
import jax
import jax.numpy as jnp

from nflows.transforms import (
    AffineCoupling,
    SplineCoupling,
    LinearTransform,
    LoftTransform,
    CompositeTransform,
    _compute_gate_value,
)
from nflows.builders import build_realnvp, build_spline_realnvp, make_alternating_mask
from nflows.distributions import StandardNormal


# ============================================================================
# _compute_gate_value tests
# ============================================================================
class TestComputeGateValue:
    """Tests for _compute_gate_value utility function."""

    def test_none_identity_gate_returns_none(self):
        """Returns None when identity_gate is None."""
        context = jnp.array([1.0, 2.0])
        result = _compute_gate_value(None, context)
        assert result is None

    def test_none_context_returns_none(self):
        """Returns None when context is None."""
        gate_fn = lambda c: c[0]
        result = _compute_gate_value(gate_fn, None)
        assert result is None

    def test_single_sample_scalar_output(self):
        """Gate function on single sample returns scalar."""
        gate_fn = lambda c: c[0]
        context = jnp.array([0.5, 1.0])
        result = _compute_gate_value(gate_fn, context)
        assert result.shape == ()
        assert jnp.isclose(result, 0.5)

    def test_batch_returns_correct_shape(self):
        """Gate function on batch returns (batch,) shape."""
        gate_fn = lambda c: jnp.sum(c)
        context = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _compute_gate_value(gate_fn, context)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([3.0, 7.0, 11.0]))

    def test_sin_pi_gate(self):
        """Test sin(pi * t) gate function gives correct values."""
        gate_fn = lambda c: jnp.sin(jnp.pi * c[0])
        # At t=0: gate=0, at t=0.5: gate=1, at t=1: gate=0
        ctx_0 = jnp.array([0.0])
        ctx_half = jnp.array([0.5])
        ctx_1 = jnp.array([1.0])

        assert jnp.isclose(_compute_gate_value(gate_fn, ctx_0), 0.0, atol=1e-6)
        assert jnp.isclose(_compute_gate_value(gate_fn, ctx_half), 1.0, atol=1e-6)
        assert jnp.isclose(_compute_gate_value(gate_fn, ctx_1), 0.0, atol=1e-6)


# ============================================================================
# AffineCoupling gate tests
# ============================================================================
class TestAffineCouplingGate:
    """Tests for AffineCoupling with identity gate."""

    @pytest.fixture
    def coupling_and_params(self):
        """Create AffineCoupling with non-trivial params."""
        key = jax.random.PRNGKey(42)
        dim = 4
        mask = make_alternating_mask(dim, parity=0)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=32, n_hidden_layers=1, context_dim=1
        )
        # Perturb params to make transform non-trivial (not identity at init)
        # Add random noise to all param leaves
        key2 = jax.random.PRNGKey(123)
        params = jax.tree_util.tree_map(
            lambda p, k: p + 0.5 * jax.random.normal(k, p.shape),
            params,
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(params),
                jax.random.split(key2, len(jax.tree_util.tree_leaves(params)))
            )
        )
        return coupling, params, dim

    def test_gate_zero_gives_identity(self, coupling_and_params):
        """When g_value=0, forward returns identity transform."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        context = jnp.zeros((10, 1))
        g_value = jnp.zeros(10)

        y, log_det = coupling.forward(params, x, context, g_value=g_value)

        assert jnp.allclose(y, x, atol=1e-6), "y should equal x when g=0"
        assert jnp.allclose(log_det, 0.0, atol=1e-6), "log_det should be 0 when g=0"

    def test_gate_one_gives_normal_transform(self, coupling_and_params):
        """When g_value=1, forward returns normal (ungated) transform."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        context = jnp.ones((10, 1)) * 0.5

        # Ungated transform
        y_ungated, log_det_ungated = coupling.forward(params, x, context)
        # Gated with g=1
        g_value = jnp.ones(10)
        y_gated, log_det_gated = coupling.forward(params, x, context, g_value=g_value)

        assert jnp.allclose(y_gated, y_ungated, atol=1e-5)
        assert jnp.allclose(log_det_gated, log_det_ungated, atol=1e-5)

    def test_gate_invertibility(self, coupling_and_params):
        """forward(inverse(y)) = y with gating."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        context = jnp.ones((10, 1)) * 0.3
        g_value = jnp.ones(10) * 0.7  # Partial gate

        y, _ = coupling.forward(params, x, context, g_value=g_value)
        x_rec, _ = coupling.inverse(params, y, context, g_value=g_value)

        assert jnp.allclose(x_rec, x, atol=1e-5)

    def test_gate_gradient_no_nan(self, coupling_and_params):
        """Gradients w.r.t. gated output don't produce NaN."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (5, dim))
        context = jnp.ones((5, 1)) * 0.5
        g_value = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

        def loss_fn(p):
            y, log_det = coupling.forward(p, x, context, g_value=g_value)
            return jnp.sum(y**2) + jnp.sum(log_det)

        grads = jax.grad(loss_fn)(params)
        # Check no NaN in gradients
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert not jnp.any(jnp.isnan(g)), "Gradient contains NaN"
            assert not jnp.any(jnp.isinf(g)), "Gradient contains Inf"


# ============================================================================
# SplineCoupling gate tests
# ============================================================================
class TestSplineCouplingGate:
    """Tests for SplineCoupling with identity gate."""

    @pytest.fixture
    def coupling_and_params(self):
        """Create SplineCoupling with non-trivial params."""
        key = jax.random.PRNGKey(42)
        dim = 4
        mask = make_alternating_mask(dim, parity=0)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, num_bins=8
        )
        # Perturb params to make transform non-trivial (not identity at init)
        # Use smaller perturbation for splines to maintain numerical stability
        key2 = jax.random.PRNGKey(456)
        params = jax.tree_util.tree_map(
            lambda p, k: p + 0.1 * jax.random.normal(k, p.shape),
            params,
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(params),
                jax.random.split(key2, len(jax.tree_util.tree_leaves(params)))
            )
        )
        return coupling, params, dim

    def test_gate_zero_gives_identity(self, coupling_and_params):
        """When g_value=0, forward returns identity transform."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        # Clip to be inside spline domain
        x = jnp.clip(x, -4.0, 4.0)
        context = jnp.zeros((10, 1))
        g_value = jnp.zeros(10)

        y, log_det = coupling.forward(params, x, context, g_value=g_value)

        assert jnp.allclose(y, x, atol=1e-5), f"y should equal x when g=0, max diff: {jnp.max(jnp.abs(y - x))}"
        assert jnp.allclose(log_det, 0.0, atol=1e-5), f"log_det should be 0 when g=0, got: {log_det}"

    def test_gate_one_gives_normal_transform(self, coupling_and_params):
        """When g_value=1, forward returns normal (ungated) transform."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        x = jnp.clip(x, -4.0, 4.0)
        context = jnp.ones((10, 1)) * 0.5

        # Ungated transform
        y_ungated, log_det_ungated = coupling.forward(params, x, context)
        # Gated with g=1
        g_value = jnp.ones(10)
        y_gated, log_det_gated = coupling.forward(params, x, context, g_value=g_value)

        assert jnp.allclose(y_gated, y_ungated, atol=1e-5)
        assert jnp.allclose(log_det_gated, log_det_ungated, atol=1e-5)

    def test_gate_invertibility(self, coupling_and_params):
        """forward(inverse(y)) = y with gating."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        x = jnp.clip(x, -4.0, 4.0)
        context = jnp.ones((10, 1)) * 0.3
        g_value = jnp.ones(10) * 0.7

        y, _ = coupling.forward(params, x, context, g_value=g_value)
        x_rec, _ = coupling.inverse(params, y, context, g_value=g_value)

        assert jnp.allclose(x_rec, x, atol=1e-4)

    def test_gate_gradient_no_nan(self, coupling_and_params):
        """Gradients w.r.t. gated output don't produce NaN."""
        coupling, params, dim = coupling_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (5, dim))
        x = jnp.clip(x, -4.0, 4.0)
        context = jnp.ones((5, 1)) * 0.5
        g_value = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

        def loss_fn(p):
            y, log_det = coupling.forward(p, x, context, g_value=g_value)
            return jnp.sum(y**2) + jnp.sum(log_det)

        grads = jax.grad(loss_fn)(params)
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert not jnp.any(jnp.isnan(g)), "Gradient contains NaN"
            assert not jnp.any(jnp.isinf(g)), "Gradient contains Inf"


# ============================================================================
# LinearTransform gate tests
# ============================================================================
class TestLinearTransformGate:
    """Tests for LinearTransform with identity gate."""

    @pytest.fixture
    def transform_and_params(self):
        """Create LinearTransform with non-identity params."""
        key = jax.random.PRNGKey(42)
        dim = 4
        transform, _ = LinearTransform.create(key, dim=dim)
        # Create non-trivial params
        params = {
            "lower": jax.random.normal(key, (dim, dim)) * 0.1,
            "upper": jax.random.normal(jax.random.PRNGKey(1), (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(jax.random.PRNGKey(2), (dim,)) * 0.1,
        }
        return transform, params, dim

    def test_gate_zero_gives_identity(self, transform_and_params):
        """When g_value=0, forward returns identity transform."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        g_value = jnp.zeros(10)

        y, log_det = transform.forward(params, x, g_value=g_value)

        assert jnp.allclose(y, x, atol=1e-6), "y should equal x when g=0"
        assert jnp.allclose(log_det, 0.0, atol=1e-6), "log_det should be 0 when g=0"

    def test_gate_one_gives_normal_transform(self, transform_and_params):
        """When g_value=1, forward returns normal (ungated) transform."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))

        y_ungated, log_det_ungated = transform.forward(params, x)
        g_value = jnp.ones(10)
        y_gated, log_det_gated = transform.forward(params, x, g_value=g_value)

        assert jnp.allclose(y_gated, y_ungated, atol=1e-5)
        assert jnp.allclose(log_det_gated, log_det_ungated, atol=1e-5)

    def test_gate_invertibility(self, transform_and_params):
        """forward(inverse(y)) = y with gating."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        g_value = jnp.ones(10) * 0.7

        y, _ = transform.forward(params, x, g_value=g_value)
        x_rec, _ = transform.inverse(params, y, g_value=g_value)

        assert jnp.allclose(x_rec, x, atol=1e-5)

    def test_varying_gate_values_per_sample(self, transform_and_params):
        """Each sample should use its own gate value, not the batch mean.

        This test catches the bug where L, U matrices were scaled by mean(g_value)
        instead of per-sample g_value.
        """
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (5, dim))
        # Varying gate values: 0, 0.25, 0.5, 0.75, 1.0
        g_values = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Compute batched result
        y_batch, log_det_batch = transform.forward(params, x, g_value=g_values)

        # Compute per-sample results individually
        for i in range(5):
            x_i = x[i : i + 1]
            g_i = jnp.array([g_values[i]])
            y_i, log_det_i = transform.forward(params, x_i, g_value=g_i)

            assert jnp.allclose(y_batch[i], y_i[0], atol=1e-6), (
                f"Sample {i} with g={g_values[i]}: batched result differs from individual"
            )
            assert jnp.allclose(log_det_batch[i], log_det_i[0], atol=1e-6), (
                f"Sample {i} with g={g_values[i]}: batched log_det differs from individual"
            )


# ============================================================================
# Builder tests
# ============================================================================
class TestBuildRealNVPIdentityGate:
    """Tests for build_realnvp with identity_gate."""

    def test_identity_gate_at_zero(self):
        """Flow with identity_gate gives identity at gate=0."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (100, 4))
        ctx_0 = jnp.zeros((100, 1))  # gate = sin(0) = 0

        y, log_det = flow.forward(params, x, context=ctx_0)

        assert jnp.allclose(y, x, atol=1e-5), f"y should equal x when gate=0, max diff: {jnp.max(jnp.abs(y - x))}"
        assert jnp.allclose(log_det, 0.0, atol=1e-5), f"log_det should be 0 when gate=0"

    def test_identity_gate_at_one(self):
        """Flow with identity_gate gives identity at gate=0 for ctx=1."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (100, 4))
        ctx_1 = jnp.ones((100, 1))  # gate = sin(pi) = 0

        y, log_det = flow.forward(params, x, context=ctx_1)

        assert jnp.allclose(y, x, atol=1e-5), "y should equal x when gate=0"
        assert jnp.allclose(log_det, 0.0, atol=1e-5), "log_det should be 0 when gate=0"

    def test_identity_gate_invertibility(self):
        """Flow with identity_gate is invertible."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]  # Simple linear gate

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
        ctx = jnp.ones((50, 1)) * 0.7  # gate = 0.7

        y, _ = flow.forward(params, x, context=ctx)
        x_rec, _ = flow.inverse(params, y, context=ctx)

        assert jnp.allclose(x_rec, x, atol=1e-4)

    def test_identity_gate_permutation_raises(self):
        """identity_gate + use_permutation raises ValueError."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        with pytest.raises(ValueError, match="incompatible with use_permutation"):
            build_realnvp(
                key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
                context_dim=1, identity_gate=gate_fn, use_permutation=True
            )

    def test_identity_gate_requires_context(self):
        """identity_gate without context_dim raises ValueError."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        with pytest.raises(ValueError, match="requires context_dim > 0"):
            build_realnvp(
                key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
                context_dim=0, identity_gate=gate_fn
            )


class TestBuildSplineRealNVPIdentityGate:
    """Tests for build_spline_realnvp with identity_gate."""

    def test_identity_gate_at_zero(self):
        """Spline flow with identity_gate gives identity at gate=0."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

        flow, params = build_spline_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False, num_bins=8
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (100, 4))
        x = jnp.clip(x, -4.0, 4.0)  # Stay inside spline domain
        ctx_0 = jnp.zeros((100, 1))

        y, log_det = flow.forward(params, x, context=ctx_0)

        assert jnp.allclose(y, x, atol=1e-4), f"y should equal x when gate=0, max diff: {jnp.max(jnp.abs(y - x))}"
        assert jnp.allclose(log_det, 0.0, atol=1e-4), f"log_det should be 0 when gate=0"

    def test_identity_gate_invertibility(self):
        """Spline flow with identity_gate is invertible."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        flow, params = build_spline_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False, num_bins=8
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
        x = jnp.clip(x, -4.0, 4.0)
        ctx = jnp.ones((50, 1)) * 0.7

        y, _ = flow.forward(params, x, context=ctx)
        x_rec, _ = flow.inverse(params, y, context=ctx)

        assert jnp.allclose(x_rec, x, atol=1e-4)

    def test_identity_gate_permutation_raises(self):
        """identity_gate + use_permutation raises ValueError."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        with pytest.raises(ValueError, match="incompatible with use_permutation"):
            build_spline_realnvp(
                key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
                context_dim=1, identity_gate=gate_fn, use_permutation=True
            )


class TestBijectionIdentityGate:
    """Tests for Bijection with identity_gate."""

    def test_bijection_identity_gate(self):
        """Bijection with identity_gate works correctly."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        bijection, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False,
            return_transform_only=True
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50, 4))

        # Test at gate=0
        ctx_0 = jnp.zeros((50, 1))
        y_0, log_det_0 = bijection.forward(params, x, context=ctx_0)
        assert jnp.allclose(y_0, x, atol=1e-5)
        assert jnp.allclose(log_det_0, 0.0, atol=1e-5)

        # Test invertibility at intermediate gate
        ctx_half = jnp.ones((50, 1)) * 0.5
        y, _ = bijection.forward(params, x, context=ctx_half)
        x_rec, _ = bijection.inverse(params, y, context=ctx_half)
        assert jnp.allclose(x_rec, x, atol=1e-4)


# ============================================================================
# LoftTransform gate tests (C2)
# ============================================================================
class TestLoftTransformGate:
    """Tests for LoftTransform with identity gate (g_value parameter)."""

    @pytest.fixture
    def transform_and_params(self):
        """Create LoftTransform with non-trivial tau."""
        key = jax.random.PRNGKey(42)
        dim = 4
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)
        return transform, params, dim

    def test_gate_zero_gives_identity(self, transform_and_params):
        """When g_value=0, LoftTransform forward returns identity."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        g_value = jnp.zeros(10)

        y, log_det = transform.forward(params, x, g_value=g_value)

        assert jnp.allclose(y, x, atol=1e-6), (
            f"y should equal x when g=0, max diff: {jnp.max(jnp.abs(y - x))}"
        )
        assert jnp.allclose(log_det, 0.0, atol=1e-6), (
            f"log_det should be 0 when g=0, got: {log_det}"
        )

    def test_gate_one_gives_normal_transform(self, transform_and_params):
        """When g_value=1, LoftTransform forward matches ungated."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))

        y_ungated, ld_ungated = transform.forward(params, x)
        g_value = jnp.ones(10)
        y_gated, ld_gated = transform.forward(params, x, g_value=g_value)

        assert jnp.allclose(y_gated, y_ungated, atol=1e-5)
        assert jnp.allclose(ld_gated, ld_ungated, atol=1e-5)

    def test_gate_invertibility(self, transform_and_params):
        """forward(inverse(y)) roundtrips with intermediate gate value."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim))
        g_value = jnp.ones(10) * 0.7

        y, _ = transform.forward(params, x, g_value=g_value)
        x_rec, _ = transform.inverse(params, y, g_value=g_value)

        assert jnp.allclose(x_rec, x, atol=1e-4), (
            f"roundtrip failed, max err: {jnp.max(jnp.abs(x_rec - x))}"
        )

    def test_gate_interpolation(self, transform_and_params):
        """g=0.5 should produce output between identity and full LOFT."""
        transform, params, dim = transform_and_params
        x = jax.random.normal(jax.random.PRNGKey(0), (10, dim)) * 3.0
        g_value = jnp.ones(10) * 0.5

        y_gated, _ = transform.forward(params, x, g_value=g_value)
        y_full, _ = transform.forward(params, x)

        # y_gated should be between x and y_full (element-wise)
        diff_gated = jnp.abs(y_gated - x)
        diff_full = jnp.abs(y_full - x)
        assert jnp.all(diff_gated <= diff_full + 1e-5), (
            "g=0.5 output should be closer to identity than g=1"
        )


class TestBuildRealNVPIdentityGateWithLoft:
    """Tests for build_realnvp with identity_gate AND use_loft=True (C2)."""

    def test_identity_gate_loft_at_zero(self):
        """Flow with identity_gate + use_loft=True gives identity at gate=0.

        Uses loft_tau=0.1 so LOFT is strongly non-linear for normal inputs,
        exposing the bug where LOFT ignores g_value.
        """
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=True, loft_tau=0.1
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
        ctx_0 = jnp.zeros((50, 1))  # gate = sin(0) = 0

        y, log_det = flow.forward(params, x, context=ctx_0)

        assert jnp.allclose(y, x, atol=1e-5), (
            f"y should equal x when gate=0 with LOFT, max diff: {jnp.max(jnp.abs(y - x))}"
        )
        assert jnp.allclose(log_det, 0.0, atol=1e-5), (
            f"log_det should be 0 when gate=0 with LOFT, got max: {jnp.max(jnp.abs(log_det))}"
        )

    def test_identity_gate_loft_invertibility(self):
        """Flow with identity_gate + use_loft=True is invertible at gate=0.7."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=True, loft_tau=0.1
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
        ctx = jnp.ones((50, 1)) * 0.7

        y, _ = flow.forward(params, x, context=ctx)
        x_rec, _ = flow.inverse(params, y, context=ctx)

        assert jnp.allclose(x_rec, x, atol=1e-4), (
            f"roundtrip failed, max err: {jnp.max(jnp.abs(x_rec - x))}"
        )


class TestSampleAndLogProbWithGate:
    """Tests for sample_and_log_prob with identity_gate."""

    def test_sample_and_log_prob_consistency(self):
        """sample_and_log_prob is consistent with log_prob at various gate values."""
        key = jax.random.PRNGKey(42)
        gate_fn = lambda ctx: ctx[0]

        flow, params = build_realnvp(
            key, dim=4, num_layers=4, hidden_dim=32, n_hidden_layers=1,
            context_dim=1, identity_gate=gate_fn, use_loft=False
        )

        for gate_val in [0.0, 0.3, 0.7, 1.0]:
            ctx = jnp.ones((20, 1)) * gate_val

            samples, log_prob_forward = flow.sample_and_log_prob(
                params, jax.random.PRNGKey(123), (20,), context=ctx
            )
            log_prob_check = flow.log_prob(params, samples, context=ctx)

            assert jnp.allclose(log_prob_forward, log_prob_check, atol=1e-4), \
                f"Mismatch at gate={gate_val}"
