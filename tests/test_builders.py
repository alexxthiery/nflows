# tests/test_builders.py
"""Unit tests for flow builder functions."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflows.builders import (
    build_realnvp,
    build_spline_realnvp,
    analyze_mask_coverage,
    _make_alternating_mask,
)
from nflows.flows import Bijection
from nflows.transforms import Permutation, LinearTransform, LoftTransform
from nflows.distributions import DiagNormal, StandardNormal


# ============================================================================
# build_realnvp Tests
# ============================================================================
class TestBuildRealNVP:
    """Tests for build_realnvp builder."""

    def test_basic_build(self, key, dim):
        """Basic flow builds successfully."""
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1
        )

        assert flow is not None
        assert "base" in params
        assert "transform" in params

    def test_use_linear(self, key, dim):
        """use_linear=True adds LinearTransform at start."""
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_linear=True
        )

        # First block should be LinearTransform
        first_block = flow.transform.blocks[0]
        assert isinstance(first_block, LinearTransform)

        # First param set should have linear params
        first_params = params["transform"][0]
        assert "lower" in first_params
        assert "upper" in first_params
        assert "log_diag" in first_params

    def test_use_linear_invertibility(self, key, dim):
        """Flow with use_linear=True is invertible."""
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_linear=True
        )

        x = jax.random.normal(key, (20, dim))
        z, ld_inv = flow.inverse(params, x)
        x_rec, ld_fwd = flow.forward(params, z)

        assert jnp.abs(x - x_rec).max() < 1e-4
        assert jnp.abs(ld_fwd + ld_inv).max() < 1e-4

    def test_use_permutation(self, key):
        """use_permutation=True adds Permutation between couplings."""
        # Use odd dim=5 - reverse permutation breaks mask symmetry for odd dims
        flow, params = build_realnvp(
            key, dim=5, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_permutation=True
        )

        # Check that permutations exist in blocks
        perm_count = sum(
            1 for b in flow.transform.blocks if isinstance(b, Permutation)
        )
        # With 2 coupling layers and permutations between them, expect 1 permutation
        assert perm_count == 1

    def test_use_permutation_invertibility(self, key):
        """Flow with use_permutation=True is invertible."""
        # Use odd dim=5 - reverse permutation breaks mask symmetry for odd dims
        flow, params = build_realnvp(
            key, dim=5, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_permutation=True
        )

        x = jax.random.normal(key, (10, 5))
        z, ld_inv = flow.inverse(params, x)
        x_rec, ld_fwd = flow.forward(params, z)

        assert jnp.abs(x - x_rec).max() < 1e-4
        assert jnp.abs(ld_fwd + ld_inv).max() < 1e-4

    def test_trainable_base(self, key, dim):
        """trainable_base=True uses DiagNormal base."""
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            trainable_base=True
        )

        assert isinstance(flow.base_dist, DiagNormal)
        assert "loc" in params["base"]
        assert "log_scale" in params["base"]
        assert params["base"]["loc"].shape == (dim,)
        assert params["base"]["log_scale"].shape == (dim,)

    def test_trainable_base_invertibility(self, key, dim):
        """Flow with trainable_base=True is invertible."""
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            trainable_base=True
        )

        x = jax.random.normal(key, (20, dim))
        z, ld_inv = flow.inverse(params, x)
        x_rec, ld_fwd = flow.forward(params, z)

        assert jnp.abs(x - x_rec).max() < 1e-4

    def test_custom_base_dist(self, key, dim):
        """Custom base_dist is used when provided."""
        custom_base = DiagNormal(dim=dim)
        custom_params = {
            "loc": jnp.ones(dim),
            "log_scale": jnp.zeros(dim),
        }

        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            base_dist=custom_base,
            base_params=custom_params,
        )

        assert flow.base_dist is custom_base
        assert jnp.allclose(params["base"]["loc"], jnp.ones(dim))

    def test_context_dim(self, key, dim):
        """context_dim creates conditional flow."""
        context_dim = 3
        flow, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            context_dim=context_dim
        )

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        lp = flow.log_prob(params, x, context=context)
        assert lp.shape == (10,)
        assert not jnp.isnan(lp).any()

    def test_all_options_combined(self, key):
        """All options can be combined."""
        # Use odd dim=5 - reverse permutation breaks mask symmetry for odd dims
        flow, params = build_realnvp(
            key, dim=5, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            context_dim=2,
            use_linear=True,
            use_permutation=True,
            trainable_base=True,
        )

        x = jax.random.normal(key, (10, 5))
        context = jax.random.normal(key, (10, 2))

        z, ld_inv = flow.inverse(params, x, context=context)
        x_rec, ld_fwd = flow.forward(params, z, context=context)

        # Relaxed tolerance for combined complexity of all options
        assert jnp.abs(x - x_rec).max() < 1e-3

    def test_invalid_dim_raises(self, key):
        """dim <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            build_realnvp(key, dim=0, num_layers=2, hidden_dim=8, n_hidden_layers=1)

    def test_invalid_num_layers_raises(self, key, dim):
        """num_layers <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            build_realnvp(key, dim=dim, num_layers=0, hidden_dim=8, n_hidden_layers=1)

    def test_invalid_context_dim_raises(self, key, dim):
        """context_dim < 0 raises ValueError."""
        with pytest.raises(ValueError, match="context_dim must be non-negative"):
            build_realnvp(key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1, context_dim=-1)


# ============================================================================
# build_spline_realnvp Tests
# ============================================================================
class TestBuildSplineRealNVP:
    """Tests for build_spline_realnvp builder."""

    def test_basic_build(self, key, dim):
        """Basic spline flow builds successfully."""
        flow, params = build_spline_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1, num_bins=4
        )

        assert flow is not None
        assert "base" in params
        assert "transform" in params

    def test_use_linear(self, key, dim):
        """use_linear=True adds LinearTransform at start."""
        flow, params = build_spline_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_linear=True, num_bins=4
        )

        first_block = flow.transform.blocks[0]
        assert isinstance(first_block, LinearTransform)

    def test_use_permutation(self, key):
        """use_permutation=True adds Permutation between couplings."""
        # Use odd dim=5 - reverse permutation breaks mask symmetry for odd dims
        flow, params = build_spline_realnvp(
            key, dim=5, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_permutation=True, num_bins=4
        )

        perm_count = sum(
            1 for b in flow.transform.blocks if isinstance(b, Permutation)
        )
        assert perm_count == 1

    def test_trainable_base(self, key, dim):
        """trainable_base=True uses DiagNormal base."""
        flow, params = build_spline_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            trainable_base=True, num_bins=4
        )

        assert isinstance(flow.base_dist, DiagNormal)
        assert "loc" in params["base"]

    def test_invertibility_with_options(self, key):
        """Spline flow with all options is invertible."""
        # Use odd dim=5 - reverse permutation breaks mask symmetry for odd dims
        flow, params = build_spline_realnvp(
            key, dim=5, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            use_linear=True,
            use_permutation=True,
            trainable_base=True,
            num_bins=4,
        )

        x = jax.random.normal(key, (10, 5))
        z, ld_inv = flow.inverse(params, x)
        x_rec, ld_fwd = flow.forward(params, z)

        assert jnp.abs(x - x_rec).max() < 1e-3

    def test_invalid_num_bins_raises(self, key, dim):
        """num_bins <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_bins must be positive"):
            build_spline_realnvp(key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1, num_bins=0)

    def test_invalid_min_bin_width_raises(self, key, dim):
        """min_bin_width * num_bins >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_bin_width"):
            build_spline_realnvp(
                key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
                num_bins=4, min_bin_width=0.3  # 0.3 * 4 = 1.2 >= 1
            )


# ============================================================================
# Helper Function Tests
# ============================================================================
class TestMakeAlternatingMask:
    """Tests for _make_alternating_mask helper."""

    def test_parity_0(self):
        """parity=0 produces [1, 0, 1, 0, ...]."""
        mask = _make_alternating_mask(dim=4, parity=0)
        expected = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        assert jnp.allclose(mask, expected)

    def test_parity_1(self):
        """parity=1 produces [0, 1, 0, 1, ...]."""
        mask = _make_alternating_mask(dim=4, parity=1)
        expected = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        assert jnp.allclose(mask, expected)

    def test_odd_dim(self):
        """Works with odd dimensions."""
        mask = _make_alternating_mask(dim=5, parity=0)
        expected = jnp.array([1, 0, 1, 0, 1], dtype=jnp.float32)
        assert jnp.allclose(mask, expected)


class TestAnalyzeMaskCoverage:
    """Tests for analyze_mask_coverage helper."""

    def test_valid_coverage_passes(self, dim):
        """Valid mask schedule passes without error."""
        from nflows.transforms import AffineCoupling
        from nflows.nets import MLP

        mask0 = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mask1 = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=2*dim)

        blocks = [
            AffineCoupling(mask=mask0, conditioner=mlp),
            AffineCoupling(mask=mask1, conditioner=mlp),
        ]

        # Should not raise
        analyze_mask_coverage(blocks, dim=dim)

    def test_incomplete_coverage_raises(self, dim):
        """Incomplete mask coverage raises ValueError."""
        from nflows.transforms import AffineCoupling
        from nflows.nets import MLP

        # Same mask twice - dims 0 and 2 never transformed
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=2*dim)

        blocks = [
            AffineCoupling(mask=mask, conditioner=mlp),
            AffineCoupling(mask=mask, conditioner=mlp),
        ]

        with pytest.raises(ValueError, match="never transformed"):
            analyze_mask_coverage(blocks, dim=dim)


# ============================================================================
# return_transform_only / Bijection Tests
# ============================================================================
class TestReturnTransformOnly:
    """Tests for return_transform_only parameter and Bijection class."""

    def test_realnvp_returns_bijection(self, key, dim):
        """return_transform_only=True returns Bijection instead of Flow."""
        bijection, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            return_transform_only=True,
        )

        assert isinstance(bijection, Bijection)
        assert "transform" in params
        assert "base" not in params

    def test_spline_realnvp_returns_bijection(self, key, dim):
        """return_transform_only=True returns Bijection for spline flow."""
        bijection, params = build_spline_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            num_bins=4, return_transform_only=True,
        )

        assert isinstance(bijection, Bijection)
        assert "transform" in params
        assert "base" not in params

    def test_bijection_forward_inverse(self, key, dim):
        """Bijection forward/inverse work correctly."""
        bijection, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            return_transform_only=True,
        )

        x = jax.random.normal(key, (10, dim))
        y, ld_fwd = bijection.forward(params, x)
        x_rec, ld_inv = bijection.inverse(params, y)

        assert y.shape == x.shape
        assert ld_fwd.shape == (10,)
        assert jnp.abs(x - x_rec).max() < 1e-4
        assert jnp.abs(ld_fwd + ld_inv).max() < 1e-4

    def test_bijection_with_context(self, key, dim, context_dim):
        """Bijection works with context."""
        bijection, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            context_dim=context_dim, return_transform_only=True,
        )

        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))

        y, ld_fwd = bijection.forward(params, x, context=ctx)
        x_rec, ld_inv = bijection.inverse(params, y, context=ctx)

        assert y.shape == x.shape
        assert jnp.abs(x - x_rec).max() < 1e-4

    def test_bijection_with_feature_extractor(self, key, dim, context_dim):
        """Bijection works with context feature extractor."""
        bijection, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            context_dim=context_dim,
            context_extractor_hidden_dim=16,
            context_extractor_n_layers=1,
            return_transform_only=True,
        )

        assert bijection.feature_extractor is not None
        assert "feature_extractor" in params
        assert "transform" in params
        assert "base" not in params

        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))

        y, ld_fwd = bijection.forward(params, x, context=ctx)
        x_rec, ld_inv = bijection.inverse(params, y, context=ctx)

        assert y.shape == x.shape
        assert jnp.abs(x - x_rec).max() < 1e-4

    def test_spline_bijection_invertibility(self, key, dim):
        """Spline Bijection is invertible."""
        bijection, params = build_spline_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            num_bins=4, return_transform_only=True,
        )

        x = jax.random.normal(key, (10, dim))
        y, ld_fwd = bijection.forward(params, x)
        x_rec, ld_inv = bijection.inverse(params, y)

        assert jnp.abs(x - x_rec).max() < 1e-3
        assert jnp.abs(ld_fwd + ld_inv).max() < 1e-3

    def test_bijection_jit_compatible(self, key, dim):
        """Bijection is JIT-compatible."""
        bijection, params = build_realnvp(
            key, dim=dim, num_layers=2, hidden_dim=8, n_hidden_layers=1,
            return_transform_only=True,
        )

        @jax.jit
        def forward_jit(p, x):
            return bijection.forward(p, x)

        x = jax.random.normal(key, (10, dim))
        y, ld = forward_jit(params, x)

        assert y.shape == x.shape
        assert not jnp.isnan(y).any()
        assert not jnp.isnan(ld).any()
