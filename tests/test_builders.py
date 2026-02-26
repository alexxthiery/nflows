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
    make_alternating_mask,
    create_feature_extractor,
    assemble_bijection,
    assemble_flow,
)
from nflows.flows import Bijection, Flow
from nflows.transforms import (
    AffineCoupling,
    SplineCoupling,
    Permutation,
    LinearTransform,
    LoftTransform,
)
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
        assert "raw_diag" in first_params

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
    """Tests for make_alternating_mask helper."""

    def test_parity_0(self):
        """parity=0 produces [1, 0, 1, 0, ...]."""
        mask = make_alternating_mask(dim=4, parity=0)
        expected = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        assert jnp.allclose(mask, expected)

    def test_parity_1(self):
        """parity=1 produces [0, 1, 0, 1, ...]."""
        mask = make_alternating_mask(dim=4, parity=1)
        expected = jnp.array([0, 1, 0, 1], dtype=jnp.float32)
        assert jnp.allclose(mask, expected)

    def test_odd_dim(self):
        """Works with odd dimensions."""
        mask = make_alternating_mask(dim=5, parity=0)
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


# ============================================================================
# make_alternating_mask Validation Tests
# ============================================================================
class TestMakeAlternatingMaskValidation:
    """Additional validation tests for make_alternating_mask."""

    def test_invalid_dim_raises(self):
        """dim <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            make_alternating_mask(dim=0, parity=0)
        with pytest.raises(ValueError, match="dim must be positive"):
            make_alternating_mask(dim=-1, parity=0)

    def test_invalid_parity_raises(self):
        """parity not in {0, 1} raises ValueError."""
        with pytest.raises(ValueError, match="parity must be 0 or 1"):
            make_alternating_mask(dim=4, parity=2)
        with pytest.raises(ValueError, match="parity must be 0 or 1"):
            make_alternating_mask(dim=4, parity=-1)


# ============================================================================
# create_feature_extractor Tests
# ============================================================================
class TestCreateFeatureExtractor:
    """Tests for create_feature_extractor helper."""

    def test_basic_creation(self, key):
        """Basic feature extractor creation works."""
        fe, params = create_feature_extractor(
            key, in_dim=8, hidden_dim=32, out_dim=16
        )

        assert fe is not None
        assert params is not None

    def test_output_shape(self, key):
        """Feature extractor produces correct output shape."""
        fe, params = create_feature_extractor(
            key, in_dim=8, hidden_dim=32, out_dim=16
        )

        x = jax.random.normal(key, (10, 8))
        y = fe.apply({"params": params}, x)

        assert y.shape == (10, 16)

    def test_jit_compatible(self, key):
        """Feature extractor is JIT-compatible."""
        fe, params = create_feature_extractor(
            key, in_dim=8, hidden_dim=32, out_dim=16
        )

        @jax.jit
        def apply_fe(p, x):
            return fe.apply({"params": p}, x)

        x = jax.random.normal(key, (10, 8))
        y = apply_fe(params, x)

        assert y.shape == (10, 16)
        assert not jnp.isnan(y).any()

    def test_invalid_in_dim_raises(self, key):
        """in_dim <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="in_dim must be positive"):
            create_feature_extractor(key, in_dim=0, hidden_dim=32, out_dim=16)

    def test_invalid_hidden_dim_raises(self, key):
        """hidden_dim <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_feature_extractor(key, in_dim=8, hidden_dim=0, out_dim=16)

    def test_invalid_out_dim_raises(self, key):
        """out_dim <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="out_dim must be positive"):
            create_feature_extractor(key, in_dim=8, hidden_dim=32, out_dim=0)

    def test_invalid_n_layers_raises(self, key):
        """n_layers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            create_feature_extractor(key, in_dim=8, hidden_dim=32, out_dim=16, n_layers=0)


# ============================================================================
# assemble_bijection Tests
# ============================================================================
class TestAssembleBijection:
    """Tests for assemble_bijection function."""

    def test_basic_assembly(self, key, dim):
        """Basic bijection assembly works."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
            LoftTransform.create(keys[2], dim=dim),
        ]

        bijection, params = assemble_bijection(blocks_and_params)

        assert isinstance(bijection, Bijection)
        assert "transform" in params
        assert len(params["transform"]) == 3

    def test_mixed_coupling_types(self, key, dim):
        """Assembly works with mixed coupling types."""
        keys = jax.random.split(key, 4)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
            SplineCoupling.create(keys[2], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, num_bins=4),
            LinearTransform.create(keys[3], dim=dim),
        ]

        bijection, params = assemble_bijection(blocks_and_params)

        assert isinstance(bijection, Bijection)
        assert len(params["transform"]) == 4

    def test_invertibility(self, key, dim):
        """Assembled bijection is invertible."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
            LoftTransform.create(keys[2], dim=dim),
        ]

        bijection, params = assemble_bijection(blocks_and_params)

        x = jax.random.normal(key, (10, dim))
        y, ld_fwd = bijection.forward(params, x)
        x_rec, ld_inv = bijection.inverse(params, y)

        assert jnp.abs(x - x_rec).max() < 1e-4
        assert jnp.abs(ld_fwd + ld_inv).max() < 1e-4

    def test_with_context(self, key, dim, context_dim):
        """Assembly works with context."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
            LoftTransform.create(keys[2], dim=dim),
        ]

        bijection, params = assemble_bijection(blocks_and_params)

        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))

        y, ld_fwd = bijection.forward(params, x, context=ctx)
        x_rec, ld_inv = bijection.inverse(params, y, context=ctx)

        assert y.shape == x.shape
        assert jnp.abs(x - x_rec).max() < 1e-4

    def test_with_feature_extractor(self, key, dim):
        """Assembly works with feature extractor."""
        keys = jax.random.split(key, 4)
        raw_context_dim = 8
        effective_context_dim = 4

        # Create feature extractor
        fe, fe_params = create_feature_extractor(
            keys[0], in_dim=raw_context_dim, hidden_dim=16, out_dim=effective_context_dim
        )

        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        # Couplings use effective_context_dim
        blocks_and_params = [
            AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=effective_context_dim),
            AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=effective_context_dim),
            LoftTransform.create(keys[3], dim=dim),
        ]

        bijection, params = assemble_bijection(
            blocks_and_params,
            feature_extractor=fe,
            feature_extractor_params=fe_params,
        )

        assert bijection.feature_extractor is not None
        assert "feature_extractor" in params
        assert "transform" in params

        # Test forward/inverse with raw context
        x = jax.random.normal(key, (10, dim))
        raw_ctx = jax.random.normal(key, (10, raw_context_dim))

        y, ld_fwd = bijection.forward(params, x, context=raw_ctx)
        x_rec, ld_inv = bijection.inverse(params, y, context=raw_ctx)

        assert y.shape == x.shape
        assert jnp.abs(x - x_rec).max() < 1e-4

    def test_empty_list_raises(self):
        """Empty blocks_and_params raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            assemble_bijection([])

    def test_non_tuple_raises(self, key, dim):
        """Non-tuple elements raise ValueError."""
        mask0 = make_alternating_mask(dim, parity=0)
        coupling, params = AffineCoupling.create(key, dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1)

        with pytest.raises(ValueError, match="must be a .* tuple"):
            assemble_bijection([coupling])  # Just the coupling, not a tuple

    def test_fe_without_params_raises(self, key, dim):
        """feature_extractor without params raises ValueError."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        fe, fe_params = create_feature_extractor(keys[0], in_dim=4, hidden_dim=8, out_dim=2)

        blocks_and_params = [
            AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=2),
            AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=2),
        ]

        with pytest.raises(ValueError, match="feature_extractor_params is required"):
            assemble_bijection(blocks_and_params, feature_extractor=fe)

    def test_fe_params_without_fe_raises(self, key, dim):
        """feature_extractor_params without feature_extractor raises ValueError."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        fe, fe_params = create_feature_extractor(keys[0], in_dim=4, hidden_dim=8, out_dim=2)

        blocks_and_params = [
            AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
        ]

        with pytest.raises(ValueError, match="feature_extractor is None"):
            assemble_bijection(blocks_and_params, feature_extractor_params=fe_params)

    def test_validate_false_skips_checks(self, key, dim):
        """validate=False skips validation."""
        keys = jax.random.split(key, 2)
        # Use same mask twice - would fail mask coverage check
        mask = make_alternating_mask(dim, parity=0)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask, hidden_dim=8, n_hidden_layers=1),
        ]

        # Should fail with validate=True
        with pytest.raises(ValueError, match="never transformed"):
            assemble_bijection(blocks_and_params, validate=True)

        # Should succeed with validate=False
        bijection, params = assemble_bijection(blocks_and_params, validate=False)
        assert isinstance(bijection, Bijection)

    def test_only_non_coupling_blocks(self, key, dim):
        """Assembly works with only non-coupling blocks."""
        keys = jax.random.split(key, 2)

        blocks_and_params = [
            LinearTransform.create(keys[0], dim=dim),
            LoftTransform.create(keys[1], dim=dim),
        ]

        # Should work - no mask coverage check needed
        bijection, params = assemble_bijection(blocks_and_params)
        assert isinstance(bijection, Bijection)

    def test_identity_gate(self, key, dim):
        """assemble_bijection passes identity_gate to Bijection."""
        context_dim = 2
        gate_fn = lambda ctx: ctx[0] * (1 - ctx[0])

        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
        ]

        bijection, params = assemble_bijection(blocks_and_params, identity_gate=gate_fn)
        assert bijection.identity_gate is gate_fn

        # gate=0 at ctx[0]=0 -> identity
        x = jax.random.normal(keys[2], (5, dim))
        ctx = jnp.zeros((5, context_dim))
        y, ld = bijection.forward(params, x, context=ctx)
        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0, atol=1e-6)


# ============================================================================
# assemble_flow Tests
# ============================================================================
class TestAssembleFlow:
    """Tests for assemble_flow function."""

    def test_basic_assembly(self, key, dim):
        """Basic flow assembly works."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
            LoftTransform.create(keys[2], dim=dim),
        ]

        flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=dim))

        assert isinstance(flow, Flow)
        assert "base" in params
        assert "transform" in params

    def test_log_prob_and_sample(self, key, dim):
        """Assembled flow has working log_prob and sample."""
        keys = jax.random.split(key, 4)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
            LoftTransform.create(keys[2], dim=dim),
        ]

        flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=dim))

        # Sample
        samples = flow.sample(params, keys[3], shape=(100,))
        assert samples.shape == (100, dim)
        assert not jnp.isnan(samples).any()

        # Log prob
        log_prob = flow.log_prob(params, samples)
        assert log_prob.shape == (100,)
        assert not jnp.isnan(log_prob).any()

    def test_with_diag_normal_base(self, key, dim):
        """Assembly works with DiagNormal (trainable) base."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
        ]

        flow, params = assemble_flow(blocks_and_params, base=DiagNormal(dim=dim))

        assert "loc" in params["base"]
        assert "log_scale" in params["base"]

    def test_with_custom_base_params(self, key, dim):
        """Assembly works with custom base_params."""
        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
        ]

        custom_base_params = {
            "loc": jnp.ones(dim),
            "log_scale": jnp.zeros(dim),
        }

        flow, params = assemble_flow(
            blocks_and_params,
            base=DiagNormal(dim=dim),
            base_params=custom_base_params,
        )

        assert jnp.allclose(params["base"]["loc"], jnp.ones(dim))

    def test_base_none_raises(self, key, dim):
        """base=None raises ValueError."""
        keys = jax.random.split(key, 2)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1),
        ]

        with pytest.raises(ValueError, match="base distribution is required"):
            assemble_flow(blocks_and_params, base=None)

    def test_with_feature_extractor(self, key, dim):
        """Assembly works with feature extractor."""
        keys = jax.random.split(key, 4)
        raw_context_dim = 8
        effective_context_dim = 4

        fe, fe_params = create_feature_extractor(
            keys[0], in_dim=raw_context_dim, hidden_dim=16, out_dim=effective_context_dim
        )

        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=effective_context_dim),
            AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=effective_context_dim),
        ]

        flow, params = assemble_flow(
            blocks_and_params,
            base=StandardNormal(dim=dim),
            feature_extractor=fe,
            feature_extractor_params=fe_params,
        )

        assert "feature_extractor" in params
        assert flow.feature_extractor is not None

        # Test with context
        x = jax.random.normal(key, (10, dim))
        raw_ctx = jax.random.normal(key, (10, raw_context_dim))

        log_prob = flow.log_prob(params, x, context=raw_ctx)
        assert log_prob.shape == (10,)
        assert not jnp.isnan(log_prob).any()

    def test_identity_gate(self, key, dim):
        """assemble_flow passes identity_gate to Flow."""
        context_dim = 2
        gate_fn = lambda ctx: ctx[0] * (1 - ctx[0])

        keys = jax.random.split(key, 3)
        mask0 = make_alternating_mask(dim, parity=0)
        mask1 = make_alternating_mask(dim, parity=1)

        blocks_and_params = [
            AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
            AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=8, n_hidden_layers=1, context_dim=context_dim),
        ]

        flow, params = assemble_flow(
            blocks_and_params, base=StandardNormal(dim=dim), identity_gate=gate_fn,
        )
        assert flow.identity_gate is gate_fn

        # gate=0 at ctx[0]=0 -> log_prob equals base log_prob (no transform)
        x = jax.random.normal(keys[2], (5, dim))
        ctx = jnp.zeros((5, context_dim))
        lp = flow.log_prob(params, x, context=ctx)
        base_lp = flow.base_dist.log_prob(params["base"], x)
        assert jnp.allclose(lp, base_lp, atol=1e-6)


# ============================================================================
# Default Consistency Tests
# ============================================================================
class TestDefaultConsistency:
    """Verify builder defaults match .create() / dataclass defaults.

    These tests exist to prevent silent drift between the two APIs.
    If a default changes in one place, the test fails until both are aligned.
    """

    def _get_default(self, func, param_name):
        """Extract the default value of a keyword argument from a callable."""
        import inspect
        sig = inspect.signature(func)
        p = sig.parameters[param_name]
        assert p.default is not inspect.Parameter.empty, (
            f"{func.__qualname__} has no default for '{param_name}'"
        )
        return p.default

    def test_max_log_scale_realnvp_vs_affine_create(self):
        builder_val = self._get_default(build_realnvp, "max_log_scale")
        create_val = self._get_default(AffineCoupling.create, "max_log_scale")
        assert builder_val == create_val, (
            f"max_log_scale mismatch: build_realnvp={builder_val}, "
            f"AffineCoupling.create={create_val}"
        )

    def test_max_log_scale_affine_dataclass_vs_create(self):
        from dataclasses import fields
        dc_val = {f.name: f.default for f in fields(AffineCoupling)}["max_log_scale"]
        create_val = self._get_default(AffineCoupling.create, "max_log_scale")
        assert dc_val == create_val, (
            f"max_log_scale mismatch: AffineCoupling dataclass={dc_val}, "
            f"AffineCoupling.create={create_val}"
        )

    @pytest.mark.parametrize("param", [
        "min_bin_width", "min_bin_height", "min_derivative",
    ])
    def test_spline_builder_vs_create(self, param):
        builder_val = self._get_default(build_spline_realnvp, param)
        create_val = self._get_default(SplineCoupling.create, param)
        assert builder_val == create_val, (
            f"{param} mismatch: build_spline_realnvp={builder_val}, "
            f"SplineCoupling.create={create_val}"
        )

    @pytest.mark.parametrize("param", [
        "min_bin_width", "min_bin_height", "min_derivative", "max_derivative",
    ])
    def test_spline_dataclass_vs_create(self, param):
        from dataclasses import fields
        dc_defaults = {f.name: f.default for f in fields(SplineCoupling)}
        dc_val = dc_defaults[param]
        create_val = self._get_default(SplineCoupling.create, param)
        assert dc_val == create_val, (
            f"{param} mismatch: SplineCoupling dataclass={dc_val}, "
            f"SplineCoupling.create={create_val}"
        )
