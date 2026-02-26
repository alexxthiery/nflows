# tests/test_transforms.py
"""Unit tests for transform layers."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflows.transforms import (
    LinearTransform,
    Permutation,
    AffineCoupling,
    SplineCoupling,
    CompositeTransform,
    LoftTransform,
)
from nflows.nets import init_mlp
from conftest import check_logdet_vs_autodiff


# ============================================================================
# LinearTransform Tests
# ============================================================================
class TestLinearTransform:
    """Tests for LinearTransform (LU-parameterized)."""

    @pytest.fixture
    def identity_params(self, dim):
        """Identity transform params (L=I, U=0, s=1).

        For softplus parametrization, raw_diag = log(e-1) gives s = softplus(raw_diag) = 1.
        """
        return {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }

    @pytest.fixture
    def random_params(self, key, dim):
        """Random transform params."""
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "lower": jax.random.normal(k1, (dim, dim)) * 0.1,
            "upper": jax.random.normal(k2, (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(k3, (dim,)) * 0.5,
        }

    def test_identity_at_init(self, dim, identity_params):
        """With zero params, transform is identity."""
        transform = LinearTransform(dim=dim)
        x = jnp.arange(dim, dtype=jnp.float32).reshape(1, dim)

        y, ld = transform.forward(identity_params, x)

        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0, atol=1e-6)

    def test_forward_shape(self, key, dim, random_params, batch_size):
        """Forward returns correct shapes."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (batch_size, dim))

        y, ld = transform.forward(random_params, x)

        assert y.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_inverse_shape(self, key, dim, random_params, batch_size):
        """Inverse returns correct shapes."""
        transform = LinearTransform(dim=dim)
        y = jax.random.normal(key, (batch_size, dim))

        x, ld = transform.inverse(random_params, y)

        assert x.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_invertibility(self, key, dim, random_params):
        """forward(inverse(y)) = y and inverse(forward(x)) = x."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (50, dim))

        # Forward then inverse
        y, ld_fwd = transform.forward(random_params, x)
        x_rec, ld_inv = transform.inverse(random_params, y)

        # Relaxed tolerance for triangular solve numerical precision
        max_err = float(jnp.abs(x - x_rec).max())
        assert max_err < 1e-2, f"Max reconstruction error: {max_err}"
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_logdet_consistency(self, key, dim, random_params):
        """log_det_forward = -log_det_inverse at same point."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (20, dim))

        y, ld_fwd = transform.forward(random_params, x)
        _, ld_inv = transform.inverse(random_params, y)

        assert jnp.allclose(ld_fwd, -ld_inv, atol=1e-5)

    def test_logdet_vs_autodiff(self, key, dim, random_params):
        """Log-det matches autodiff Jacobian computation."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (dim,))  # Single sample

        y, ld = transform.forward(random_params, x)

        # Compute via autodiff
        J = jax.jacfwd(lambda z: transform.forward(random_params, z)[0])(x)
        ld_autodiff = jnp.log(jnp.abs(jnp.linalg.det(J)))

        assert jnp.allclose(ld, ld_autodiff, atol=1e-4), f"ld={ld}, autodiff={ld_autodiff}"

    def test_jit_compatible(self, key, dim, random_params):
        """LinearTransform works under JIT."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (10, dim))

        forward_jit = jax.jit(transform.forward)
        inverse_jit = jax.jit(transform.inverse)

        y, ld = forward_jit(random_params, x)
        x_rec, _ = inverse_jit(random_params, y)

        assert y.shape == (10, dim)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_wrong_input_dim_raises(self, dim, identity_params):
        """Wrong input dimension raises ValueError."""
        transform = LinearTransform(dim=dim)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input last dim"):
            transform.forward(identity_params, x_wrong)

    def test_wrong_lower_shape_raises(self, dim):
        """Wrong lower matrix shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim + 1, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="lower must have shape"):
            transform.forward(bad_params, x)

    def test_wrong_upper_shape_raises(self, dim):
        """Wrong upper matrix shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim + 1)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="upper must have shape"):
            transform.forward(bad_params, x)

    def test_wrong_raw_diag_shape_raises(self, dim):
        """Wrong raw_diag shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.zeros(dim + 1),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="raw_diag must have shape"):
            transform.forward(bad_params, x)

    def test_missing_params_raises(self, dim):
        """Missing params keys raises KeyError."""
        transform = LinearTransform(dim=dim)
        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError):
            transform.forward({}, x)


# ============================================================================
# Permutation Tests
# ============================================================================
class TestPermutation:
    """Tests for Permutation transform."""

    @pytest.fixture
    def reverse_perm(self, dim):
        """Reverse permutation [dim-1, dim-2, ..., 0]."""
        return jnp.arange(dim - 1, -1, -1)

    @pytest.fixture
    def identity_perm(self, dim):
        """Identity permutation [0, 1, ..., dim-1]."""
        return jnp.arange(dim)

    def test_forward_reverses(self, dim, reverse_perm):
        """Forward with reverse perm reverses the vector."""
        transform = Permutation(perm=reverse_perm)
        x = jnp.arange(dim, dtype=jnp.float32).reshape(1, dim)

        y, ld = transform.forward({}, x)

        expected = jnp.flip(x, axis=-1)
        assert jnp.allclose(y, expected)
        assert jnp.allclose(ld, 0.0)  # Permutation has unit determinant

    def test_identity_perm(self, key, dim, identity_perm):
        """Identity permutation leaves input unchanged."""
        transform = Permutation(perm=identity_perm)
        x = jax.random.normal(key, (10, dim))

        y, ld = transform.forward({}, x)

        assert jnp.allclose(y, x)
        assert jnp.allclose(ld, 0.0)

    def test_invertibility(self, key, dim, reverse_perm):
        """forward(inverse(y)) = y."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (50, dim))

        y, _ = transform.forward({}, x)
        x_rec, _ = transform.inverse({}, y)

        assert jnp.allclose(x, x_rec, atol=1e-6)

    def test_inverse_reverses_forward(self, key, dim, reverse_perm):
        """Inverse undoes forward."""
        transform = Permutation(perm=reverse_perm)
        y = jax.random.normal(key, (20, dim))

        x, _ = transform.inverse({}, y)
        y_rec, _ = transform.forward({}, x)

        assert jnp.allclose(y, y_rec, atol=1e-6)

    def test_logdet_always_zero(self, key, dim, reverse_perm):
        """Log-det is always zero for permutations."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (30, dim))

        _, ld_fwd = transform.forward({}, x)
        _, ld_inv = transform.inverse({}, x)

        assert jnp.allclose(ld_fwd, 0.0)
        assert jnp.allclose(ld_inv, 0.0)

    def test_output_shape(self, key, dim, reverse_perm, batch_size):
        """Forward/inverse return correct shapes."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (batch_size, dim))

        y, ld = transform.forward({}, x)

        assert y.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_jit_compatible(self, key, dim, reverse_perm):
        """Permutation works under JIT."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (10, dim))

        forward_jit = jax.jit(transform.forward)
        y, _ = forward_jit({}, x)

        assert y.shape == (10, dim)

    def test_wrong_input_dim_raises(self, dim, reverse_perm):
        """Wrong input dimension raises ValueError."""
        transform = Permutation(perm=reverse_perm)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input with last dimension"):
            transform.forward({}, x_wrong)

    def test_non_1d_perm_raises(self):
        """Non-1D permutation raises ValueError."""
        with pytest.raises(ValueError, match="must be 1D"):
            Permutation(perm=jnp.zeros((3, 3)))

    def test_non_integer_perm_raises(self):
        """Non-integer permutation raises TypeError."""
        with pytest.raises(TypeError, match="must be integer"):
            Permutation(perm=jnp.array([0.0, 1.0, 2.0]))

    def test_custom_permutation(self, key):
        """Custom permutation works correctly."""
        perm = jnp.array([2, 0, 3, 1])
        transform = Permutation(perm=perm)
        x = jnp.array([[0, 1, 2, 3]], dtype=jnp.float32)

        y, _ = transform.forward({}, x)

        expected = jnp.array([[2, 0, 3, 1]], dtype=jnp.float32)
        assert jnp.allclose(y, expected)


# ============================================================================
# LoftTransform Tests
# ============================================================================
class TestLoftTransform:
    """Tests for LoftTransform."""

    def test_identity_near_zero(self, key, dim):
        """Near origin, LOFT is approximately identity."""
        transform = LoftTransform(dim=dim, tau=5.0)
        x = jax.random.normal(key, (100, dim)) * 0.1  # Small values

        y, ld = transform.forward({}, x)

        # For |x| << tau, loft(x) ≈ x
        assert jnp.allclose(y, x, atol=0.01)

    def test_invertibility(self, key, dim):
        """forward(inverse(y)) = y."""
        transform = LoftTransform(dim=dim, tau=3.0)
        x = jax.random.normal(key, (50, dim)) * 5  # Mix of small and large

        y, ld_fwd = transform.forward({}, x)
        x_rec, ld_inv = transform.inverse({}, y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_logdet_consistency(self, key, dim):
        """log_det_forward = -log_det_inverse."""
        transform = LoftTransform(dim=dim, tau=2.0)
        x = jax.random.normal(key, (30, dim)) * 3

        y, ld_fwd = transform.forward({}, x)
        _, ld_inv = transform.inverse({}, y)

        assert jnp.allclose(ld_fwd, -ld_inv, atol=1e-5)

    def test_compresses_tails(self, dim):
        """LOFT compresses large values (log behavior in tails)."""
        transform = LoftTransform(dim=dim, tau=1.0)
        x_large = jnp.full((1, dim), 100.0)

        y, _ = transform.forward({}, x_large)

        # y should be much smaller than x due to log compression
        # loft(100) = 1 + log(100 - 1 + 1) = 1 + log(100) ≈ 5.6
        assert jnp.all(y < x_large / 10)

    def test_wrong_dim_raises(self, dim):
        """Wrong input dimension raises ValueError."""
        transform = LoftTransform(dim=dim, tau=1.0)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input last dim"):
            transform.forward({}, x_wrong)

    def test_invalid_dim_raises(self):
        """Non-positive dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LoftTransform(dim=0, tau=1.0)

    def test_invalid_tau_raises(self):
        """Non-positive tau raises ValueError."""
        with pytest.raises(ValueError, match="tau must be strictly positive"):
            LoftTransform(dim=4, tau=0.0)


# ============================================================================
# CompositeTransform Tests
# ============================================================================
class TestCompositeTransform:
    """Tests for CompositeTransform."""

    def test_single_block(self, key, dim):
        """Composite with single block equals that block."""
        perm = jnp.arange(dim - 1, -1, -1)
        block = Permutation(perm=perm)
        composite = CompositeTransform(blocks=[block])

        x = jax.random.normal(key, (20, dim))

        y_composite, ld_composite = composite.forward([{}], x)
        y_block, ld_block = block.forward({}, x)

        assert jnp.allclose(y_composite, y_block)
        assert jnp.allclose(ld_composite, ld_block)

    def test_two_reverses_is_identity(self, key, dim):
        """Two reverse permutations = identity."""
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [Permutation(perm=perm), Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        x = jax.random.normal(key, (30, dim))

        y, ld = composite.forward([{}, {}], x)

        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0)

    def test_invertibility(self, key, dim):
        """Composite transform is invertible."""
        perm = jnp.arange(dim - 1, -1, -1)
        loft = LoftTransform(dim=dim, tau=5.0)
        blocks = [Permutation(perm=perm), loft]
        composite = CompositeTransform(blocks=blocks)

        x = jax.random.normal(key, (40, dim))
        params = [{}, {}]

        y, ld_fwd = composite.forward(params, x)
        x_rec, ld_inv = composite.inverse(params, y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_wrong_params_length_raises(self, dim):
        """Wrong number of param sets raises ValueError."""
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [Permutation(perm=perm), Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="expected 2 param sets"):
            composite.forward([{}], x)  # Only 1 param set for 2 blocks


# ============================================================================
# AffineCoupling Error Handling Tests
# ============================================================================
class TestAffineCouplingErrors:
    """Error handling tests for AffineCoupling."""

    def test_non_1d_mask_raises(self):
        """Non-1D mask raises ValueError."""
        from nflows.nets import MLP
        mlp = MLP(x_dim=4, hidden_dim=8, n_hidden_layers=1, out_dim=8)

        with pytest.raises(ValueError, match="must be 1D"):
            AffineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=2 * dim)
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)

    def test_wrong_input_dim_raises(self, key, dim):
        """Wrong input dimension raises ValueError."""
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp, mlp_params = init_mlp(key, x_dim=dim, context_dim=0, hidden_dim=8, n_hidden_layers=1, out_dim=2 * dim)
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input with last dimension"):
            coupling.forward({"mlp": mlp_params}, x_wrong)


# ============================================================================
# SplineCoupling Error Handling Tests
# ============================================================================
class TestSplineCouplingErrors:
    """Error handling tests for SplineCoupling."""

    def test_non_1d_mask_raises(self):
        """Non-1D mask raises ValueError."""
        from nflows.nets import MLP
        mlp = MLP(x_dim=4, hidden_dim=8, n_hidden_layers=1, out_dim=92)  # 4 * (3*8 - 1)

        with pytest.raises(ValueError, match="must be 1D"):
            SplineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp, num_bins=8)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)


# ============================================================================
# init_params Tests
# ============================================================================
class TestInitParams:
    """Tests for init_params methods on all transforms."""

    def test_linear_transform_init_params_structure(self, key, dim):
        """LinearTransform.init_params returns correct structure."""
        transform = LinearTransform(dim=dim)
        params = transform.init_params(key)

        assert "lower" in params
        assert "upper" in params
        assert "raw_diag" in params
        assert params["lower"].shape == (dim, dim)
        assert params["upper"].shape == (dim, dim)
        assert params["raw_diag"].shape == (dim,)

    def test_linear_transform_init_params_identity(self, key, dim):
        """LinearTransform default init produces identity transform."""
        transform = LinearTransform(dim=dim)
        params = transform.init_params(key)
        x = jax.random.normal(key, (10, dim))

        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_permutation_init_params_empty(self, key, dim):
        """Permutation.init_params returns empty dict."""
        perm = jnp.arange(dim - 1, -1, -1)
        transform = Permutation(perm=perm)
        params = transform.init_params(key)

        assert params == {}

    def test_loft_init_params_empty(self, key, dim):
        """LoftTransform.init_params returns empty dict."""
        transform = LoftTransform(dim=dim, tau=5.0)
        params = transform.init_params(key)

        assert params == {}

    def test_affine_coupling_init_params_structure(self, key, dim):
        """AffineCoupling.init_params returns correct structure."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        params = coupling.init_params(key)

        assert "mlp" in params
        assert "net" in params["mlp"]
        assert "dense_out" in params["mlp"]["net"]

    def test_affine_coupling_init_params_identity(self, key, dim):
        """AffineCoupling default init produces near-identity transform."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1)
        x = jax.random.normal(k2, (10, dim))

        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_affine_coupling_init_params_with_context(self, key, dim, context_dim):
        """AffineCoupling.init_params works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1, context_dim=context_dim)
        x = jax.random.normal(k2, (10, dim))
        ctx = jax.random.normal(k2, (10, context_dim))

        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_spline_coupling_init_params_structure(self, key, dim):
        """SplineCoupling.init_params returns correct structure."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        params = coupling.init_params(key)

        assert "mlp" in params
        assert "net" in params["mlp"]
        assert "dense_out" in params["mlp"]["net"]

    def test_spline_coupling_init_params_near_identity(self, key, dim):
        """SplineCoupling default init produces near-identity transform."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1)
        # Use values within spline bounds
        x = jax.random.uniform(k2, (10, dim), minval=-4.0, maxval=4.0)

        y, ld = coupling.forward(params, x)

        # Should be close to identity (within tolerance for spline approx)
        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_spline_coupling_init_params_with_context(self, key, dim, context_dim):
        """SplineCoupling.init_params works with context_dim > 0."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1, context_dim=context_dim)
        x = jax.random.uniform(k2, (10, dim), minval=-4.0, maxval=4.0)
        ctx = jax.random.normal(k2, (10, context_dim))

        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_composite_init_params_structure(self, key, dim):
        """CompositeTransform.init_params returns list of correct length."""
        perm = jnp.arange(dim - 1, -1, -1)
        loft = LoftTransform(dim=dim, tau=5.0)
        blocks = [Permutation(perm=perm), loft, Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        params = composite.init_params(key)

        assert isinstance(params, list)
        assert len(params) == 3
        assert params[0] == {}  # Permutation
        assert params[1] == {}  # LoftTransform
        assert params[2] == {}  # Permutation

    def test_composite_init_params_with_coupling(self, key, dim):
        """CompositeTransform.init_params works with coupling blocks."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [coupling, Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        params = composite.init_params(key)

        assert len(params) == 2
        assert "mlp" in params[0]  # AffineCoupling params
        assert params[1] == {}      # Permutation params

    def test_composite_init_params_with_context(self, key, dim, context_dim):
        """CompositeTransform.init_params passes context_dim to sub-blocks."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)
        composite = CompositeTransform(blocks=[coupling])

        params = composite.init_params(key, context_dim=context_dim)
        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))

        # Should work with context
        y, ld = composite.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)


# ============================================================================
# Factory Method Tests (required_out_dim and create)
# ============================================================================
class TestAffineCouplingFactory:
    """Tests for AffineCoupling.required_out_dim and create."""

    def test_required_out_dim(self):
        """required_out_dim returns 2 * dim."""
        assert AffineCoupling.required_out_dim(4) == 8
        assert AffineCoupling.required_out_dim(10) == 20
        assert AffineCoupling.required_out_dim(1) == 2

    def test_create_returns_coupling_and_params(self, key, dim):
        """create returns (coupling, params) tuple."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        assert isinstance(coupling, AffineCoupling)
        assert isinstance(params, dict)
        assert "mlp" in params

    def test_create_works_without_context_dim_in_init_params(self, key, dim):
        """create + init_params works without specifying context_dim (inferred)."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_identity_at_init(self, key, dim):
        """create produces identity transform at init."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_with_context(self, key, dim, context_dim):
        """create works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            context_dim=context_dim,
        )

        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))
        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        mask = jnp.array([1, 0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="dim must be positive"):
            AffineCoupling.create(key, dim=0, mask=mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_mask_mismatch_raises(self, key, dim):
        """create raises ValueError when mask doesn't match dim."""
        wrong_mask = jnp.array([1, 0, 1], dtype=jnp.float32)
        with pytest.raises(ValueError, match="mask shape"):
            AffineCoupling.create(key, dim=dim, mask=wrong_mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_invalid_hidden_dim_raises(self, key, dim):
        """create raises ValueError for hidden_dim <= 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            AffineCoupling.create(key, dim=dim, mask=mask, hidden_dim=0, n_hidden_layers=2)


class TestSplineCouplingFactory:
    """Tests for SplineCoupling.required_out_dim and create."""

    def test_required_out_dim(self):
        """required_out_dim returns dim * (3K - 1)."""
        assert SplineCoupling.required_out_dim(4, num_bins=8) == 4 * (3 * 8 - 1)  # 92
        assert SplineCoupling.required_out_dim(2, num_bins=4) == 2 * (3 * 4 - 1)  # 22
        assert SplineCoupling.required_out_dim(1, num_bins=2) == 1 * (3 * 2 - 1)  # 5

    def test_create_returns_coupling_and_params(self, key, dim):
        """create returns (coupling, params) tuple."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        assert isinstance(coupling, SplineCoupling)
        assert isinstance(params, dict)
        assert "mlp" in params

    def test_create_works_without_context_dim_in_init_params(self, key, dim):
        """create + init_params works without specifying context_dim (inferred)."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_near_identity_at_init(self, key, dim):
        """create produces near-identity transform at init."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_create_with_context(self, key, dim, context_dim):
        """create works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            num_bins=8, context_dim=context_dim,
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        ctx = jax.random.normal(key, (10, context_dim))
        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        mask = jnp.array([1, 0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="dim must be positive"):
            SplineCoupling.create(key, dim=0, mask=mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_mask_mismatch_raises(self, key, dim):
        """create raises ValueError when mask doesn't match dim."""
        wrong_mask = jnp.array([1, 0, 1], dtype=jnp.float32)
        with pytest.raises(ValueError, match="mask shape"):
            SplineCoupling.create(key, dim=dim, mask=wrong_mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_invalid_num_bins_raises(self, key, dim):
        """create raises ValueError for num_bins <= 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="num_bins must be positive"):
            SplineCoupling.create(key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=0)


class TestCustomConditionerWithRequiredOutDim:
    """Tests that raw constructor + required_out_dim works for custom architectures."""

    def test_affine_coupling_custom_conditioner(self, key, dim):
        """Raw constructor works with custom MLP using required_out_dim."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        out_dim = AffineCoupling.required_out_dim(dim)

        # Custom MLP with different config
        from nflows.nets import MLP
        custom_mlp = MLP(
            x_dim=dim, context_dim=0, hidden_dim=128, n_hidden_layers=5, out_dim=out_dim
        )

        coupling = AffineCoupling(mask=mask, conditioner=custom_mlp)
        params = coupling.init_params(key)  # Should infer context_dim=0

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape

    def test_spline_coupling_custom_conditioner(self, key, dim):
        """Raw constructor works with custom MLP using required_out_dim."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = SplineCoupling.required_out_dim(dim, num_bins)

        # Custom MLP with different config
        from nflows.nets import MLP
        custom_mlp = MLP(
            x_dim=dim, context_dim=0, hidden_dim=128, n_hidden_layers=5, out_dim=out_dim
        )

        coupling = SplineCoupling(mask=mask, conditioner=custom_mlp, num_bins=num_bins)
        params = coupling.init_params(key)  # Should infer context_dim=0

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape


class TestConditionerInterfaceFallback:
    """Tests for conditioner interface fallback behavior."""

    def test_affine_coupling_skips_init_without_interface(self, key, dim):
        """AffineCoupling skips auto-init if conditioner lacks interface methods."""
        from flax import linen as nn

        # Minimal conditioner without get_output_layer/set_output_layer
        class MinimalConditioner(nn.Module):
            out_dim: int
            context_dim: int = 0

            @nn.compact
            def __call__(self, x, context=None):
                return nn.Dense(self.out_dim)(x)

        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        out_dim = AffineCoupling.required_out_dim(dim)
        conditioner = MinimalConditioner(out_dim=out_dim)

        coupling = AffineCoupling(mask=mask, conditioner=conditioner)
        params = coupling.init_params(key)

        # Should work without error (skips auto-init)
        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)
        assert y.shape == x.shape

    def test_spline_coupling_raises_without_interface(self, key, dim):
        """SplineCoupling raises error if conditioner lacks interface methods."""
        from flax import linen as nn

        # Minimal conditioner without get_output_layer/set_output_layer
        class MinimalConditioner(nn.Module):
            out_dim: int
            context_dim: int = 0

            @nn.compact
            def __call__(self, x, context=None):
                return nn.Dense(self.out_dim)(x)

        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = SplineCoupling.required_out_dim(dim, num_bins)
        conditioner = MinimalConditioner(out_dim=out_dim)

        coupling = SplineCoupling(mask=mask, conditioner=conditioner, num_bins=num_bins)

        with pytest.raises(RuntimeError, match="get_output_layer.*set_output_layer"):
            coupling.init_params(key)


# ============================================================================
# Factory Tests for Simple Transforms (LinearTransform, Permutation, LoftTransform)
# ============================================================================
class TestLinearTransformFactory:
    """Tests for LinearTransform.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        transform, params = LinearTransform.create(key, dim=dim)

        assert isinstance(transform, LinearTransform)
        assert isinstance(params, dict)
        assert "lower" in params
        assert "upper" in params
        assert "raw_diag" in params

    def test_create_identity_at_init(self, key, dim):
        """create produces identity transform at init."""
        transform, params = LinearTransform.create(key, dim=dim)

        x = jax.random.normal(key, (10, dim))
        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LinearTransform.create(key, dim=0)

    def test_create_with_context(self, key, dim):
        """create with context_dim > 0 produces conditioner."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        assert transform.conditioner is not None
        assert transform.context_dim == context_dim
        assert "mlp" in params

    def test_create_with_context_identity_at_init(self, key, dim):
        """Conditional LinearTransform is identity at initialization."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))
        y, ld = transform.forward(params, x, context)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_with_context_invertibility(self, key, dim):
        """Conditional LinearTransform is invertible."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        # Perturb params away from identity
        k1, k2 = jax.random.split(key)
        params["lower"] = jax.random.normal(k1, (dim, dim)) * 0.1
        params["upper"] = jax.random.normal(k2, (dim, dim)) * 0.1

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        y, ld_fwd = transform.forward(params, x, context)
        x_rec, ld_inv = transform.inverse(params, y, context)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_create_with_context_different_contexts_give_different_outputs(self, key, dim):
        """Different contexts produce different transformations."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        # Perturb MLP output layer to get non-zero conditioner output
        # (at init, output layer is zero so all contexts give same result)
        k1 = jax.random.PRNGKey(999)
        net_params = params["mlp"]["net"]
        net_params["dense_out"]["kernel"] = jax.random.normal(
            k1, net_params["dense_out"]["kernel"].shape
        ) * 0.5

        x = jax.random.normal(key, (5, dim))
        context1 = jax.random.normal(jax.random.PRNGKey(1), (5, context_dim))
        context2 = jax.random.normal(jax.random.PRNGKey(2), (5, context_dim))

        y1, _ = transform.forward(params, x, context1)
        y2, _ = transform.forward(params, x, context2)

        # Outputs should differ for different contexts
        assert not jnp.allclose(y1, y2, atol=1e-3)

    def test_create_invalid_context_dim_raises(self, key, dim):
        """create raises ValueError for context_dim < 0."""
        with pytest.raises(ValueError, match="context_dim must be non-negative"):
            LinearTransform.create(key, dim=dim, context_dim=-1)


class TestPermutationFactory:
    """Tests for Permutation.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        perm = jnp.arange(dim - 1, -1, -1)
        transform, params = Permutation.create(key, perm=perm)

        assert isinstance(transform, Permutation)
        assert params == {}

    def test_create_works(self, key, dim):
        """create produces working transform."""
        perm = jnp.arange(dim - 1, -1, -1)  # reverse
        transform, params = Permutation.create(key, perm=perm)

        x = jax.random.normal(key, (10, dim))
        y, ld = transform.forward(params, x)

        # Reverse permutation should reverse the last dimension
        assert jnp.allclose(y, x[..., ::-1])
        assert jnp.allclose(ld, 0.0)


class TestLoftTransformFactory:
    """Tests for LoftTransform.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)

        assert isinstance(transform, LoftTransform)
        assert params == {}

    def test_create_near_identity_near_zero(self, key, dim):
        """create produces near-identity transform for small inputs."""
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)

        # Small inputs should be nearly unchanged
        x = jax.random.normal(key, (10, dim)) * 0.1
        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-4)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LoftTransform.create(key, dim=0)

    def test_create_invalid_tau_raises(self, key, dim):
        """create raises ValueError for tau <= 0."""
        with pytest.raises(ValueError, match="tau must be positive"):
            LoftTransform.create(key, dim=dim, tau=0.0)


# ============================================================================
# LOFT overflow regression tests (C1)
# ============================================================================
class TestLoftOverflow:
    """Regression tests for LOFT inverse overflow with large inputs."""

    def test_loft_inv_large_input_finite(self):
        """loft_inv must return finite values for |y| >> tau."""
        from nflows.scalar_function import loft_inv
        y = jnp.array([2000.0, -2000.0, 1100.0, -1100.0])
        x = loft_inv(y, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"

    def test_loft_roundtrip_large_z(self):
        """loft(loft_inv(y)) should roundtrip for y within recoverable range.

        The clamp at 80 means exact roundtrip works for |y| <= tau + 80.
        Values beyond that saturate but remain finite.
        """
        from nflows.scalar_function import loft, loft_inv
        # Values within recoverable range (tau=1000, max exact = 1080)
        y = jnp.array([1050.0, -1050.0, 1079.0, -1079.0])
        x = loft_inv(y, tau=1000.0)
        y_rec = loft(x, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"
        assert jnp.allclose(y_rec, y, atol=1e-3), (
            f"roundtrip failed: max err={jnp.max(jnp.abs(y_rec - y))}"
        )

    def test_loft_inv_beyond_clamp_saturates_finitely(self):
        """Values beyond tau+80 saturate but remain finite (no inf/nan)."""
        from nflows.scalar_function import loft_inv
        y = jnp.array([1200.0, -1200.0, 5000.0, -5000.0])
        x = loft_inv(y, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"

    def test_loft_transform_inverse_large(self, key):
        """LoftTransform.inverse must handle large values without overflow."""
        dim = 4
        transform, params = LoftTransform.create(key, dim=dim, tau=1000.0)
        y = jnp.full((5, dim), 2000.0)
        x, log_det = transform.inverse(params, y)
        assert jnp.all(jnp.isfinite(x)), f"inverse produced non-finite x: {x}"
        assert jnp.all(jnp.isfinite(log_det)), f"inverse produced non-finite log_det: {log_det}"


# ============================================================================
# Log-det vs autodiff correctness tests (C3)
# ============================================================================
class TestLogdetVsAutodiff:
    """Verify hand-derived log-det formulas match autodiff Jacobian.

    Uses check_logdet_vs_autodiff from conftest.py on single samples (no batch).
    """

    def test_affine_coupling(self):
        """AffineCoupling log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"AffineCoupling log-det error: {result['error']}"
        )

    def test_affine_coupling_conditional(self):
        """AffineCoupling with context: log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1,
            context_dim=2
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        ctx = jax.random.normal(jax.random.PRNGKey(2), (2,))
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z, ctx), x
        )
        assert result["error"] < 1e-4, (
            f"AffineCoupling conditional log-det error: {result['error']}"
        )

    def test_spline_coupling(self):
        """SplineCoupling log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1,
            num_bins=8
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,)) * 0.5
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z), x
        )
        assert result["error"] < 1e-3, (
            f"SplineCoupling log-det error: {result['error']}"
        )

    def test_linear_transform(self):
        """LinearTransform log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        transform, _ = LinearTransform.create(key, dim=dim)
        params = {
            "lower": jax.random.normal(key, (dim, dim)) * 0.1,
            "upper": jax.random.normal(jax.random.PRNGKey(1), (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(jax.random.PRNGKey(2), (dim,)) * 0.5,
        }
        x = jax.random.normal(jax.random.PRNGKey(3), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"LinearTransform log-det error: {result['error']}"
        )

    def test_loft_transform(self):
        """LoftTransform log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,)) * 3.0
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"LoftTransform log-det error: {result['error']}"
        )

    def test_permutation(self):
        """Permutation log-det is zero (matches autodiff)."""

        dim = 4
        perm = jnp.array([2, 0, 3, 1])
        transform, params = Permutation.create(jax.random.PRNGKey(0), perm=perm)
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-6, (
            f"Permutation log-det error: {result['error']}"
        )

    def test_composite_transform(self):
        """CompositeTransform (affine + linear) log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        k1, k2 = jax.random.split(key)
        coupling, c_params = AffineCoupling.create(
            k1, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1
        )
        linear, _ = LinearTransform.create(k2, dim=dim)
        l_params = {
            "lower": jax.random.normal(k2, (dim, dim)) * 0.1,
            "upper": jax.random.normal(jax.random.PRNGKey(10), (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(jax.random.PRNGKey(11), (dim,)) * 0.3,
        }
        composite = CompositeTransform(blocks=[coupling, linear])
        all_params = [c_params, l_params]
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: composite.forward(all_params, z), x
        )
        assert result["error"] < 1e-3, (
            f"CompositeTransform log-det error: {result['error']}"
        )
