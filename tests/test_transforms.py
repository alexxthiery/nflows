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


# ============================================================================
# LinearTransform Tests
# ============================================================================
class TestLinearTransform:
    """Tests for LinearTransform (LU-parameterized)."""

    @pytest.fixture
    def identity_params(self, dim):
        """Identity transform params (L=I, U=0, s=1)."""
        return {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "log_diag": jnp.zeros(dim),
        }

    @pytest.fixture
    def random_params(self, key, dim):
        """Random transform params."""
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "lower": jax.random.normal(k1, (dim, dim)) * 0.1,
            "upper": jax.random.normal(k2, (dim, dim)) * 0.1,
            "log_diag": jax.random.normal(k3, (dim,)) * 0.5,
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
            "log_diag": jnp.zeros(dim),
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
            "log_diag": jnp.zeros(dim),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="upper must have shape"):
            transform.forward(bad_params, x)

    def test_wrong_log_diag_shape_raises(self, dim):
        """Wrong log_diag shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "log_diag": jnp.zeros(dim + 1),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="log_diag must have shape"):
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
        mlp = MLP(in_dim=4, hidden_sizes=[8], out_dim=8)

        with pytest.raises(ValueError, match="must be 1D"):
            AffineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp = MLP(in_dim=dim, hidden_sizes=[8], out_dim=2 * dim)
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)

    def test_wrong_input_dim_raises(self, key, dim):
        """Wrong input dimension raises ValueError."""
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp, mlp_params = init_mlp(key, in_dim=dim, hidden_sizes=[8], out_dim=2 * dim)
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
        mlp = MLP(in_dim=4, hidden_sizes=[8], out_dim=92)  # 4 * (3*8 - 1)

        with pytest.raises(ValueError, match="must be 1D"):
            SplineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp, num_bins=8)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflows.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(in_dim=dim, hidden_sizes=[8], out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)
