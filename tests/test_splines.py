# tests/test_splines.py
"""Direct tests for rational-quadratic spline primitives."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from nflows.splines import (
    _normalize_bin_params,
    _select_bins,
    _gather_bin_params,
    _rational_quadratic_forward_inner,
    _rational_quadratic_inverse_inner,
    rational_quadratic_spline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spline_params(key, batch_shape=(), num_bins=8):
    """Generate random unnormalized spline parameters."""
    k1, k2, k3 = jr.split(key, 3)
    widths = jr.normal(k1, batch_shape + (num_bins,))
    heights = jr.normal(k2, batch_shape + (num_bins,))
    derivs = jr.normal(k3, batch_shape + (num_bins - 1,))
    return widths, heights, derivs


def _make_normalized_params(key, batch_shape=(), num_bins=8, tail_bound=3.0,
                            min_bin_width=1e-3, min_bin_height=1e-3,
                            min_derivative=1e-3, max_derivative=10.0):
    """Generate normalized spline parameters (x_k, y_k, derivatives)."""
    widths, heights, derivs = _random_spline_params(key, batch_shape, num_bins)
    return _normalize_bin_params(
        widths, heights, derivs,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        max_derivative=max_derivative,
    )


# ---------------------------------------------------------------------------
# _normalize_bin_params
# ---------------------------------------------------------------------------

class TestNormalizeBinParams:

    def test_output_shapes(self, key):
        """x_k, y_k have K+1 entries; derivatives have K+1 entries."""
        K = 8
        x_k, y_k, d = _make_normalized_params(key, batch_shape=(5,), num_bins=K)
        assert x_k.shape == (5, K + 1)
        assert y_k.shape == (5, K + 1)
        assert d.shape == (5, K + 1)

    def test_boundary_values(self, key):
        """First and last knots are exactly -B and +B."""
        B = 4.0
        x_k, y_k, _ = _make_normalized_params(key, num_bins=8, tail_bound=B)
        assert float(x_k[0]) == pytest.approx(-B, abs=0)
        assert float(x_k[-1]) == pytest.approx(B, abs=0)
        assert float(y_k[0]) == pytest.approx(-B, abs=0)
        assert float(y_k[-1]) == pytest.approx(B, abs=0)

    def test_knots_strictly_increasing(self, key):
        """x_k and y_k must be strictly increasing."""
        x_k, y_k, _ = _make_normalized_params(key, batch_shape=(10,), num_bins=8)
        dx = jnp.diff(x_k, axis=-1)
        dy = jnp.diff(y_k, axis=-1)
        assert jnp.all(dx > 0), f"x_k not strictly increasing: min diff = {dx.min()}"
        assert jnp.all(dy > 0), f"y_k not strictly increasing: min diff = {dy.min()}"

    def test_widths_sum_to_2B(self, key):
        """Bin widths should sum to 2*tail_bound."""
        B = 3.0
        x_k, _, _ = _make_normalized_params(key, num_bins=8, tail_bound=B)
        total_width = float(x_k[-1] - x_k[0])
        assert total_width == pytest.approx(2.0 * B, abs=1e-6)

    def test_heights_sum_to_2B(self, key):
        """Bin heights should sum to 2*tail_bound."""
        B = 3.0
        _, y_k, _ = _make_normalized_params(key, num_bins=8, tail_bound=B)
        total_height = float(y_k[-1] - y_k[0])
        assert total_height == pytest.approx(2.0 * B, abs=1e-6)

    def test_min_bin_width_respected(self, key):
        """Each bin width >= min_bin_width * 2 * B."""
        B = 3.0
        min_w = 0.05
        x_k, _, _ = _make_normalized_params(
            key, num_bins=8, tail_bound=B, min_bin_width=min_w
        )
        widths = jnp.diff(x_k)
        min_actual = float(widths.min())
        assert min_actual >= min_w * 2 * B - 1e-6

    def test_min_bin_height_respected(self, key):
        """Each bin height >= min_bin_height * 2 * B."""
        B = 3.0
        min_h = 0.05
        _, y_k, _ = _make_normalized_params(
            key, num_bins=8, tail_bound=B, min_bin_height=min_h
        )
        heights = jnp.diff(y_k)
        min_actual = float(heights.min())
        assert min_actual >= min_h * 2 * B - 1e-6

    def test_boundary_derivatives_are_one(self, key):
        """Boundary derivatives must be exactly 1 for smooth tails."""
        _, _, d = _make_normalized_params(key, num_bins=8)
        assert float(d[0]) == pytest.approx(1.0, abs=0)
        assert float(d[-1]) == pytest.approx(1.0, abs=0)

    def test_internal_derivatives_bounded(self, key):
        """Internal derivatives in [min_derivative, max_derivative]."""
        min_d, max_d = 0.01, 5.0
        _, _, d = _make_normalized_params(
            key, batch_shape=(20,), num_bins=8,
            min_derivative=min_d, max_derivative=max_d,
        )
        internal = d[:, 1:-1]
        assert jnp.all(internal >= min_d - 1e-7)
        assert jnp.all(internal <= max_d + 1e-7)

    def test_error_infeasible_min_bin_width(self):
        """min_bin_width * K >= 1 should raise."""
        K = 4
        with pytest.raises(ValueError, match="min_bin_width"):
            _normalize_bin_params(
                jnp.zeros(K), jnp.zeros(K), jnp.zeros(K - 1),
                tail_bound=3.0, min_bin_width=0.3,
                min_bin_height=1e-3, min_derivative=1e-3, max_derivative=10.0,
            )

    def test_error_infeasible_min_bin_height(self):
        """min_bin_height * K >= 1 should raise."""
        K = 4
        with pytest.raises(ValueError, match="min_bin_height"):
            _normalize_bin_params(
                jnp.zeros(K), jnp.zeros(K), jnp.zeros(K - 1),
                tail_bound=3.0, min_bin_width=1e-3,
                min_bin_height=0.3, min_derivative=1e-3, max_derivative=10.0,
            )

    def test_error_derivative_shape_mismatch(self):
        """Derivative dim must be K-1."""
        K = 4
        with pytest.raises(ValueError, match="unnormalized_derivatives"):
            _normalize_bin_params(
                jnp.zeros(K), jnp.zeros(K), jnp.zeros(K),  # K instead of K-1
                tail_bound=3.0, min_bin_width=1e-3,
                min_bin_height=1e-3, min_derivative=1e-3, max_derivative=10.0,
            )

    def test_single_bin(self, key):
        """K=1: one bin spanning [-B, B], no internal derivatives."""
        K = 1
        k1, k2 = jr.split(key)
        w = jr.normal(k1, (K,))
        h = jr.normal(k2, (K,))
        d = jnp.zeros((0,))  # K-1 = 0
        x_k, y_k, derivs = _normalize_bin_params(
            w, h, d,
            tail_bound=3.0, min_bin_width=1e-3,
            min_bin_height=1e-3, min_derivative=1e-3, max_derivative=10.0,
        )
        assert x_k.shape == (2,)
        assert float(x_k[0]) == pytest.approx(-3.0, abs=0)
        assert float(x_k[1]) == pytest.approx(3.0, abs=0)

    def test_batch_shapes(self, key):
        """Works with multi-dimensional batch shapes."""
        x_k, y_k, d = _make_normalized_params(
            key, batch_shape=(3, 4), num_bins=5
        )
        assert x_k.shape == (3, 4, 6)
        assert y_k.shape == (3, 4, 6)
        assert d.shape == (3, 4, 6)


# ---------------------------------------------------------------------------
# _select_bins
# ---------------------------------------------------------------------------

class TestSelectBins:

    def test_interior_points(self):
        """Points inside each bin are assigned the correct bin index."""
        # 3 bins: [-3, -1, 1, 3]
        x_k = jnp.array([-3.0, -1.0, 1.0, 3.0])
        x = jnp.array([-2.0, 0.0, 2.0])
        idx, num_bins = _select_bins(x, x_k)
        assert num_bins == 3
        assert list(idx.tolist()) == [0, 1, 2]

    def test_left_edge_goes_to_bin(self):
        """x exactly at a left edge goes to that bin."""
        x_k = jnp.array([-3.0, -1.0, 1.0, 3.0])
        x = jnp.array([-1.0])  # edge between bin 0 and 1
        idx, _ = _select_bins(x, x_k)
        assert int(idx[0]) == 1  # x >= -1.0, so bin 1

    def test_right_boundary_clamped(self):
        """x at the right boundary is clamped to last bin."""
        x_k = jnp.array([-3.0, -1.0, 1.0, 3.0])
        x = jnp.array([3.0])
        idx, _ = _select_bins(x, x_k)
        assert int(idx[0]) == 2  # clamped to K-1

    def test_batch_shape(self):
        """Works with batch dimensions."""
        x_k = jnp.array([[-3.0, 0.0, 3.0], [-3.0, 0.0, 3.0]])
        x = jnp.array([-1.0, 1.0])
        idx, num_bins = _select_bins(x, x_k)
        assert idx.shape == (2,)
        assert num_bins == 2


# ---------------------------------------------------------------------------
# _gather_bin_params
# ---------------------------------------------------------------------------

class TestGatherBinParams:

    def test_correct_values(self):
        """Gathered values match manual indexing."""
        # Use 2D arrays: (batch=2, K+1=4)
        x_k = jnp.array([[-3.0, -1.0, 1.0, 3.0],
                          [-3.0, -1.0, 1.0, 3.0]])
        y_k = jnp.array([[-3.0, -0.5, 0.5, 3.0],
                          [-3.0, -0.5, 0.5, 3.0]])
        d = jnp.array([[1.0, 1.5, 2.0, 1.0],
                        [1.0, 1.5, 2.0, 1.0]])
        bin_idx = jnp.array([1, 2])

        xl, xr, yl, yr, dl, dr = _gather_bin_params(x_k, y_k, d, bin_idx)
        # bin 1: x_left=-1, x_right=1
        assert float(xl[0]) == pytest.approx(-1.0)
        assert float(xr[0]) == pytest.approx(1.0)
        assert float(yl[0]) == pytest.approx(-0.5)
        assert float(yr[0]) == pytest.approx(0.5)
        assert float(dl[0]) == pytest.approx(1.5)
        assert float(dr[0]) == pytest.approx(2.0)
        # bin 2: x_left=1, x_right=3
        assert float(xl[1]) == pytest.approx(1.0)
        assert float(xr[1]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Forward / inverse roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Test forward(inverse(y)) = y and inverse(forward(x)) = x."""

    @pytest.mark.parametrize("num_bins", [1, 4, 8, 32])
    def test_invertibility_various_K(self, key, num_bins):
        """Roundtrip holds for various numbers of bins."""
        k1, k2 = jr.split(key)
        w, h, d = _random_spline_params(k1, batch_shape=(50,), num_bins=num_bins)
        x = jr.uniform(k2, (50,), minval=-2.5, maxval=2.5)

        y, ld_fwd = rational_quadratic_spline(
            x, w, h, d, tail_bound=3.0, inverse=False
        )
        x_rec, ld_inv = rational_quadratic_spline(
            y, w, h, d, tail_bound=3.0, inverse=True
        )

        assert jnp.allclose(x, x_rec, atol=1e-4), (
            f"max recon error = {jnp.abs(x - x_rec).max()}"
        )
        # float32 quadratic solver in inverse gives ~3e-4 logdet error
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=5e-4), (
            f"max logdet error = {jnp.abs(ld_fwd + ld_inv).max()}"
        )

    @pytest.mark.parametrize("tail_bound", [1.0, 3.0, 10.0])
    def test_invertibility_various_bounds(self, key, tail_bound):
        """Roundtrip holds for various tail bounds."""
        k1, k2 = jr.split(key)
        w, h, d = _random_spline_params(k1, batch_shape=(50,), num_bins=8)
        x = jr.uniform(k2, (50,), minval=-tail_bound * 0.9, maxval=tail_bound * 0.9)

        y, ld_fwd = rational_quadratic_spline(
            x, w, h, d, tail_bound=tail_bound, inverse=False
        )
        x_rec, ld_inv = rational_quadratic_spline(
            y, w, h, d, tail_bound=tail_bound, inverse=True
        )
        assert jnp.allclose(x, x_rec, atol=1e-4)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=5e-4)

    def test_invertibility_batch_2d(self, key):
        """Roundtrip with 2D batch shape."""
        k1, k2 = jr.split(key)
        shape = (4, 8)
        w, h, d = _random_spline_params(k1, batch_shape=shape, num_bins=8)
        x = jr.uniform(k2, shape, minval=-2.5, maxval=2.5)

        y, ld_fwd = rational_quadratic_spline(
            x, w, h, d, tail_bound=3.0, inverse=False
        )
        x_rec, ld_inv = rational_quadratic_spline(
            y, w, h, d, tail_bound=3.0, inverse=True
        )
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_inverse_of_inverse(self, key):
        """inverse(forward(x)) recovers x; separate path from forward(inverse)."""
        k1, k2 = jr.split(key)
        w, h, d = _random_spline_params(k1, batch_shape=(50,), num_bins=8)
        y = jr.uniform(k2, (50,), minval=-2.5, maxval=2.5)

        x, ld_inv = rational_quadratic_spline(
            y, w, h, d, tail_bound=3.0, inverse=True
        )
        y_rec, ld_fwd = rational_quadratic_spline(
            x, w, h, d, tail_bound=3.0, inverse=False
        )
        assert jnp.allclose(y, y_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# Tail behavior
# ---------------------------------------------------------------------------

class TestTails:

    def test_identity_outside_bound(self, key):
        """Outside [-B, B], transform is identity with log_det=0."""
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=8)
        B = 3.0
        x_tail = jnp.array([-10.0, -3.5, 3.5, 10.0, 100.0])

        # Broadcast params
        w_b = jnp.broadcast_to(w, (5,) + w.shape)
        h_b = jnp.broadcast_to(h, (5,) + h.shape)
        d_b = jnp.broadcast_to(d, (5,) + d.shape)

        y, ld = rational_quadratic_spline(
            x_tail, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        assert jnp.allclose(y, x_tail, atol=1e-7)
        assert jnp.allclose(ld, 0.0, atol=1e-7)

    def test_inverse_identity_outside_bound(self, key):
        """Inverse is also identity outside [-B, B]."""
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=8)
        B = 3.0
        y_tail = jnp.array([-10.0, -3.5, 3.5, 10.0])

        w_b = jnp.broadcast_to(w, (4,) + w.shape)
        h_b = jnp.broadcast_to(h, (4,) + h.shape)
        d_b = jnp.broadcast_to(d, (4,) + d.shape)

        x, ld = rational_quadratic_spline(
            y_tail, w_b, h_b, d_b, tail_bound=B, inverse=True
        )
        assert jnp.allclose(x, y_tail, atol=1e-7)
        assert jnp.allclose(ld, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------

class TestBoundary:

    def test_boundary_maps_to_boundary(self, key):
        """x = -B maps to y = -B; x = +B maps to y = +B."""
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=8)
        B = 3.0
        x_boundary = jnp.array([-B, B])

        w_b = jnp.broadcast_to(w, (2,) + w.shape)
        h_b = jnp.broadcast_to(h, (2,) + h.shape)
        d_b = jnp.broadcast_to(d, (2,) + d.shape)

        y, _ = rational_quadratic_spline(
            x_boundary, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        assert float(y[0]) == pytest.approx(-B, abs=1e-5)
        assert float(y[1]) == pytest.approx(B, abs=1e-5)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:

    def test_monotone_forward(self, key):
        """Forward transform is strictly monotone: x1 < x2 => y1 < y2."""
        k1, k2 = jr.split(key)
        K = 8
        B = 3.0
        n = 200
        w, h, d = _random_spline_params(k1, batch_shape=(), num_bins=K)

        x = jnp.linspace(-B, B, n)
        w_b = jnp.broadcast_to(w, (n,) + w.shape)
        h_b = jnp.broadcast_to(h, (n,) + h.shape)
        d_b = jnp.broadcast_to(d, (n,) + d.shape)

        y, _ = rational_quadratic_spline(
            x, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        dy = jnp.diff(y)
        assert jnp.all(dy > 0), f"Not monotone: min dy = {dy.min()}"

    def test_positive_derivative(self, key):
        """log_det should be finite (derivative > 0 everywhere inside)."""
        k1, k2 = jr.split(key)
        K = 8
        B = 3.0
        n = 200
        w, h, d = _random_spline_params(k1, batch_shape=(), num_bins=K)

        x = jnp.linspace(-B + 0.01, B - 0.01, n)
        w_b = jnp.broadcast_to(w, (n,) + w.shape)
        h_b = jnp.broadcast_to(h, (n,) + h.shape)
        d_b = jnp.broadcast_to(d, (n,) + d.shape)

        _, ld = rational_quadratic_spline(
            x, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        assert jnp.all(jnp.isfinite(ld)), f"Non-finite log_det found"


# ---------------------------------------------------------------------------
# Log-det vs autodiff
# ---------------------------------------------------------------------------

class TestLogDet:

    def test_logdet_matches_autodiff(self, key):
        """Analytical log_det matches JAX autodiff for scalar input."""
        k1, k2 = jr.split(key)
        K = 8
        B = 3.0
        w, h, d = _random_spline_params(k1, batch_shape=(), num_bins=K)

        x_vals = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        for x_val in x_vals:
            x = jnp.array(x_val)
            # Scalar forward
            def fwd(z):
                y, _ = rational_quadratic_spline(
                    z, w, h, d, tail_bound=B, inverse=False
                )
                return y

            _, ld = rational_quadratic_spline(
                x, w, h, d, tail_bound=B, inverse=False
            )
            # autodiff derivative
            dydx = jax.grad(fwd)(x)
            ld_auto = jnp.log(jnp.abs(dydx))

            assert float(ld) == pytest.approx(float(ld_auto), abs=1e-4), (
                f"x={x_val}: ld={float(ld)}, ld_auto={float(ld_auto)}"
            )

    def test_logdet_inverse_matches_autodiff(self, key):
        """Inverse log_det matches autodiff."""
        k1, k2 = jr.split(key)
        K = 8
        B = 3.0
        w, h, d = _random_spline_params(k1, batch_shape=(), num_bins=K)

        # Scalar forward to get a valid y value
        x_scalar = jnp.array(0.5)
        y_scalar, _ = rational_quadratic_spline(
            x_scalar, w, h, d, tail_bound=B, inverse=False
        )

        def inv(z):
            x, _ = rational_quadratic_spline(
                z, w, h, d, tail_bound=B, inverse=True
            )
            return x

        _, ld_inv = rational_quadratic_spline(
            y_scalar, w, h, d, tail_bound=B, inverse=True
        )
        dxdy = jax.grad(inv)(y_scalar)
        ld_auto = jnp.log(jnp.abs(dxdy))

        assert float(ld_inv) == pytest.approx(float(ld_auto), abs=1e-4)


# ---------------------------------------------------------------------------
# Near-identity transform
# ---------------------------------------------------------------------------

class TestNearIdentity:

    def test_zero_params_near_identity(self, key):
        """Zero unnormalized params => uniform bins => near-identity."""
        K = 8
        B = 3.0
        n = 50
        w = jnp.zeros((n, K))
        h = jnp.zeros((n, K))
        d = jnp.zeros((n, K - 1))
        x = jnp.linspace(-B + 0.1, B - 0.1, n)

        y, ld = rational_quadratic_spline(
            x, w, h, d, tail_bound=B, inverse=False
        )
        # With uniform bins and sigmoid(0)=0.5 for derivatives,
        # should be close to identity
        assert jnp.allclose(y, x, atol=0.3), (
            f"max deviation = {jnp.abs(y - x).max()}"
        )


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:

    def test_extreme_inputs_no_nan(self, key):
        """Very large inputs (in tail) produce finite outputs."""
        K = 8
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=K)
        x = jnp.array([-1e6, -1e3, 1e3, 1e6])

        w_b = jnp.broadcast_to(w, (4,) + w.shape)
        h_b = jnp.broadcast_to(h, (4,) + h.shape)
        d_b = jnp.broadcast_to(d, (4,) + d.shape)

        y, ld = rational_quadratic_spline(
            x, w_b, h_b, d_b, tail_bound=3.0, inverse=False
        )
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(ld))

    def test_gradients_finite(self, key):
        """Gradients through the spline are finite."""
        K = 8
        B = 3.0
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=K)
        x = jnp.array(0.5)

        def loss(x_in):
            y, ld = rational_quadratic_spline(
                x_in, w, h, d, tail_bound=B, inverse=False
            )
            return y + ld

        g = jax.grad(loss)(x)
        assert jnp.isfinite(g), f"Gradient is {g}"

    def test_gradients_wrt_params(self, key):
        """Gradients wrt spline parameters are finite."""
        K = 8
        B = 3.0
        k1, k2 = jr.split(key)
        w, h, d = _random_spline_params(k1, batch_shape=(), num_bins=K)
        x = jnp.array(0.5)

        def loss(w_in, h_in, d_in):
            y, ld = rational_quadratic_spline(
                x, w_in, h_in, d_in, tail_bound=B, inverse=False
            )
            return y + ld

        gw, gh, gd = jax.grad(loss, argnums=(0, 1, 2))(w, h, d)
        assert jnp.all(jnp.isfinite(gw))
        assert jnp.all(jnp.isfinite(gh))
        assert jnp.all(jnp.isfinite(gd))

    def test_jit_compatible(self, key):
        """Spline forward/inverse work under jit."""
        K = 8
        B = 3.0
        w, h, d = _random_spline_params(key, batch_shape=(10,), num_bins=K)
        x = jr.uniform(key, (10,), minval=-2.5, maxval=2.5)

        @jax.jit
        def fwd(x_in):
            return rational_quadratic_spline(
                x_in, w, h, d, tail_bound=B, inverse=False
            )

        @jax.jit
        def inv(y_in):
            return rational_quadratic_spline(
                y_in, w, h, d, tail_bound=B, inverse=True
            )

        y, ld_fwd = fwd(x)
        x_rec, ld_inv = inv(y)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_inputs_at_knot_positions(self, key):
        """Inputs exactly at internal knot positions don't cause issues."""
        K = 4
        B = 3.0
        w, h, d = _random_spline_params(key, batch_shape=(), num_bins=K)
        x_k, y_k, derivs = _normalize_bin_params(
            w, h, d,
            tail_bound=B, min_bin_width=1e-3,
            min_bin_height=1e-3, min_derivative=1e-3, max_derivative=10.0,
        )
        # Use internal knot positions as input
        x = x_k[1:-1]  # internal knots
        n = len(x)

        w_b = jnp.broadcast_to(w, (n,) + w.shape)
        h_b = jnp.broadcast_to(h, (n,) + h.shape)
        d_b = jnp.broadcast_to(d, (n,) + d.shape)

        y, ld = rational_quadratic_spline(
            x, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(ld))

        # Roundtrip should still work
        x_rec, _ = rational_quadratic_spline(
            y, w_b, h_b, d_b, tail_bound=B, inverse=True
        )
        assert jnp.allclose(x, x_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases for K=1
# ---------------------------------------------------------------------------

class TestSingleBin:

    def test_single_bin_roundtrip(self, key):
        """K=1 still produces valid invertible transform."""
        K = 1
        B = 3.0
        k1, k2 = jr.split(key)
        w = jr.normal(k1, (20, K))
        h = jr.normal(k2, (20, K))
        d = jnp.zeros((20, 0))  # K-1 = 0
        x = jr.uniform(key, (20,), minval=-2.5, maxval=2.5)

        y, ld_fwd = rational_quadratic_spline(
            x, w, h, d, tail_bound=B, inverse=False
        )
        x_rec, ld_inv = rational_quadratic_spline(
            y, w, h, d, tail_bound=B, inverse=True
        )
        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_single_bin_monotone(self, key):
        """K=1 transform is still monotone."""
        K = 1
        B = 3.0
        w = jr.normal(key, (K,))
        h = jr.normal(key, (K,))
        d = jnp.zeros((0,))
        n = 50
        x = jnp.linspace(-B, B, n)

        w_b = jnp.broadcast_to(w, (n,) + w.shape)
        h_b = jnp.broadcast_to(h, (n,) + h.shape)
        d_b = jnp.broadcast_to(d, (n,) + d.shape)

        y, _ = rational_quadratic_spline(
            x, w_b, h_b, d_b, tail_bound=B, inverse=False
        )
        dy = jnp.diff(y)
        assert jnp.all(dy > 0)
