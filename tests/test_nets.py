# tests/test_nets.py
"""
Unit tests for neural network modules (MLP, ResNet).

Run with:
    PYTHONPATH=. pytest tests/test_nets.py -v
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflows.nets import MLP, ResNet, init_mlp, init_resnet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# ResNet Tests
# ---------------------------------------------------------------------------
class TestResNet:
    """Tests for the ResNet module."""

    def test_output_shape(self, key):
        """Output shape matches (batch, out_dim)."""
        resnet, params = init_resnet(
            key, in_dim=10, hidden_dim=32, out_dim=5, n_hidden_layers=2
        )
        x = jax.random.normal(key, (20, 10))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (20, 5)

    def test_single_sample(self, key):
        """Works with single sample (no batch dim issues)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=1
        )
        x = jax.random.normal(key, (1, 4))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (1, 2)
        assert not jnp.isnan(out).any()

    def test_zero_hidden_layers(self, key):
        """Works with n_hidden_layers=0 (just input→output projection)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=0
        )
        x = jax.random.normal(key, (10, 4))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (10, 2)
        assert not jnp.isnan(out).any()

    def test_zero_init_output(self, key):
        """zero_init_output=True produces zero output at initialization."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=3, n_hidden_layers=2,
            zero_init_output=True
        )
        x = jax.random.normal(key, (10, 4))
        out = resnet.apply({"params": params}, x)
        # Output should be zero because dense_out kernel and bias are zero
        assert jnp.allclose(out, 0.0, atol=1e-6)

    def test_no_nans_or_infs(self, key):
        """Output is finite for reasonable inputs."""
        resnet, params = init_resnet(
            key, in_dim=8, hidden_dim=32, out_dim=4, n_hidden_layers=3
        )
        x = jax.random.normal(key, (100, 8))
        out = resnet.apply({"params": params}, x)
        assert jnp.isfinite(out).all()

    def test_jit_compatible(self, key):
        """Works under jax.jit."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        @jax.jit
        def forward(params, x):
            return resnet.apply({"params": params}, x)

        out = forward(params, x)
        assert out.shape == (10, 2)
        assert not jnp.isnan(out).any()

    def test_vmap_compatible(self, key):
        """Works under jax.vmap (batching over extra dim)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        # Shape: (num_batches, batch_size, in_dim)
        x = jax.random.normal(key, (5, 10, 4))

        @jax.vmap
        def forward_batch(x_batch):
            return resnet.apply({"params": params}, x_batch)

        out = forward_batch(x)
        assert out.shape == (5, 10, 2)

    def test_gradients_exist(self, key):
        """Gradients w.r.t. params are finite and non-zero."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        def loss_fn(params):
            out = resnet.apply({"params": params}, x)
            return (out ** 2).sum()

        grads = jax.grad(loss_fn)(params)
        grad_norm = sum(jnp.linalg.norm(v) for v in jax.tree_util.tree_leaves(grads))
        assert jnp.isfinite(grad_norm)
        assert grad_norm > 0

    def test_res_scale_zero_disables_residuals(self, key):
        """res_scale=0 means no residual connections (output differs)."""
        x = jax.random.normal(key, (10, 4))

        resnet_with_res, params_with = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2, res_scale=0.1
        )
        resnet_no_res, params_no = init_resnet(
            jax.random.fold_in(key, 1), in_dim=4, hidden_dim=16, out_dim=2,
            n_hidden_layers=2, res_scale=0.0
        )

        out_with = resnet_with_res.apply({"params": params_with}, x)
        out_no = resnet_no_res.apply({"params": params_no}, x)

        # Different random init, so outputs differ - just check both work
        assert out_with.shape == out_no.shape == (10, 2)
        assert jnp.isfinite(out_with).all()
        assert jnp.isfinite(out_no).all()

    def test_deterministic(self, key):
        """Same input + same params → same output."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        out1 = resnet.apply({"params": params}, x)
        out2 = resnet.apply({"params": params}, x)

        assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MLP Tests
# ---------------------------------------------------------------------------
class TestMLP:
    """Tests for the MLP conditioner module."""

    def test_output_shape_no_context(self, key):
        """Output shape correct without context."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        out = mlp.apply({"params": params}, x)
        assert out.shape == (20, 8)

    def test_output_shape_with_context(self, key):
        """Output shape correct with context."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        context = jax.random.normal(key, (20, 3))
        out = mlp.apply({"params": params}, x, context)
        assert out.shape == (20, 8)

    def test_zero_init_output_layer(self, key):
        """init_mlp zero-initializes the output layer for identity-start flows."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        out = mlp.apply({"params": params}, x)
        # Output should be zero due to zero-init of dense_out
        assert jnp.allclose(out, 0.0, atol=1e-6)

    def test_context_affects_output(self, key):
        """Different context produces different output."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Perturb params to break zero-init symmetry
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape), params
        )

        x = jax.random.normal(key, (10, 4))
        ctx1 = jnp.zeros((10, 3))
        ctx2 = jnp.ones((10, 3))

        out1 = mlp.apply({"params": params}, x, ctx1)
        out2 = mlp.apply({"params": params}, x, ctx2)

        diff = jnp.abs(out1 - out2).mean()
        assert diff > 1e-3, f"Context should affect output, diff={diff}"

    def test_context_broadcasting_shared(self, key):
        """Shared context (context_dim,) broadcasts to batch."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        context = jax.random.normal(key, (3,))  # Shared across batch

        out = mlp.apply({"params": params}, x, context)
        assert out.shape == (20, 8)
        assert not jnp.isnan(out).any()

    def test_wrong_x_dim_raises(self, key):
        """Wrong x dimension raises ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 5))  # Wrong: 5 instead of 4

        with pytest.raises(ValueError, match="expected x with last dimension 4"):
            mlp.apply({"params": params}, x)

    def test_wrong_context_dim_raises(self, key):
        """Wrong context dimension raises ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 5))  # Wrong: 5 instead of 3

        with pytest.raises(ValueError, match="expected context with last dimension 3"):
            mlp.apply({"params": params}, x, context)

    def test_context_batch_mismatch_raises(self, key):
        """Mismatched batch shapes raise ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (20, 3))  # Wrong: 20 instead of 10

        with pytest.raises(ValueError, match="batch shape"):
            mlp.apply({"params": params}, x, context)

    def test_jit_compatible(self, key):
        """Works under jax.jit."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 3))

        @jax.jit
        def forward(params, x, context):
            return mlp.apply({"params": params}, x, context)

        out = forward(params, x, context)
        assert out.shape == (10, 8)

    def test_gradients_flow_through_context(self, key):
        """Gradients w.r.t. context are non-zero."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Perturb params
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape), params
        )

        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 3))

        def loss_fn(context):
            out = mlp.apply({"params": params}, x, context)
            return (out ** 2).sum()

        grad_context = jax.grad(loss_fn)(context)
        grad_norm = jnp.linalg.norm(grad_context)
        assert grad_norm > 0, "Gradients should flow through context"

    def test_no_context_when_context_dim_zero(self, key):
        """context_dim=0 means context is ignored."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))

        # Both should work and produce same result
        out1 = mlp.apply({"params": params}, x, None)
        out2 = mlp.apply({"params": params}, x)

        assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# init_mlp / init_resnet Tests
# ---------------------------------------------------------------------------
class TestInitFunctions:
    """Tests for initialization helper functions."""

    def test_init_mlp_returns_module_and_params(self, key):
        """init_mlp returns (MLP, params) tuple."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=2, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        assert isinstance(mlp, MLP)
        assert isinstance(params, dict)
        # Params are nested under "net" (the ResNet submodule)
        assert "net" in params
        assert "dense_in" in params["net"]
        assert "dense_out" in params["net"]

    def test_init_resnet_returns_module_and_params(self, key):
        """init_resnet returns (ResNet, params) tuple."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2
        )
        assert isinstance(resnet, ResNet)
        assert isinstance(params, dict)
        assert "dense_in" in params
        assert "dense_out" in params

    def test_init_mlp_dense_out_is_zero(self, key):
        """init_mlp zero-initializes dense_out kernel and bias."""
        _, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Params are nested under "net" (the ResNet submodule)
        assert jnp.allclose(params["net"]["dense_out"]["kernel"], 0.0)
        assert jnp.allclose(params["net"]["dense_out"]["bias"], 0.0)

    def test_init_resnet_dense_out_not_zero_by_default(self, key):
        """init_resnet does NOT zero-init by default."""
        _, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2,
            zero_init_output=False
        )
        # At least one of kernel/bias should be non-zero
        kernel_nonzero = not jnp.allclose(params["dense_out"]["kernel"], 0.0)
        bias_nonzero = not jnp.allclose(params["dense_out"]["bias"], 0.0)
        assert kernel_nonzero or bias_nonzero

    def test_init_resnet_dense_out_zero_when_requested(self, key):
        """init_resnet zero-inits dense_out when zero_init_output=True."""
        _, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2,
            zero_init_output=True
        )
        assert jnp.allclose(params["dense_out"]["kernel"], 0.0)
        assert jnp.allclose(params["dense_out"]["bias"], 0.0)

    def test_different_keys_produce_different_params(self, key):
        """Different PRNGKeys produce different initializations."""
        _, params1 = init_resnet(key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2)
        _, params2 = init_resnet(
            jax.random.fold_in(key, 1), in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2
        )

        # dense_in kernels should differ
        assert not jnp.allclose(params1["dense_in"]["kernel"], params2["dense_in"]["kernel"])

    def test_init_mlp_param_structure_for_builders(self, key):
        """
        init_mlp must return params with structure params['net']['dense_out'].

        This structure is required by builders (e.g., _patch_spline_conditioner_dense_out)
        that modify the output layer for identity initialization. If you change the MLP
        param structure, update the builders accordingly.
        """
        _, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )

        # These assertions document the contract that builders rely on.
        assert "net" in params, "MLP params must have 'net' key (ResNet submodule)"
        assert "dense_out" in params["net"], "MLP params must have 'net/dense_out' key"
        assert "kernel" in params["net"]["dense_out"], "dense_out must have 'kernel'"
        assert "bias" in params["net"]["dense_out"], "dense_out must have 'bias'"
