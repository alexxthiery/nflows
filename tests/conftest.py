# tests/conftest.py
"""Shared pytest fixtures for nflows tests."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def key():
    """Default JAX PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dim():
    """Default feature dimension."""
    return 4


@pytest.fixture
def context_dim():
    """Default context dimension for conditional flows."""
    return 2


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 32


def check_invertibility(forward_fn, inverse_fn, x, atol=1e-5):
    """
    Helper to verify forward(inverse(x)) ≈ x and log_det consistency.

    Returns dict with errors for assertions.
    """
    z, ld_inv = inverse_fn(x)
    x_rec, ld_fwd = forward_fn(z)

    reconstruction_error = jnp.abs(x - x_rec).max()
    logdet_error = jnp.abs(ld_fwd + ld_inv).max()

    return {
        "reconstruction_error": float(reconstruction_error),
        "logdet_error": float(logdet_error),
        "z": z,
        "x_rec": x_rec,
    }


def check_logdet_vs_autodiff(forward_fn, x, atol=1e-4):
    """
    Compare log_det from forward pass against autodiff Jacobian.

    Works for single sample (no batch dimension).
    """
    y, ld = forward_fn(x)

    # Compute Jacobian via autodiff
    J = jax.jacfwd(lambda z: forward_fn(z)[0])(x)
    ld_autodiff = jnp.log(jnp.abs(jnp.linalg.det(J)))

    error = float(jnp.abs(ld - ld_autodiff))
    return {
        "error": error,
        "ld": float(ld),
        "ld_autodiff": float(ld_autodiff),
    }
