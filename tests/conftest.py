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
