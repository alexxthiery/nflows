# nflows/nets.py
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jnp.ndarray
PRNGKey = jax.Array  # type alias for JAX random keys


def loft(z: Array, tau: float) -> Array:
    """
    LOFT transform.

    g(z) = sign(z) * [ min(|z|, tau) + log( max(|z| - tau, 0) + 1 ) ]

    Args:
        z: Input array (any shape).
        tau: Positive threshold (float). Controls where linear part transitions
             to logarithmic tails.

    Returns:
        Array with same shape as z.
        
    Reference:
    "STABLE TRAINING OF NORMALIZING FLOWS FOR HIGH-DIMENSIONAL VARIATIONAL INFERENCE" by DANIEL ANDRADE
    """
    # Promote tau to array with same dtype for stable broadcasting under JIT.
    tau = jnp.asarray(tau, dtype=z.dtype)

    abs_z = jnp.abs(z)
    sign_z = jnp.sign(z)

    # Linear core: min(|z|, tau)
    core = jnp.minimum(abs_z, tau)

    # Log tails: log1p(max(|z| - tau, 0))
    tail = jnp.log1p(jnp.maximum(abs_z - tau, 0.0))

    return sign_z * (core + tail)


def loft_inv(y: Array, tau: float) -> Array:
    """
    Inverse LOFT transform.

    g^{-1}(y) = sign(y) * [ min(|y|, tau) + exp( max(|y| - tau, 0) ) - 1 ]

    Args:
        y: Transformed array (any shape).
        tau: Same threshold used in loft.

    Returns:
        Array with same shape as y, satisfying loft(loft_inv(y)) ≈ y.
        
    Reference:
    "STABLE TRAINING OF NORMALIZING FLOWS FOR HIGH-DIMENSIONAL VARIATIONAL INFERENCE" by DANIEL ANDRADE
    """
    tau = jnp.asarray(tau, dtype=y.dtype)

    abs_y = jnp.abs(y)
    sign_y = jnp.sign(y)

    core = jnp.minimum(abs_y, tau)
    tail = jnp.maximum(abs_y - tau, 0.0)

    # Clamp tail to prevent float32 overflow in expm1 (overflows at ~88.7).
    tail = jnp.minimum(tail, 80.0)

    # expm1 for numerical stability when tail is small
    return sign_y * (core + jnp.expm1(tail))


def loft_log_abs_det_jac(
        z: Array,
        tau: float,
        ) -> Array:
    """
    Log |det J_g(z)| of LOFT, useful for flows.

    For each element:
        g'(z) = 1 / (max(|z| - tau, 0) + 1)
        log |g'(z)| = -log( max(|z| - tau, 0) + 1 )

    Args:
        z: Input array.
        tau: Same threshold used in loft.

    Returns:
        log |det J| per element (same shape as z)
    """
    tau = jnp.asarray(tau, dtype=z.dtype)

    abs_z = jnp.abs(z)
    # log( max(|z| - tau, 0) + 1 )
    log_term = jnp.log1p(jnp.maximum(abs_z - tau, 0.0))

    log_jac = -log_term

    return log_jac
