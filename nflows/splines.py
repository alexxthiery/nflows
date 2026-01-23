"""
Core implementation of monotonic rational-quadratic splines (Durkan et al., 2019).

This module provides the scalar forward and inverse transformations used in
Rational-Quadratic Coupling layers. Each call operates on a single scalar
dimension (with arbitrary batch shape), and is intended to be vmapped over
multiple dimensions inside a flow layer.

Key properties:
  - Strictly monotonic, continuously differentiable, invertible.
  - Closed-form forward map, inverse map, and log-Jacobian determinant.
  - Spline is defined on [-B, B]; outside this interval the transform is linear
    with derivative 1, ensuring global invertibility and stable tails.
  - Parameters (bin widths, heights, derivatives) are produced by a conditioner
    network and normalized here to satisfy required constraints:
        * widths/heights ≥ eps, sum to 2B,
        * derivatives > 0, boundary derivatives fixed to 1.

Not included:
  - No batching over feature dimensions (use jax.vmap at the coupling layer).
  - No coupling logic, masking, or network definitions.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn

Array = jnp.ndarray


def _normalize_bin_params(
    unnormalized_widths: Array,
    unnormalized_heights: Array,
    unnormalized_derivatives: Array,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
    max_derivative: float,
) -> Tuple[Array, Array, Array]:
    """
    Convert raw NN outputs into valid spline parameters.

    Inputs:
        unnormalized_widths:    (..., K)
        unnormalized_heights:   (..., K)
        unnormalized_derivatives: (..., K-1)  internal knot derivatives
        tail_bound:             scalar B
        min_bin_width:          lower bound for each bin width (fraction of total)
        min_bin_height:         lower bound for each bin height (fraction of total)
        min_derivative:         lower bound for derivatives at internal knots
        max_derivative:         upper bound for derivatives at internal knots
        
    Outputs:
      x_k:        (..., K+1)  x-knots in [-B, B], strictly increasing
      y_k:        (..., K+1)  y-knots in [-B, B], strictly increasing
      derivatives:(..., K+1)  derivatives at all knots, positive, with
                              boundary derivatives fixed to 1 (for smooth tails)
    """
    num_bins = unnormalized_widths.shape[-1]
    
    # Sanity checks: min widths/heights must be feasible.
    if min_bin_width * num_bins >= 1.0:
        raise ValueError(
            f"min_bin_width * num_bins must be < 1, got {min_bin_width * num_bins}."
        )
    if min_bin_height * num_bins >= 1.0:
        raise ValueError(
            f"min_bin_height * num_bins must be < 1, got {min_bin_height * num_bins}."
        )
    
    if num_bins < 1:
        raise ValueError(
            f"_normalize_bin_params: expected at least 1 bin, got {num_bins}."
        )

    # Normalize widths: softmax + minimum width per bin, scaled to cover [-B, B].
    widths = jnn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    widths = widths * (2.0 * tail_bound)

    # Normalize heights analogously.
    heights = jnn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    heights = heights * (2.0 * tail_bound)

    # Cumulative sums to get knot positions in [-B, B].
    # Pad with 0, then shift by -B to get correct positions.
    x_k = jnp.cumsum(widths, axis=-1)  # (..., K)
    x_k = jnp.pad(
        x_k,
        pad_width=[(0, 0)] * (x_k.ndim - 1) + [(1, 0)],
        constant_values=0.0,
    )  # (..., K+1) with leading 0
    x_k = x_k - tail_bound  # shift to start at -B, end at +B
    # Force exact boundary values to avoid floating-point drift
    x_k = x_k.at[..., 0].set(-tail_bound)
    x_k = x_k.at[..., -1].set(tail_bound)

    y_k = jnp.cumsum(heights, axis=-1)  # (..., K)
    y_k = jnp.pad(
        y_k,
        pad_width=[(0, 0)] * (y_k.ndim - 1) + [(1, 0)],
        constant_values=0.0,
    )
    y_k = y_k - tail_bound
    # Force exact boundary values to avoid floating-point drift
    y_k = y_k.at[..., 0].set(-tail_bound)
    y_k = y_k.at[..., -1].set(tail_bound)

    # Derivatives at internal knots: positive via softplus + epsilon.
    # Boundary derivatives are set to 1 to match identity tails.
    if unnormalized_derivatives.shape[-1] != num_bins - 1:
        raise ValueError(
            "_normalize_bin_params: expected unnormalized_derivatives last dim "
            f"{num_bins - 1}, got {unnormalized_derivatives.shape[-1]}."
        )

    # internal_derivatives = min_derivative + jnn.softplus(unnormalized_derivatives)
    
    # Use sigmoid to also enforce an lower/upper bound on derivatives for stability.
    internal_derivatives = (
        min_derivative
        + (max_derivative - min_derivative)
        * jnn.sigmoid(unnormalized_derivatives)
        )
    
    ones = jnp.ones_like(internal_derivatives[..., :1])
    derivatives = jnp.concatenate(
        [ones, internal_derivatives, ones], axis=-1
    )  # (..., K+1)

    return x_k, y_k, derivatives


def _select_bins(x: Array, x_k: Array) -> Tuple[Array, Array]:
    """
    For each x, select the bin index k such that x_k[..., k] <= x < x_k[..., k+1].

    Inputs:
      x:    (...,)
      x_k:  (..., K+1), increasing along the last axis.

    Returns:
      bin_idx: (...,) integer indices in [0, K-1]
      num_bins: int, number of bins K
    """
    num_bins = x_k.shape[-1] - 1
    if num_bins < 1:
        raise ValueError(
            f"_select_bins: expected at least 1 bin, got {num_bins}."
        )

    # Compare each x to the left edges of the bins.
    # mask_bins[..., j] = True if x >= x_k[..., j]
    x_expanded = jnp.expand_dims(x, axis=-1)  # (..., 1)
    left_edges = x_k[..., :-1]               # (..., K)
    mask_bins = x_expanded >= left_edges

    # Count how many left edges are <= x, subtract 1 to get bin index.
    bin_idx = jnp.sum(mask_bins, axis=-1) - 1  # (...,)

    # Clamp to valid range in case of numerical issues.
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    return bin_idx, num_bins


def _gather_bin_params(
    x_k: Array,
    y_k: Array,
    derivatives: Array,
    bin_idx: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Gather per-bin parameters for each x given its bin index.

    Inputs:
      x_k:        (..., K+1)
      y_k:        (..., K+1)
      derivatives:(..., K+1)
      bin_idx:    (...,)

    Returns:
      x_left:   (...,)
      x_right:  (...,)
      y_left:   (...,)
      y_right:  (...,)
      d_left:   (...,)
      d_right:  (...,)
    """
    # For K bins, we have K+1 knots. Slice to get left/right edges of each bin.
    # *_all arrays have shape (..., K), indexed by bin number.
    x_left_all = x_k[..., :-1]
    x_right_all = x_k[..., 1:]
    y_left_all = y_k[..., :-1]
    y_right_all = y_k[..., 1:]
    d_left_all = derivatives[..., :-1]
    d_right_all = derivatives[..., 1:]

    # Use take_along_axis to gather the bin-specific values for each sample.
    # idx has shape (..., 1) so the gather produces (..., 1), then we squeeze.
    idx = jnp.expand_dims(bin_idx, axis=-1)  # (..., 1)

    def gather(a: Array) -> Array:
        return jnp.take_along_axis(a, idx, axis=-1)[..., 0]

    x_left = gather(x_left_all)
    x_right = gather(x_right_all)
    y_left = gather(y_left_all)
    y_right = gather(y_right_all)
    d_left = gather(d_left_all)
    d_right = gather(d_right_all)

    return x_left, x_right, y_left, y_right, d_left, d_right


def _rational_quadratic_forward_inner(
    x: Array,
    x_k: Array,
    y_k: Array,
    derivatives: Array,
    tail_bound: float,
) -> Tuple[Array, Array]:
    """
    Core forward pass of the rational-quadratic spline on [-B, B].

    Inputs:
      x:          (...,)
      x_k:        (..., K+1)
      y_k:        (..., K+1)
      derivatives:(..., K+1)
      tail_bound: scalar B

    Returns (on the full real line):
      y:          (...,)
      log_dydx:   (...,)
    """
    B = float(tail_bound)
    eps = 1e-12

    # Masks for tails and inside region.
    inside_mask = (x >= -B) & (x <= B)
    x_inside = jnp.where(inside_mask, x, 0.0)

    # Select bin indices for inside points.
    bin_idx, num_bins = _select_bins(x_inside, x_k)

    # Gather per-bin parameters.
    (
        x_left,
        x_right,
        y_left,
        y_right,
        d_left,
        d_right,
    ) = _gather_bin_params(x_k, y_k, derivatives, bin_idx)

    # Bin widths and heights.
    dx = x_right - x_left
    dy = y_right - y_left

    # Average slope in the bin.
    s = dy / dx  # (...,)

    # Normalized position within the bin.
    xi = (x_inside - x_left) / dx
    xi = jnp.clip(xi, 0.0, 1.0)
    one_minus_xi = 1.0 - xi

    # Rational-quadratic spline as in Durkan et al.
    # y = y_left + (dy * (s * xi^2 + d_left * xi * (1 - xi))) /
    #                 (s + (d_right + d_left - 2s) * xi * (1 - xi))
    numerator = dy * (s * xi * xi + d_left * xi * one_minus_xi)
    denom = s + (d_right + d_left - 2.0 * s) * xi * one_minus_xi
    denom = jnp.maximum(denom, eps)

    y_inside = y_left + numerator / denom

    # Derivative dy/dx, from eq. (5) in the paper.
    # d y / d x = ( s^2 * ( d_right * xi^2
    #                      + 2 s xi (1 - xi)
    #                      + d_left (1 - xi)^2 )
    #              / ( s + (d_right + d_left - 2 s) xi (1 - xi) )^2 )
    derivative_numer = (
        s * s
        * (
            d_right * xi * xi
            + 2.0 * s * xi * one_minus_xi
            + d_left * one_minus_xi * one_minus_xi
        )
    )
    derivative_denom = denom * denom
    dy_dx_inside = derivative_numer / jnp.maximum(derivative_denom, eps)
    dy_dx_inside = jnp.maximum(dy_dx_inside, eps)

    # Combine tails and inside.
    # Outside [-B, B], the transform is identity with derivative 1.
    y = jnp.where(inside_mask, y_inside, x)
    log_dydx = jnp.where(inside_mask, jnp.log(dy_dx_inside), jnp.zeros_like(x))

    return y, log_dydx


def _rational_quadratic_inverse_inner(
    y: Array,
    x_k: Array,
    y_k: Array,
    derivatives: Array,
    tail_bound: float,
) -> Tuple[Array, Array]:
    """
    Core inverse of the rational-quadratic spline on [-B, B].

    Inputs:
      y:          (...,)
      x_k:        (..., K+1)
      y_k:        (..., K+1)
      derivatives:(..., K+1)
      tail_bound: scalar B

    Returns (on the full real line):
      x:          (...,)
      log_dxdY:   (...,)  where log_dxdY = log |dx/dy|
    """
    B = float(tail_bound)
    eps = 1e-12

    inside_mask = (y >= -B) & (y <= B)
    y_inside = jnp.where(inside_mask, y, 0.0)

    # Select bin indices based on y_k (since y is in output space).
    bin_idx, num_bins = _select_bins(y_inside, y_k)

    (
        x_left,
        x_right,
        y_left,
        y_right,
        d_left,
        d_right,
    ) = _gather_bin_params(x_k, y_k, derivatives, bin_idx)

    dx = x_right - x_left
    dy = y_right - y_left
    s = dy / dx

    # We solve for xi in [0, 1] such that the rational-quadratic expression equals y.
    # Using Durkan et al.'s equations (6)-(8):
    #
    # Let t = s, d0 = d_left, d1 = d_right, h = dy, u = y_inside - y_left.
    #
    # a = h * (t - d0) + u * (d1 + d0 - 2 t)
    # b = h * d0        - u * (d1 + d0 - 2 t)
    # c = -t * u
    #
    # Then xi is the root in (0, 1) of a xi^2 + b xi + c = 0.

    u = y_inside - y_left
    t = s
    d0 = d_left
    d1 = d_right
    h = dy

    a = h * (t - d0) + u * (d1 + d0 - 2.0 * t)
    b = h * d0 - u * (d1 + d0 - 2.0 * t)
    c = -t * u

    # Quadratic formula: xi = (-b + sqrt(b² - 4ac)) / (2a)
    # Stable form: xi = 2c / (-b - sqrt(b² - 4ac))
    # This always computes the root in [0, 1] for a valid monotonic spline.
    #
    # Fix #3: Clamp discriminant to 0 (not eps), add eps inside sqrt.
    disc = b * b - 4.0 * a * c
    disc = jnp.maximum(disc, 0.0)
    sqrt_disc = jnp.sqrt(disc + eps)

    # Fix #2: Ensure denominator is never zero.
    # The formula 2c / (-b - sqrt_disc) is equivalent to (-b + sqrt_disc) / (2a).
    denom = -b - sqrt_disc
    denom = jnp.where(jnp.abs(denom) < eps, eps, denom)

    xi = 2.0 * c / denom
    xi = jnp.clip(xi, 0.0, 1.0)
    x_inside = x_left + xi * dx

    # Compute dy/dx at the recovered x (same formula as in forward), then invert.
    one_minus_xi = 1.0 - xi
    denom_forward = t + (d1 + d0 - 2.0 * t) * xi * one_minus_xi
    denom_forward = jnp.maximum(denom_forward, eps)

    derivative_numer = (
        t * t
        * (
            d1 * xi * xi
            + 2.0 * t * xi * one_minus_xi
            + d0 * one_minus_xi * one_minus_xi
        )
    )
    derivative_denom = denom_forward * denom_forward
    dy_dx_inside = derivative_numer / jnp.maximum(derivative_denom, eps)
    dy_dx_inside = jnp.maximum(dy_dx_inside, eps)

    log_dy_dx_inside = jnp.log(dy_dx_inside)
    log_dxdY_inside = -log_dy_dx_inside

    x = jnp.where(inside_mask, x_inside, y)
    log_dxdY = jnp.where(inside_mask, log_dxdY_inside, jnp.zeros_like(y))

    return x, log_dxdY


def rational_quadratic_spline(
    inputs: Array,
    unnormalized_widths: Array,
    unnormalized_heights: Array,
    unnormalized_derivatives: Array,
    tail_bound: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    max_derivative: float = 10.0,
    inverse: bool = False,
) -> Tuple[Array, Array]:
    """
    Monotonic rational-quadratic spline transform (Durkan et al., 2019).

    This function acts elementwise on `inputs` and is intended to be used
    inside coupling or autoregressive layers. It operates on a *single*
    scalar dimension per call, with arbitrary batch shape.

    Inputs:
      inputs:                 (...,)  values to transform
      unnormalized_widths:    (..., K)
      unnormalized_heights:   (..., K)
      unnormalized_derivatives:(..., K-1)
      tail_bound:             scalar B; spline acts on [-B, B], outside is linear
      min_bin_width:          lower bound for each bin width (fraction of total)
      min_bin_height:         lower bound for each bin height (fraction of total)
      min_derivative:         lower bound for derivatives at internal knots
      inverse:                if True, apply inverse transform

    Returns:
      outputs:                (...,)
      logabsdet:              (...,) log |d outputs / d inputs|
    """
    x_k, y_k, derivatives = _normalize_bin_params(
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        max_derivative=max_derivative,
    )

    if not inverse:
        return _rational_quadratic_forward_inner(
            inputs,
            x_k=x_k,
            y_k=y_k,
            derivatives=derivatives,
            tail_bound=tail_bound,
        )
    else:
        return _rational_quadratic_inverse_inner(
            inputs,
            x_k=x_k,
            y_k=y_k,
            derivatives=derivatives,
            tail_bound=tail_bound,
        )