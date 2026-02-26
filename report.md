# nflows Deep Analysis Report

Analysis of the nflows JAX normalizing flows library.
253 tests, all passing. 7 source modules, ~2400 lines of code.

## Executive Summary

The library is well-structured with good test coverage for the happy path.
Three critical issues and several high-severity findings emerged across numerical stability, API consistency, and test coverage.

| Severity | Count |
|----------|-------|
| Critical | 3     |
| High     | 7     |
| Medium   | 7     |
| Low      | 6     |

## Critical Issues

### C1. LOFT inverse overflows for large inputs

**File:** `scalar_function.py:71`
**Category:** Numerical stability

`loft_inv` uses `jnp.expm1(tail)` where `tail = max(abs_y - tau, 0.0)`.
With default `tau=1000`, any `|y| > 1089` causes float32 overflow (`expm1` overflows at ~88.7).
Even moderate tails (`|y| = 1100`) produce `inf`.

**Impact:** Silent `inf`/`nan` corruption in log-probabilities and gradients for heavy-tailed distributions.

**Fix:** Clamp tail before expm1: `tail = jnp.minimum(tail, 80.0)` for float32.

### C2. LoftTransform + identity_gate breaks identity guarantee

**File:** `transforms.py:1489`
**Category:** Correctness

`LoftTransform` does not support `g_value`. `CompositeTransform` skips passing `g_value` to unsupported blocks.
When `identity_gate` is active and `gate=0`, couplings and `LinearTransform` become identity, but `LoftTransform` still applies its full transformation.
The flow is NOT identity at `gate=0` when `use_loft=True`.

All identity gate tests explicitly use `use_loft=False`, confirming the bug is known but unresolved.

**Impact:** The identity gate's core guarantee is violated. Users combining `identity_gate` with `use_loft=True` (the default) get incorrect behavior.

**Fix:** Either add `g_value` support to `LoftTransform` (interpolate toward identity when `g=0`), or raise an error when `identity_gate` is set with `use_loft=True`.

### C3. No gradient correctness tests (log-det vs autodiff Jacobian)

**File:** `tests/conftest.py:54`
**Category:** Test coverage

`check_logdet_vs_autodiff` exists in conftest.py but is never called by any test.
No test verifies that hand-derived log-det formulas match the true Jacobian from autodiff.

**Impact:** The fundamental correctness guarantee of a normalizing flow (correct change-of-variables) is unverified. A wrong sign, missing term, or summation error in any log-det would silently produce wrong densities.

**Fix:** Add log-det vs autodiff checks for each transform type (affine, spline, linear, LOFT, composite).

## High Severity Issues

### H1. float32 log-det accumulation loses precision in deep flows

**File:** `transforms.py:1555`
**Category:** Numerical stability

`CompositeTransform` accumulates log-det in float32 by default (float64 only if `jax_enable_x64` is set, which is off by default).
With 20 coupling layers each contributing log-det $\sim 50$, the accumulated value $\sim 1000$ has only ~4 digits of precision in float32.
This directly degrades gradient quality.

**Fix:** Use Kahan compensated summation, or cast the log-det accumulator to float64 locally.

### H2. Inconsistent defaults between `.create()` and builders

**Category:** API inconsistency

| Parameter | `.create()` | Builder |
|-----------|-------------|---------|
| `max_log_scale` | 1.0 | 5.0 |
| `min_bin_width` | 1e-3 | 1e-2 |
| `min_bin_height` | 1e-3 | 1e-2 |

Users building flows manually via `.create()` get much more restrictive defaults than builder users. Training dynamics differ without any warning.

**Fix:** Align defaults across both APIs.

### H3. identity_gate sees raw context; couplings see extracted features

**File:** `flows.py:81`
**Category:** Correctness

The feature extractor transforms raw context before couplings see it, but `identity_gate` receives raw context.
The gate function and coupling networks operate on different representations of the same input.

**Fix:** Document the design choice, or add an option to let the gate function receive extracted features.

### H4. `_compute_gate_value` silently breaks batched gate functions

**File:** `transforms.py:42`
**Category:** Gotcha

Uses `jax.vmap(identity_gate)(context)` for batched input, meaning the gate function must be written for single samples.
A user who writes a gate function operating on batched arrays gets silently wrong results.

**Fix:** Document the single-sample contract prominently. Consider detecting batch-aware gate functions.

### H5. `assemble_bijection` cannot set identity_gate

**File:** `builders.py:312`
**Category:** API inconsistency

The assembly API never sets `identity_gate` on the resulting `Bijection`.
Users who use the lower-level assembly API lose access to identity gating.

**Fix:** Add `identity_gate` parameter to `assemble_bijection` and `assemble_flow`.

### H6. No direct tests for splines.py

**Category:** Test coverage

The rational quadratic spline is the core numerical primitive for spline flows.
No test exercises it directly; all coverage is indirect through `SplineCoupling`.
Boundary behavior, bin edge cases, and numerical precision are untested in isolation.

**Fix:** Add direct tests for `rational_quadratic_spline`: forward/inverse consistency, boundary values at $\pm$ tail_bound, single-bin case, extreme parameter values.

### H7. No dim=1 tests

**Category:** Test coverage

Single-dimension flows are a valid degenerate case. Coupling layers split dimensions into two groups; with `dim=1`, one group is empty.
Alternating masks, permutations, and conditioner input dimensions may all behave unexpectedly.

**Fix:** Parameterize existing tests with `dim=1` to verify edge case handling.

## Medium Severity Issues

### M1. Spline inverse discriminant precision

**File:** `splines.py:376`

The discriminant clamp (`maximum(disc, 0.0)`) followed by `sqrt(disc + eps)` with `eps=1e-12` introduces O(1e-6) error in `xi` when the discriminant is near zero.
For narrow bins (min_bin_width=1e-2), this can push forward-inverse round-trip error above 1e-5 in float32.

### M2. LinearTransform unbounded softplus diagonal

**File:** `transforms.py:180`

`softplus(raw_diag)` has no upper bound. Large `raw_diag` values produce unbounded scale factors, causing output explosions during training.
Unlike affine coupling (which uses tanh clamping), there is no safety rail.

### M3. Activation mismatch: tanh vs elu

Feature extractor defaults to `tanh`; MLP defaults to `elu`.
Flows built with the builder vs manually have different gradient dynamics.
`tanh` saturates for large inputs, potentially slowing feature extractor training.

### M4. MLP silently drops context when context_dim=0

**File:** `nets.py:198`

If `context_dim=0` and context is passed, the MLP silently ignores it.
Users who forget to set `context_dim` get an unconditional flow that appears to work but ignores conditioning.

### M5. Mutable dataclasses

All transforms use `@dataclass` (not frozen). `__post_init__` mutates attributes.
Sharing transform objects between layers can produce aliasing bugs.

### M6. No `__init__.py` exports

Users must use fully qualified imports (`from nflows.builders import build_realnvp`).
No top-level API surface for discoverability.

### M7. No pip-installable package

No `pyproject.toml`, `setup.py`, or `setup.cfg`. Cannot be installed as a dependency.

## Low Severity Issues

### L1. `Flow.sample` discards second PRNG key

`key_base, _ = jax.random.split(key)` wastes one key. Minor inefficiency.

### L2. Shift/scale coupling in AffineCoupling

Default `max_shift = exp(max_log_scale)` couples shift and scale bounds. Unintuitive but harmless.

### L3. No vmap compatibility tests for flows

Only ResNet has vmap tests. Flow-level vmap is untested.

### L4. No training loop smoke test

No test verifies that a flow can actually be trained (loss decreases over a few steps).

### L5. Key variable reuse in builders

`key` is reused after `jax.random.split(key)`. Correct but confusing for readers.

### L6. No serialization round-trip test

Saving and loading params is untested. Low risk with standard JAX pytrees.

## Recommendations (Priority Order)

1. Fix C2 (LoftTransform + identity_gate): raise error or add g_value support
2. Fix C1 (LOFT overflow): add expm1 clamping
3. Add C3 (gradient correctness tests): wire up existing `check_logdet_vs_autodiff`
4. Align H2 (default values) across `.create()` and builders
5. Add H6 (direct spline tests) and H7 (dim=1 tests)
6. Address H1 (log-det precision) with compensated summation
7. Add `identity_gate` to assembly API (H5)
8. Document H3 (context representation split) and H4 (gate function contract)
