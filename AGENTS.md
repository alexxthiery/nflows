# AGENTS.md

Project context for coding agents (Claude Code, Cursor, Copilot, etc.).

## Project Summary

Minimal normalizing flows library in JAX. Provides RealNVP and spline flow builders, conditional flows, identity gating, and an assembly API for custom architectures. Not a pip package; clone and import directly.

## Tech Stack

- **JAX** (core compute, JIT, vmap, autodiff)
- **Flax** (conditioner MLPs via `linen`)
- **Python 3.10+** (type unions with `|`)
- No pip package, no `__init__.py` exports

## Project Structure

```
nflows/
  __init__.py          empty
  builders.py          High-level constructors + assembly API
  flows.py             Flow and Bijection classes
  transforms.py        All transform types + CompositeTransform
  distributions.py     StandardNormal, DiagNormal
  nets.py              MLP conditioner, ResNet init
  splines.py           Rational-quadratic spline primitives
  scalar_function.py   LOFT forward/inverse scalar functions
tests/
  conftest.py          Shared fixtures + check_logdet_vs_autodiff
  test_builders.py
  test_transforms.py
  test_identity_gate.py
  test_conditional_flow.py
  test_splines.py
  test_distributions.py
  test_nets.py
```

## Module Dependency Graph

```
builders -> flows, transforms, distributions, nets
flows    -> transforms (gate), nets (types)
transforms -> nets (MLP), splines, scalar_function
nets     -> flax.linen
```

## Entry Points

- **User entry**: `build_realnvp()`, `build_spline_realnvp()` in `builders.py`
- **Low-level**: `TransformClass.create()` + `assemble_bijection()`/`assemble_flow()`
- **Core types**: `Flow`, `Bijection` in `flows.py`

## Key Patterns

- **Explicit params**: no state in objects. All params passed as PyTree dicts.
- **Transform interface**: `forward(params, x, context=None, g_value=None) -> (y, log_det)`
- **Zero-init**: conditioner output layers initialized to zero so flows start as identity.
- **Mask convention**: `mask=1` means frozen (passed through), `mask=0` means transformed. Alternating parity between layers.
- **Gate contract**: `identity_gate` callable must be written for single sample `(context_dim,)`; batching via `jax.vmap`.
- **Feature extractor split**: gate sees raw context, couplings see extracted features.

## Dev Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_builders.py -v

# Run tests matching pattern
python -m pytest tests/ -k "identity_gate" -v

# Run with float64 enabled
JAX_ENABLE_X64=1 python -m pytest tests/ -v
```

## Known Issues

Key items:

- **C1**: LOFT inverse overflows for large inputs (`scalar_function.py:71`)
- **C2**: LoftTransform + identity_gate breaks identity guarantee (LOFT not gated)
- **C3**: No gradient correctness tests (log-det vs autodiff Jacobian)
- **H2**: Inconsistent defaults between `.create()` and builders (`max_log_scale` 1.0 vs 5.0)

## Gotchas

- `identity_gate` single-sample contract: gate function receives `(context_dim,)`, not batched. `jax.vmap` handles batching. Writing a batch-aware gate silently produces wrong results.
- Raw context vs extracted: when using a feature extractor, the gate still gets raw context.
- No `__init__.py` exports: must use `from nflows.builders import build_realnvp`.
- `use_loft=True` (default) is incompatible with `identity_gate` (C2 above).
- Builder `max_log_scale` default is 5.0, but `AffineCoupling.create()` default is 1.0.

## Documentation Map

| Need | Read |
|------|------|
| Quick start, install | [README.md](README.md) |
| How to do X (examples) | [USAGE.md](USAGE.md) |
| API signatures, options tables | [REFERENCE.md](REFERENCE.md) |
| Math, design decisions | [INTERNALS.md](INTERNALS.md) |
| Adding transforms/distributions | [EXTENDING.md](EXTENDING.md) |
