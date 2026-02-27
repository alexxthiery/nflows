# nflows

A minimal, hackable normalizing flows library in JAX.

## Overview

- **RealNVP** with affine coupling layers
- **Spline flows** with rational-quadratic splines [durkan2019]
- **Conditional flows** with context concatenation and optional feature extraction
- **Identity gating** for smooth boundary conditions on context-dependent flows

## Installation

```bash
git clone <repo-url>
cd nflows
pip install -e .          # installs jax, jaxlib, flax
pip install -e ".[test]"  # also installs pytest
```

## Quick Start

```python
import jax
from nflows.builders import build_realnvp

key = jax.random.PRNGKey(0)
flow, params = build_realnvp(
    key, dim=4, num_layers=4, hidden_dim=64, n_hidden_layers=2,
)

samples = flow.sample(params, key, shape=(1000,))                      # (1000, 4)
log_prob = flow.log_prob(params, samples)                               # (1000,)
samples, log_prob = flow.sample_and_log_prob(params, key, shape=(1000,))
```

## Features

- [Affine flow (RealNVP)](USAGE.md#affine-flow-realnvp) with affine coupling layers
- [Spline flows](USAGE.md#spline-flows) for more expressive transforms
- [Conditional flows](USAGE.md#conditional-flows) with context variables
- [Feature extractor](USAGE.md#feature-extractor) for high-dimensional context
- [Transform-only mode](USAGE.md#transform-only-mode-bijection) (bijection without base distribution)
- [Identity gating](USAGE.md#identity-gating) for boundary conditions
- [Custom architectures](USAGE.md#custom-architectures-assembly-api) via assembly API
- [Training recipes](USAGE.md#training) for forward KL (MLE) and reverse KL (VI)

## Design

- **Explicit parameters**: no hidden state; params are always passed as PyTrees
- **JAX-native**: JIT-compatible, vmap-friendly, functional style
- **Zero-init conditioners**: flows start as identity for stable training
- **LOFT stabilization**: logarithmic tails prevent numerical issues in high dimensions

## Documentation

| Document | Content |
|----------|---------|
| [USAGE.md](USAGE.md) | How-to cookbook with copy-pasteable examples |
| [REFERENCE.md](REFERENCE.md) | API reference: classes, builders, options, param structure |
| [INTERNALS.md](INTERNALS.md) | Math foundations and design decisions |
| [EXTENDING.md](EXTENDING.md) | Recipes for custom transforms, distributions, conditioners |
| [AGENTS.md](AGENTS.md) | Project context for coding agents |

## References

- Dinh et al. (2017). "Density estimation using Real-NVP"
- Durkan et al. (2019). "Neural Spline Flows"
- Andrade (2021). "Stable Training of Normalizing Flows for High-dimensional Variational Inference"
