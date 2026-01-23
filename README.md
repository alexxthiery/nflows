# nflows

A minimal, hackable implementation of normalizing flows in JAX.

## Overview

Normalizing flows learn invertible transformations between a simple base distribution (e.g., Gaussian) and complex target distributions. This library provides:

- **RealNVP** with affine coupling layers
- **Spline flows** with rational-quadratic splines (Durkan et al., 2019)
- **Conditional flows** with context concatenation
- **LOFT stabilization** for high-dimensional settings

## Installation

```bash
pip install jax flax
# Clone and use directly
git clone <repo-url>
cd nflows
```

## Quick Start

### Unconditional Flow

```python
import jax
from nflows.builders import build_realnvp

# Build a RealNVP flow
key = jax.random.PRNGKey(0)
flow, params = build_realnvp(
    key,
    dim=4,              # Data dimension
    num_layers=4,       # Number of coupling layers
    hidden_dim=64,      # Conditioner MLP hidden width
    n_hidden_layers=2,  # Number of residual blocks
)

# Sample from the flow
samples = flow.sample(params, key, shape=(1000,))  # (1000, 4)

# Evaluate log-density
log_prob = flow.log_prob(params, samples)  # (1000,)

# Efficient sample + log_prob in one pass
samples, log_prob = flow.sample_and_log_prob(params, key, shape=(1000,))
```

### Conditional Flow

Model conditional densities `q(x | context)`:

```python
from nflows.builders import build_realnvp

# Build conditional flow
flow, params = build_realnvp(
    key,
    dim=4,
    num_layers=4,
    hidden_dim=64,
    n_hidden_layers=2,
    context_dim=2,  # Conditioning variable dimension
)

# Context can be per-sample or shared
context = jax.random.normal(key, (1000, 2))  # Per-sample context

# All methods accept context
samples = flow.sample(params, key, (1000,), context=context)
log_prob = flow.log_prob(params, samples, context=context)
```

### Spline Flows

More expressive than affine coupling:

```python
from nflows.builders import build_spline_realnvp

flow, params = build_spline_realnvp(
    key,
    dim=4,
    num_layers=4,
    hidden_dim=64,
    n_hidden_layers=2,
    num_bins=8,         # Spline bins per dimension
    tail_bound=5.0,     # Linear tails outside [-B, B]
)
```

### Transform-Only Mode (Bijection)

When you only need the invertible transform without a base distribution:

```python
from nflows.builders import build_realnvp

# Get just the bijection (no base distribution)
bijection, params = build_realnvp(
    key,
    dim=4,
    num_layers=4,
    hidden_dim=64,
    n_hidden_layers=2,
    context_dim=2,              # Works with context
    return_transform_only=True,  # Return Bijection instead of Flow
)

# Forward/inverse with tractable Jacobian
x = jax.random.normal(key, (100, 4))
context = jax.random.normal(key, (100, 2))

y, log_det = bijection.forward(params, x, context=context)
x_rec, _   = bijection.inverse(params, y, context=context)
```

Use cases: change of variables, learned coordinate transforms, custom base distributions.

### Custom Architectures (Assembly API)

For non-standard architectures (mixing coupling types, custom layer order):

```python
from nflows.builders import make_alternating_mask, assemble_bijection
from nflows.transforms import AffineCoupling, SplineCoupling, LoftTransform

keys = jax.random.split(key, 4)
mask0 = make_alternating_mask(dim=4, parity=0)
mask1 = make_alternating_mask(dim=4, parity=1)

# Mix affine and spline couplings
blocks_and_params = [
    AffineCoupling.create(keys[0], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2),
    AffineCoupling.create(keys[1], dim=4, mask=mask1, hidden_dim=64, n_hidden_layers=2),
    SplineCoupling.create(keys[2], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2, num_bins=8),
    LoftTransform.create(keys[3], dim=4),
]

bijection, params = assemble_bijection(blocks_and_params)
```

### Custom Architecture with Context and Feature Extractor

```python
from nflows.builders import make_alternating_mask, create_feature_extractor, assemble_bijection
from nflows.transforms import AffineCoupling, SplineCoupling, LoftTransform

keys = jax.random.split(key, 5)
dim = 4
raw_context_dim = 16      # raw context input dimension
effective_context_dim = 8  # learned representation dimension

# Create feature extractor: transforms raw context before coupling layers
fe, fe_params = create_feature_extractor(
    keys[0], in_dim=raw_context_dim, hidden_dim=32, out_dim=effective_context_dim
)

# Couplings receive effective_context_dim (output of feature extractor)
mask0 = make_alternating_mask(dim, parity=0)
mask1 = make_alternating_mask(dim, parity=1)

blocks_and_params = [
    AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2,
                          context_dim=effective_context_dim),
    AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=64, n_hidden_layers=2,
                          context_dim=effective_context_dim),
    SplineCoupling.create(keys[3], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2,
                          num_bins=8, context_dim=effective_context_dim),
    LoftTransform.create(keys[4], dim=dim),
]

# Assemble with feature extractor
bijection, params = assemble_bijection(
    blocks_and_params,
    feature_extractor=fe,
    feature_extractor_params=fe_params,
)

# Usage: pass raw context — feature extractor transforms it internally
x = jax.random.normal(key, (100, dim))
raw_context = jax.random.normal(key, (100, raw_context_dim))

y, log_det = bijection.forward(params, x, context=raw_context)
x_rec, _ = bijection.inverse(params, y, context=raw_context)
```

### Identity Gating (Time-Dependent Flows)

Smoothly interpolate between identity and learned transform based on context.
The gate function maps context to a scalar:
- **gate = 0**: transform is identity (x → x, log_det = 0)
- **gate = 1**: transform acts normally
- **0 < gate < 1**: interpolates between identity and learned transform

```python
import jax.numpy as jnp
from nflows.builders import build_realnvp

# Gate = t*(1-t): returns 0 at t=0 and t=1, peaks at t=0.5
gate_fn = lambda ctx: ctx[0] * (1 - ctx[0])

flow, params = build_realnvp(
    key, dim=4, num_layers=4, hidden_dim=64, n_hidden_layers=2,
    context_dim=1,
    identity_gate=gate_fn,
)

# At t=0 or t=1: gate=0 → F(x) = x (identity)
# At t=0.5: gate=0.25 → partial interpolation toward learned transform
```

Use cases: time-dependent flows, boundary conditions, curriculum learning.

## Architecture

### Core Components

| Module | Description |
|--------|-------------|
| `flows.py` | `Flow` class with `sample`, `log_prob`, `forward`, `inverse`; `Bijection` for transform-only use |
| `transforms.py` | Invertible transforms: `AffineCoupling`, `SplineCoupling`, `LinearTransform`, `Permutation`, `LoftTransform`, `CompositeTransform` |
| `distributions.py` | Base distributions: `StandardNormal`, `DiagNormal` |
| `nets.py` | Conditioner networks: `MLP` with optional context |
| `builders.py` | High-level constructors: `build_realnvp`, `build_spline_realnvp` |

### Parameter Structure

Parameters are explicit PyTrees:

```python
params = {
    "base": {...},       # Base distribution params (empty for StandardNormal)
    "transform": [...]   # List of per-block params
}
```

### Transform API

All transforms follow the same interface:

```python
# Forward: z -> x (sampling direction)
x, log_det = transform.forward(params, z, context=None)

# Inverse: x -> z (density evaluation direction)
z, log_det = transform.inverse(params, x, context=None)
```

## Conditioning

For conditional flows, context is concatenated to the masked input before the conditioner MLP:

```python
# input to MLP = [x_masked, context]
flow, params = build_realnvp(..., context_dim=2)
```

Context can be per-sample `(batch, context_dim)` or shared `(context_dim,)`.

## Training

The library provides density evaluation; training is up to you:

```python
import optax

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, x, context=None):
    # Negative log-likelihood
    return -flow.log_prob(params, x, context=context).mean()

@jax.jit
def step(params, opt_state, x, context=None):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, context)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
for batch in data_loader:
    params, opt_state, loss = step(params, opt_state, batch)
```

## Key Design Choices

- **Explicit parameters**: No hidden state. Parameters are always passed explicitly.
- **JAX-native**: JIT-compatible, vmap-friendly, functional style.
- **Zero-init conditioners**: Flows start as identity for stable training.
- **LOFT stabilization**: Logarithmic tails prevent numerical issues in high dimensions.

## References

- Dinh et al. (2017). "Density estimation using Real-NVP"
- Durkan et al. (2019). "Neural Spline Flows"
- Andrade (2021). "Stable Training of Normalizing Flows for High-dimensional Variational Inference"
