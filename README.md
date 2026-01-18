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
