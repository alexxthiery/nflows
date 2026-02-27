# Usage

How-to cookbook for nflows. Each section is self-contained with a copy-pasteable example.
For API details, see [REFERENCE.md](REFERENCE.md). For math, see [INTERNALS.md](INTERNALS.md).

**Contents:**

- [Affine Flow (RealNVP)](#affine-flow-realnvp)
- [Spline Flows](#spline-flows)
- [Conditional Flows](#conditional-flows)
- [Feature Extractor](#feature-extractor)
- [Transform-Only Mode](#transform-only-mode-bijection)
- [Identity Gating](#identity-gating)
- [Custom Architectures](#custom-architectures-assembly-api)
- [Assembly with Context](#assembly-with-context-and-feature-extractor)
- [Training](#training)

## Affine Flow (RealNVP)

Affine coupling layers as in Dinh et al. (2017). Build a flow, draw samples, evaluate log-density.

```python
import jax
from nflows.builders import build_realnvp

key = jax.random.PRNGKey(0)
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
)

samples = flow.sample(params, key, shape=(1000,))          # (1000, 16)
log_probs = flow.log_prob(params, samples)                  # (1000,)
samples, log_probs = flow.sample_and_log_prob(params, key, shape=(1000,))
```

Full options: [REFERENCE.md#builder-options](REFERENCE.md#builder-options)

## Spline Flows

Rational-quadratic splines are more expressive than affine couplings.

```python
from nflows.builders import build_spline_realnvp

flow, params = build_spline_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    num_bins=8,           # spline resolution
    tail_bound=5.0,       # linear tails outside [-B, B]
)
```

Spline-specific options: [REFERENCE.md#builder-options](REFERENCE.md#builder-options).
How splines work: [INTERNALS.md#spline-coupling](INTERNALS.md#spline-coupling)

## Conditional Flows

Model $p(x \mid \text{context})$ by setting `context_dim > 0`. Context is concatenated to conditioner inputs.

```python
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=4,
)

context = jax.random.normal(key, (1000, 4))   # per-sample context
samples = flow.sample(params, key, shape=(1000,), context=context)
log_probs = flow.log_prob(params, samples, context=context)
```

Context can be per-sample `(batch, context_dim)` or shared `(context_dim,)`.

How conditioning works: [INTERNALS.md#conditional-normalizing-flows](INTERNALS.md#conditional-normalizing-flows)

## Feature Extractor

For high-dimensional or heterogeneous context, a learned ResNet can preprocess context before coupling layers.

```python
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=64,                      # raw context dimension
    context_extractor_hidden_dim=128,    # >0 enables the extractor
    context_extractor_n_layers=2,        # depth of extractor
    context_feature_dim=16,              # output dim (default: same as context_dim)
)
```

The extractor is shared across all coupling layers. Its params live in `params["feature_extractor"]`.

Extractor options: [REFERENCE.md#context-feature-extractor](REFERENCE.md#context-feature-extractor)

## Transform-Only Mode (Bijection)

When you only need the invertible map with tractable Jacobian, without a base distribution:

```python
bijection, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=4,
    return_transform_only=True,
)

x = jax.random.normal(key, (1000, 16))
context = jax.random.normal(key, (1000, 4))

y, log_det_fwd = bijection.forward(params, x, context=context)
x_rec, log_det_inv = bijection.inverse(params, y, context=context)
# log_det_fwd + log_det_inv approx 0 (invertibility)
```

### Use cases

**Change of variables in integration:**

```python
z = sample_base(key, shape)
x, log_det = bijection.forward(params, z, context=context)
# Integrate f(x) * exp(log_det) under base measure
```

**Custom base distribution:**

```python
from nflows.flows import Flow

bijection, bij_params = build_realnvp(..., return_transform_only=True)
my_flow = Flow(
    base_dist=my_custom_dist,
    transform=bijection.transform,
    feature_extractor=bijection.feature_extractor,
)
```

## Identity Gating

Enforce that the transform is identity at specific context values.
The gate function maps context to a scalar; wherever it returns 0, the transform becomes identity ($x \to x$, $\log\det = 0$).

```python
import jax.numpy as jnp
from nflows.builders import build_realnvp

# Gate = sin(pi * t): identity at t=0 and t=1, full transform at t=0.5
gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

flow, params = build_realnvp(
    key, dim=4, num_layers=4, hidden_dim=64, n_hidden_layers=2,
    context_dim=1,
    identity_gate=gate_fn,
)

# At t=0: transform is identity
x = jax.random.normal(key, (100, 4))
ctx_0 = jnp.zeros((100, 1))
y_0, log_det_0 = flow.forward(params, x, context=ctx_0)
# y_0 approx x, log_det_0 approx 0

# At t=0.5: full learned transform
ctx_half = jnp.ones((100, 1)) * 0.5
y_half, _ = flow.forward(params, x, context=ctx_half)
```

**Constraints:**
- Requires `context_dim > 0`
- Incompatible with `use_permutation=True`
- Gate receives **raw context**, even with a feature extractor. Couplings see extracted features; the gate does not. This is by design: the gate encodes known structure (e.g., boundary conditions), so it operates on interpretable inputs.
- Gate must be written for a **single sample** `(context_dim,)`. Batching is handled via `jax.vmap`.

How it works: [INTERNALS.md#identity-gate](INTERNALS.md#identity-gate)

## Custom Architectures (Assembly API)

Mix coupling types and control layer order using the assembly API.

```python
import jax
from nflows.builders import make_alternating_mask, assemble_bijection, assemble_flow
from nflows.transforms import AffineCoupling, SplineCoupling, LinearTransform, LoftTransform
from nflows.distributions import StandardNormal

keys = jax.random.split(jax.random.PRNGKey(0), 5)
dim = 8
mask0 = make_alternating_mask(dim, parity=0)
mask1 = make_alternating_mask(dim, parity=1)

blocks_and_params = [
    AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2),
    AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=64, n_hidden_layers=2),
    SplineCoupling.create(keys[2], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2, num_bins=8),
    LinearTransform.create(keys[3], dim=dim),
    LoftTransform.create(keys[4], dim=dim),
]

# As Bijection (no base distribution)
bijection, params = assemble_bijection(blocks_and_params)

# As Flow (with base distribution)
flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=dim))
```

Assembly API reference: [REFERENCE.md#assembly-api](REFERENCE.md#assembly-api)

## Assembly with Context and Feature Extractor

When using the assembly API with conditioning, create the feature extractor separately and pass the output dimension as `context_dim` to each coupling.

```python
from nflows.builders import make_alternating_mask, create_feature_extractor, assemble_bijection
from nflows.transforms import AffineCoupling, LoftTransform

keys = jax.random.split(key, 4)
dim = 8
raw_context_dim = 16
effective_context_dim = 8

# Create feature extractor
fe, fe_params = create_feature_extractor(
    keys[0], in_dim=raw_context_dim, hidden_dim=32, out_dim=effective_context_dim,
)

# Couplings use effective_context_dim (not raw)
mask0 = make_alternating_mask(dim, parity=0)
mask1 = make_alternating_mask(dim, parity=1)

blocks_and_params = [
    AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=64,
                          n_hidden_layers=2, context_dim=effective_context_dim),
    AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=64,
                          n_hidden_layers=2, context_dim=effective_context_dim),
    LoftTransform.create(keys[3], dim=dim),
]

bijection, params = assemble_bijection(
    blocks_and_params,
    feature_extractor=fe,
    feature_extractor_params=fe_params,
)

# Pass raw context; the extractor transforms it internally
raw_context = jax.random.normal(key, (100, raw_context_dim))
y, log_det = bijection.forward(params, x, context=raw_context)
```

## Training

The library provides density evaluation; training loops are up to you.

### Forward KL (Maximum Likelihood)

Train on observed data by minimizing negative log-likelihood.

```python
import optax

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, x, context=None):
    return -flow.log_prob(params, x, context=context).mean()

@jax.jit
def step(params, opt_state, x, context=None):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, context)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for batch in data_loader:
    params, opt_state, loss = step(params, opt_state, batch)
```

### Reverse KL (Variational Inference)

Train by sampling from the flow and minimizing $\text{KL}(q \| p)$ against an unnormalized target.

```python
import jax.numpy as jnp
import optax

def log_target(x):
    """Unnormalized log density of target."""
    return -0.5 * jnp.sum(x**2, axis=-1)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, key):
    x, log_q = flow.sample_and_log_prob(params, key, shape=(256,))
    return jnp.mean(log_q - log_target(x))

@jax.jit
def train_step(params, opt_state, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for i in range(1000):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, subkey)
```
