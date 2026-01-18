# Usage

## Build and sample from a flow

```python
import jax
from nflows.builders import build_realnvp

key = jax.random.PRNGKey(0)

# Build flow: returns (flow_definition, initial_params)
flow, params = build_realnvp(
    key=key,
    dim=16,
    num_layers=8,
    hidden_dim=256,
    n_hidden_layers=2,
)

# Sample
samples = flow.sample(params, key, shape=(1000,))  # (1000, 16)

# Evaluate log density
log_probs = flow.log_prob(params, samples)  # (1000,)
```

## Train with reverse KL (variational inference)

```python
import jax
import jax.numpy as jnp
import optax
from nflows.builders import build_realnvp

def log_target(x):
    """Unnormalized log density of target distribution."""
    return -0.5 * jnp.sum(x**2, axis=-1)  # example: standard normal

# Build flow
key = jax.random.PRNGKey(0)
flow, params = build_realnvp(key, dim=2, num_layers=4, hidden_dim=64, n_hidden_layers=2)

# Loss: reverse KL = E_q[log q - log π̃]
def loss_fn(params, key):
    x, log_q = flow.sample_and_log_prob(params, key, shape=(256,))
    return jnp.mean(log_q - log_target(x))

# Training loop
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for step in range(1000):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, subkey)
```

## Spline flows (more expressive)

```python
from nflows.builders import build_spline_realnvp

flow, params = build_spline_realnvp(
    key=key,
    dim=16,
    num_layers=8,
    hidden_dim=256,
    n_hidden_layers=2,
    num_bins=8,           # spline resolution
    tail_bound=5.0,       # linear tails outside [-B, B]
)
```

## Conditional flows

```python
flow, params = build_realnvp(
    key=key,
    dim=16,
    num_layers=8,
    hidden_dim=256,
    n_hidden_layers=2,
    context_dim=4,  # conditioning variable dimension
)

# Sampling and density evaluation with context
context = jnp.ones((1000, 4))
samples = flow.sample(params, key, shape=(1000,), context=context)
log_probs = flow.log_prob(params, samples, context=context)
```

## Transform-only mode (Bijection)

When you only need the invertible transform with tractable Jacobian—without defining
a probability distribution—use `return_transform_only=True`:

```python
from nflows.builders import build_realnvp

# Build bijection (no base distribution)
bijection, params = build_realnvp(
    key=key,
    dim=16,
    num_layers=8,
    hidden_dim=256,
    n_hidden_layers=2,
    context_dim=4,
    return_transform_only=True,
)

# Forward and inverse maps
x = jax.random.normal(key, (1000, 16))
context = jnp.ones((1000, 4))

y, log_det_fwd = bijection.forward(params, x, context=context)
x_rec, log_det_inv = bijection.inverse(params, y, context=context)

# log_det_fwd + log_det_inv ≈ 0 (invertibility)
```

### Use cases

**Change of variables in integration:**
```python
# Compute E_p[f(x)] via change of variables
# where x = bijection(z) and z ~ base_measure
z = sample_base(key, shape)
x, log_det = bijection.forward(params, z, context=context)
# Integrate f(x) * exp(log_det) under base measure
```

**Custom base distribution:**
```python
from nflows.flows import Flow

# Use bijection with your own base distribution
bijection, bij_params = build_realnvp(..., return_transform_only=True)
my_flow = Flow(base_dist=my_custom_dist, transform=bijection.transform,
               feature_extractor=bijection.feature_extractor)
```
