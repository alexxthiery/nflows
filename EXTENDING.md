# Extending nflows

Recipes for adding custom Transforms, Distributions, and Conditioners.

---

## Adding a Custom Transform

### Required Interface

```python
def forward(params, x, context=None) -> (y, log_det)
def inverse(params, y, context=None) -> (x, log_det)
```

- `params`: PyTree of learnable parameters (can be empty dict `{}`)
- `x`, `y`: Arrays of shape `(..., dim)`
- `context`: Optional conditioning tensor, shape `(..., context_dim)` or `None`
- `log_det`: Log absolute Jacobian determinant, shape `(...,)`

### Optional Methods

```python
def init_params(key, context_dim=0) -> params     # Initialize parameters
@classmethod
def create(cls, key, ...) -> (transform, params)  # Factory method
```

**Important:** All transforms must use the signature `init_params(key, context_dim=0)`.
Transforms that don't use `context_dim` should accept but ignore the argument.
This ensures uniform interface for `CompositeTransform`.

Example for a parameter-free transform:

```python
def init_params(self, key, context_dim=0):
    del key, context_dim  # Unused
    return {}
```

### Templates

- **`Permutation`** — Simplest: no learnable parameters, just shuffles dimensions
- **`LoftTransform`** — Parameter-free but with nontrivial computation
- **`AffineCoupling`** — Full example with conditioner network

### Integration

Use with `CompositeTransform` to chain transforms:

```python
from nflows.transforms import CompositeTransform

composite = CompositeTransform(blocks=[transform1, transform2, transform3])
y, log_det = composite.forward(params_list, x)
```

---

## Adding a Custom Distribution

### Required Interface

```python
def log_prob(params, x) -> log_prob
def sample(params, key, shape) -> samples
```

- `params`: PyTree of parameters (can be `None` for parameter-free distributions)
- `x`: Array of shape `(..., dim)`
- `log_prob`: Log probability, shape `(...,)`
- `key`: JAX PRNGKey
- `shape`: Tuple for batch dimensions (samples will be `shape + (dim,)`)

### Optional Methods

```python
def init_params() -> params  # Initialize parameters (no key needed)
```

### Templates

- **`StandardNormal`** — Simplest: isotropic Gaussian, no learnable params
- **`DiagNormal`** — Diagonal Gaussian with learnable `loc` and `log_scale`

### Integration

Use with `Flow` as the base distribution:

```python
from nflows.flows import Flow

flow = Flow(transform=my_transform, base_dist=my_distribution, dim=dim)
log_prob = flow.log_prob(params, x)
samples = flow.sample(params, key, (batch_size,))
```

---

## Adding a Custom Conditioner

### Required Interface

```python
def apply({"params": params}, x, context) -> output  # Flax convention
context_dim: int  # Attribute (0 for unconditional)
```

- Must follow Flax module conventions
- `x`: Input tensor, shape `(..., x_dim)`
- `context`: Optional conditioning tensor or `None`
- `output`: Shape `(..., out_dim)` where `out_dim` depends on the coupling type

### Optional Methods (for auto-initialization)

```python
def get_output_layer(params) -> {"kernel": Array, "bias": Array}
def set_output_layer(params, kernel, bias) -> params
```

If present, coupling layers will automatically initialize the output layer:
- `AffineCoupling`: Zero-initializes for identity start
- `SplineCoupling`: Sets biases for identity spline (raises error if methods missing)

**Note:** `SplineCoupling` emits a warning if the derivative range `[min_derivative, max_derivative]`
does not contain 1.0, since identity-like initialization requires `derivative ≈ 1`. In this case,
the midpoint derivative is used instead.

### Template

- **`MLP`** in `nets.py` — Full implementation with context handling and output layer access

### Output Dimensions

Conditioner output size depends on the coupling type:

```python
AffineCoupling.required_out_dim(dim)           # Returns 2 * dim
SplineCoupling.required_out_dim(dim, num_bins) # Returns dim * (3 * num_bins - 1)
```
