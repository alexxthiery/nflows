# API Reference

**Contents:**

- [Core Classes](#core-classes)
- [Builders](#builders)
- [Assembly API](#assembly-api)
- [Transforms](#transforms)
- [Distributions](#distributions)
- [Parameter Structure](#parameter-structure)
- [Forward/Inverse Convention](#forwardinverse-convention)
- [Context Feature Extractor](#context-feature-extractor)

## Core Classes

### Flow

Normalizing flow distribution: base distribution + invertible transform.

```python
from nflows.flows import Flow

flow = Flow(base_dist, transform, feature_extractor=None, identity_gate=None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(params, z, context=None)` | `(x, log_det)` |
| `inverse` | `(params, x, context=None)` | `(z, log_det)` |
| `log_prob` | `(params, x, context=None)` | `log_prob` |
| `sample` | `(params, key, shape, context=None)` | `x` |
| `sample_and_log_prob` | `(params, key, shape, context=None)` | `(x, log_prob)` |

Shapes: `x`, `z` are `(..., dim)`. `log_det`, `log_prob` are `(...,)`. `context` is `(..., context_dim)` or `None`.

### Bijection

Invertible transform with tractable Jacobian, no base distribution.

```python
from nflows.flows import Bijection

bijection = Bijection(transform, feature_extractor=None, identity_gate=None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(params, x, context=None)` | `(y, log_det)` |
| `inverse` | `(params, y, context=None)` | `(x, log_det)` |

## Builders

### build_realnvp

```python
from nflows.builders import build_realnvp

flow_or_bijection, params = build_realnvp(
    key, dim, num_layers, hidden_dim, n_hidden_layers, **options
)
```

### build_spline_realnvp

```python
from nflows.builders import build_spline_realnvp

flow_or_bijection, params = build_spline_realnvp(
    key, dim, num_layers, hidden_dim, n_hidden_layers, **options
)
```

### Builder Options

**Shared options** (both builders):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context_dim` | int | 0 | Conditioning variable dimension; 0 = unconditional |
| `context_extractor_hidden_dim` | int | 0 | Feature extractor hidden width; 0 = disabled |
| `context_extractor_n_layers` | int | 2 | Residual blocks in context extractor |
| `context_feature_dim` | int or None | None | Extractor output dim; None = same as context_dim |
| `res_scale` | float | 0.1 | Scale factor for residual connections |
| `activation` | callable | `jax.nn.tanh` | Conditioner MLP activation |
| `use_permutation` | bool | False | Reverse permutations between couplings |
| `use_linear` | bool | False | Prepend LU-parameterized linear transform |
| `use_loft` | bool | True | Append LoftTransform for tail stabilization |
| `loft_tau` | float | 1000.0 | LOFT threshold parameter |
| `trainable_base` | bool | False | Use DiagNormal with learnable loc/scale |
| `base_dist` | object or None | None | Custom base distribution; overrides trainable_base |
| `base_params` | PyTree or None | None | Params for custom base_dist |
| `return_transform_only` | bool | False | Return Bijection instead of Flow |
| `identity_gate` | callable or None | None | Context -> scalar gate for identity interpolation |

**Affine-specific** (`build_realnvp` only):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_log_scale` | float | 5.0 | Bound on log-scale via tanh clamping |

**Spline-specific** (`build_spline_realnvp` only):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_bins` | int | 8 | Number of spline bins (K) |
| `tail_bound` | float | 5.0 | Spline acts on [-B, B]; linear tails outside |
| `min_bin_width` | float | 1e-2 | Floor for bin widths |
| `min_bin_height` | float | 1e-2 | Floor for bin heights |
| `min_derivative` | float | 1e-2 | Lower bound for knot derivatives |
| `max_derivative` | float | 10.0 | Upper bound for knot derivatives |

## Assembly API

For custom architectures (mixing coupling types, non-standard layer order).

### assemble_bijection

```python
from nflows.builders import assemble_bijection

bijection, params = assemble_bijection(
    blocks_and_params,
    feature_extractor=None,
    feature_extractor_params=None,
    validate=True,
    identity_gate=None,
)
```

`blocks_and_params` is a list of `(transform, params)` tuples from `.create()` calls.
Returns params dict: `{"transform": [...], "feature_extractor": ...}`.

### assemble_flow

```python
from nflows.builders import assemble_flow

flow, params = assemble_flow(
    blocks_and_params,
    base,
    base_params=None,
    feature_extractor=None,
    feature_extractor_params=None,
    validate=True,
    identity_gate=None,
)
```

Returns params dict: `{"base": ..., "transform": [...], "feature_extractor": ...}`.

### Utilities

```python
from nflows.builders import make_alternating_mask, create_feature_extractor

mask = make_alternating_mask(dim, parity)    # parity: 0 or 1
fe, fe_params = create_feature_extractor(key, in_dim, hidden_dim, out_dim, n_layers=2)
```

## Transforms

All transforms share the interface:

```python
y, log_det = transform.forward(params, x, context=None, g_value=None)
x, log_det = transform.inverse(params, y, context=None, g_value=None)
transform, params = TransformClass.create(key, **kwargs)
```

| Transform | `.create()` params | Per-block param shape | Notes |
|-----------|-------------------|----------------------|-------|
| `AffineCoupling` | `dim, mask, hidden_dim, n_hidden_layers, context_dim=0, max_log_scale=1.0` | `{"mlp": {...}}` | RealNVP-style; output dim = 2*dim |
| `SplineCoupling` | `dim, mask, hidden_dim, n_hidden_layers, context_dim=0, num_bins=8, tail_bound=5.0` | `{"mlp": {...}}` | RQ-spline; output dim = dim*(3K-1) |
| `LinearTransform` | `dim` | `{"lower": (d,d), "upper": (d,d), "log_diag": (d,)}` | LU-parameterized invertible linear |
| `Permutation` | `perm` (1D index array) | `{}` | Fixed dimension shuffle; log_det=0 |
| `LoftTransform` | `dim, tau=1000.0` | `{}` | Log-soft tails for stability |
| `CompositeTransform` | `blocks` (list of transforms) | list of per-block params | Sequential composition |

## Distributions

```python
from nflows.distributions import StandardNormal, DiagNormal
```

| Distribution | Constructor | Params | Description |
|-------------|-------------|--------|-------------|
| `StandardNormal` | `StandardNormal(dim)` | `None` | Isotropic N(0, I) |
| `DiagNormal` | `DiagNormal(dim)` | `{"loc": (dim,), "log_scale": (dim,)}` | Diagonal covariance |

Both provide: `log_prob(params, x)`, `sample(params, key, shape)`, `init_params()`.

## Parameter Structure

### Flow params

```python
params = {
    "base": None,            # StandardNormal (no params)
    # or: {"loc": (dim,), "log_scale": (dim,)}  # DiagNormal
    "transform": [
        {"mlp": {...}},      # coupling layer 0
        {"mlp": {...}},      # coupling layer 1
        {},                  # LoftTransform (no params)
    ],
    # "feature_extractor": {...}  # only if extractor enabled
}
```

### Bijection params

```python
params = {
    "transform": [
        {"mlp": {...}},      # coupling layer 0
        {"mlp": {...}},      # coupling layer 1
        {},                  # LoftTransform (no params)
    ],
    # "feature_extractor": {...}  # only if extractor enabled
}
```

Optional blocks when enabled:
- `use_linear=True`: first entry is `{"lower": (d,d), "upper": (d,d), "log_diag": (d,)}`
- `use_permutation=True`: empty dict `{}` entries between couplings

## Forward/Inverse Convention

| Method | Direction | Log-det sign |
|--------|-----------|--------------|
| `forward(z)` | $z \to x$ | $+\log\lvert\det \partial x/\partial z\rvert$ |
| `inverse(x)` | $x \to z$ | $+\log\lvert\det \partial z/\partial x\rvert$ |

`inverse` returns the log-det of the inverse map (negative of the forward log-det).

Density evaluation uses:
```python
log_prob(x) = base.log_prob(z) + log_det_inv   # where z, log_det_inv = inverse(x)
```

Efficient sampling uses:
```python
log_prob(x) = base.log_prob(z) - log_det_fwd   # where x, log_det_fwd = forward(z)
```

## Context Feature Extractor

When `context_extractor_hidden_dim > 0`, a shared ResNet preprocesses raw context before coupling layers see it.

| Parameter | Description |
|-----------|-------------|
| `context_extractor_hidden_dim` | Hidden layer width; 0 disables the extractor |
| `context_extractor_n_layers` | Residual blocks in the extractor (default: 2) |
| `context_feature_dim` | Output dimension; defaults to `context_dim` |

The extractor params live in `params["feature_extractor"]`.
Coupling layers receive the extracted features (dimension = `context_feature_dim`), not the raw context.

When using the assembly API, create the extractor separately and pass the output dimension as `context_dim` to each coupling:

```python
fe, fe_params = create_feature_extractor(key, in_dim=raw_dim, hidden_dim=32, out_dim=8)
# couplings use context_dim=8 (the extractor output dim)
bijection, params = assemble_bijection(blocks, feature_extractor=fe, feature_extractor_params=fe_params)
```

See [USAGE.md](USAGE.md#assembly-with-context-and-feature-extractor) for full examples.
For the math behind conditioning, see [INTERNALS.md](INTERNALS.md#conditional-normalizing-flows).
