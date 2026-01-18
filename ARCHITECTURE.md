# Architecture

## Core Design

A normalizing flow = **base distribution** + **invertible transform**.

```
z ~ p_base(z)  -->  T(z)  -->  x ~ q(x)
```

The `Flow` class composes these and provides:
- `sample(params, key, shape)` — draw samples via forward transform
- `log_prob(params, x)` — compute density via inverse transform + change of variables
- `sample_and_log_prob(params, key, shape)` — efficient combined operation for VI

### Transform-Only Mode (Bijection)

Sometimes you only need the invertible transform with tractable Jacobian, without a base
distribution. Use `return_transform_only=True` to get a `Bijection` instead of a `Flow`:

```python
bijection, params = build_realnvp(..., return_transform_only=True)

y, log_det = bijection.forward(params, x, context=ctx)
x_rec, _   = bijection.inverse(params, y, context=ctx)
```

Use cases:
- Change of variables in integration
- Learned coordinate transformations
- Composing with custom base distributions
- Reparameterization tricks

## Parameter Convention

All parameters are explicit PyTrees passed to methods (no params stored in objects):

```python
# Flow params
params = {
    "base": {...},       # base distribution params (or {} for StandardNormal)
    "transform": [...],  # list of per-block params
}

# Bijection params (return_transform_only=True)
params = {
    "transform": [...],  # list of per-block params
    # "feature_extractor": {...}  # only if context_extractor_hidden_dim > 0
}
```

## Transform Composition

Transforms are composed via `CompositeTransform`:

```
x  -->  [Block_1]  -->  [Block_2]  -->  ...  -->  [Block_n]  -->  y
```

Each block implements `forward(params, x, context)` and `inverse(params, y, context)`,
returning `(output, log_det)`.

Available transforms:
- `AffineCoupling` — RealNVP-style affine coupling
- `SplineCoupling` — Rational-quadratic spline coupling (more expressive)
- `LinearTransform` — LU-parameterized invertible linear layer
- `Permutation` — Fixed dimension permutation
- `LoftTransform` — Log-soft tail extension for stability in high dimensions

## Conditional Flows

All transforms accept an optional `context` argument. For conditional flows, set
`context_dim > 0` in builders — the context is concatenated to conditioner inputs.

### Context Feature Extractor

Optionally, a learned MLP can preprocess the context before it's used in the flow:

```python
flow, params = build_realnvp(
    key, dim=2,
    context_dim=5,                      # Raw context dimension
    context_extractor_hidden_dim=64,    # > 0 enables feature extraction
    context_extractor_n_layers=2,       # Depth of extractor (default: 2)
    context_feature_dim=32,             # Output dim (default: same as context_dim)
    ...
)
```

When enabled:
1. Context passes through a shared `ResNet` feature extractor once
2. Extracted features replace raw context for all coupling layers
3. Gradients flow through the extractor for end-to-end training

The feature extractor params are stored in `params["feature_extractor"]`.

**When to use:** When raw context features are not directly informative and a learned
embedding improves conditioning (e.g., high-dimensional contexts, heterogeneous features).

## Key Builder Options

| Option | Default | Purpose |
|--------|---------|---------|
| `hidden_dim` | (required) | Width of conditioner MLP hidden layers |
| `n_hidden_layers` | (required) | Number of residual blocks in conditioner MLP |
| `res_scale` | 0.1 | Scale factor for residual connections |
| `trainable_base` | False | Use DiagNormal base with learnable loc/scale |
| `use_linear` | False | Add global LU linear transform at start |
| `use_permutation` | False | Insert reverse permutations between couplings |
| `use_loft` | True | Append LoftTransform at end for tail stabilization |
| `max_log_scale` | 5.0 | Bound on affine coupling scale (stability) |
| `loft_tau` | 1000.0 | LOFT threshold for tail stabilization |
| `activation` | tanh | Conditioner MLP activation |
| `context_extractor_hidden_dim` | 0 | Hidden dim for context feature extractor (0 = disabled) |
| `context_extractor_n_layers` | 2 | Residual blocks in context extractor |
| `context_feature_dim` | None | Output dim of extractor (default: same as `context_dim`) |
| `return_transform_only` | False | Return `Bijection` instead of `Flow` (no base distribution) |
