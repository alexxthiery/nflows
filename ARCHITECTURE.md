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

## Parameter Convention

All parameters are explicit PyTrees passed to methods (no params stored in objects):

```python
params = {
    "base": {...},       # base distribution params (or {} for StandardNormal)
    "transform": [...],  # list of per-block params
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

## Key Builder Options

| Option | Default | Purpose |
|--------|---------|---------|
| `trainable_base` | False | Use DiagNormal base with learnable loc/scale |
| `use_linear` | False | Add global LU linear transform at start |
| `use_permutation` | False | Insert reverse permutations between couplings |
| `max_log_scale` | 5.0 | Bound on affine coupling scale (stability) |
| `loft_tau` | 1000.0 | LOFT threshold for tail stabilization |
| `activation` | tanh | Conditioner MLP activation |
