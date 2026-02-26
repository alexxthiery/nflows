# Internals

Mathematical foundations and design decisions behind nflows.

**Contents:**

- [Change of Variables](#change-of-variables)
- [Coupling Layers](#coupling-layers)
- [Composing Transformations](#composing-transformations)
- [Conditional Normalizing Flows](#conditional-normalizing-flows)
- [Identity Gate](#identity-gate)
- [Numerical Stability](#numerical-stability)
- [Density Evaluation](#density-evaluation)
- [References](#references)

## Change of Variables

Let $z \sim p_0(z)$ be a sample from the base distribution (typically standard Gaussian), and let $f: \mathbb{R}^d \to \mathbb{R}^d$ be an invertible transformation.
The transformed variable $x = f(z)$ has density:

$$
p(x) = p_0(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

Taking logarithms:

$$
\log p(x) = \log p_0(z) + \log \left| \det \frac{\partial z}{\partial x} \right|
$$

where $z = f^{-1}(x)$.

The computational bottleneck is computing $\det \frac{\partial f}{\partial z}$, which is $O(d^3)$ for general transformations.
Modern normalizing flows design transformations with tractable (cheaper-than-cubic) Jacobians.

## Coupling Layers

### Affine Coupling (RealNVP)

The affine coupling layer splits the input into two parts using a binary mask $m \in \{0,1\}^d$.

**Forward** ($z \to x$):

$$
\begin{aligned}
x_{\text{masked}} &= z \odot m \\
x_{\text{transformed}} &= z \odot (1-m) \cdot \exp(s(z \odot m)) + t(z \odot m) \\
x &= x_{\text{masked}} + x_{\text{transformed}}
\end{aligned}
$$

where $s, t: \mathbb{R}^d \to \mathbb{R}^d$ are scale and shift functions (conditioner MLP outputs).

**Jacobian**: triangular, so its determinant is the product of diagonal elements:

$$
\log \left| \det \frac{\partial x}{\partial z} \right| = \sum_{i: m_i=0} s_i(z \odot m)
$$

**Inverse** ($x \to z$):

$$
\begin{aligned}
z_{\text{masked}} &= x \odot m \\
z_{\text{transformed}} &= (x \odot (1-m) - t(x \odot m)) \cdot \exp(-s(x \odot m)) \\
z &= z_{\text{masked}} + z_{\text{transformed}}
\end{aligned}
$$

The inverse is computed analytically without iterative methods.

### Spline Coupling

Instead of affine transformations, spline coupling uses monotonic rational-quadratic splines [durkan2019].

For each dimension $i$ where $m_i = 0$:

$$
x_i = \text{RQSpline}(z_i; \theta_i)
$$

where $\theta_i = (w_i, h_i, d_i)$ are spline parameters (bin widths, heights, knot derivatives) predicted by the conditioner.

**Properties**:
- Monotonic and invertible by construction
- $C^1$ continuous
- More expressive than affine (can model multimodal conditionals)
- Identity outside $[-B, B]$ (linear tails)

**Parameterization** (per dimension, $K$ bins):
- Widths: $K$ parameters -> softmax -> min_width floor -> scale to $2B$
- Heights: $K$ parameters -> softmax -> min_height floor -> scale to $2B$
- Derivatives: $K-1$ internal derivatives (boundary derivatives fixed to 1)

Internal knot derivatives are bounded via sigmoid:

$$
d = d_{\min} + (d_{\max} - d_{\min}) \cdot \sigma(d_{\text{raw}})
$$

where $d_{\min}$ and $d_{\max}$ are hyperparameters (defaults: 0.01 and 10.0).

Total: $3K - 1$ parameters per transformed dimension.

## Composing Transformations

### Composite Transform

Multiple transforms compose sequentially:

$$
f = f_n \circ f_{n-1} \circ \cdots \circ f_1
$$

The log-determinant accumulates:

$$
\log \left| \det \frac{\partial f}{\partial z} \right| = \sum_{i=1}^{n} \log \left| \det \frac{\partial f_i}{\partial z_{i-1}} \right|
$$

### Mask Alternation

A single coupling layer only transforms dimensions where $m_i = 0$.
To ensure all dimensions are transformed, masks alternate:

- Layer 1: mask = $[1, 0, 1, 0, \ldots]$ (parity=0)
- Layer 2: mask = $[0, 1, 0, 1, \ldots]$ (parity=1)
- Layer 3: parity=0 again, etc.

The builder validates mask coverage at construction time and raises an error if any original dimension is never transformed (accounting for permutations).

### Permutations

Fixed permutations between coupling layers improve mixing.

For a permutation $\pi$, the forward transform is $y_i = x_{\pi(i)}$.
The Jacobian is a permutation matrix with determinant $\pm 1$, so $\log\lvert\det\rvert = 0$.

This library uses a reverse permutation $\pi(i) = d - 1 - i$ between coupling layers when `use_permutation=True`.

### Linear Transform

Optional global linear transformation with LU-style parameterization.

Factor an invertible matrix $W \in \mathbb{R}^{d \times d}$ as:

$$
W = L \cdot T
$$

where:
- $L = \text{tril}(L_{\text{raw}}, k=-1) + I$, unit-diagonal lower triangular
- $T = \text{triu}(U_{\text{raw}}, k=1) + \text{diag}(\exp(\log s))$, upper triangular with positive diagonal

**Parameters**: `lower` $(d, d)$, `upper` $(d, d)$, `log_diag` $(d,)$.

**Complexity**: forward/inverse $O(d^2)$ via triangular solves; log-determinant $O(d)$:

$$
\log\lvert\det W\rvert = \sum_{i=1}^{d} \text{log\_diag}_i
$$

Initialized to identity ($L = I$, $T = I$) via zero initialization.

## Conditional Normalizing Flows

Conditional flows model $p(x \mid c)$ where $c$ is a conditioning variable.

### Formulation

The base distribution remains unconditional: $z \sim p_0(z)$.
The transformation becomes context-dependent: $x = f(z; c)$.
For fixed $c$, the map $f(\cdot; c)$ must be invertible in $z$.

Conditional density:

$$
\log p(x \mid c) = \log p_0(z) + \log \left| \det \frac{\partial z}{\partial x} \right|
$$

where $z = f^{-1}(x; c)$.
The Jacobian is taken w.r.t. $x$ (or $z$), not $c$.
The conditioning variable affects transformation parameters but does not participate in the change of variables.

### Concatenation Strategy

Context is concatenated to the conditioner input:

$$
[s, t] = \text{MLP}([z_{\text{masked}}, c])
$$

The MLP input dimension becomes `dim + context_dim`.
Simple to implement and works well for low-dimensional context, but context influence may be diluted in deep networks.

### Context Broadcasting

Context can be per-sample `(B, context_dim)` or shared `(context_dim,)`.
Shared context is broadcast to match batch dimensions.

## Identity Gate

The identity gate enables smooth interpolation between the identity transform and the learned transform based on context.

### Mechanics

When `identity_gate` is provided, each coupling layer scales its conditioner output (the raw parameters that define the transform) by the gate value before computing the transformation:

- `gate = 0`: conditioner output zeroed -> transform is identity, `log_det = 0`
- `gate = 1`: conditioner output unchanged -> transform acts normally
- `0 < gate < 1`: smooth interpolation

For affine coupling, this means $s \to g \cdot s$ and $t \to g \cdot t$.
For spline coupling, the raw spline parameters are scaled by $g$, pulling the spline toward identity.

`LinearTransform` also supports gating: $W \to g \cdot W + (1-g) \cdot I$.

### Raw Context vs Extracted Features

The gate function always receives raw context, even when a feature extractor is used.
Coupling layers see extracted features, but the gate does not.

This is by design: the gate encodes known structure (e.g., boundary conditions at specific parameter values), so it operates on interpretable inputs rather than a learned representation that changes during training.

### Constraints

- Requires `context_dim > 0` (gate operates on context)
- Incompatible with `use_permutation=True` (permutations cannot be smoothly gated)
- `LoftTransform` does not support gating (known issue)

### Single-Sample Contract

The gate function must be written for a single sample with shape `(context_dim,)`.
Batching is handled internally via `jax.vmap`.
A gate function that expects batched input `(batch, context_dim)` will produce silently wrong results.

The library validates this at build time using `jax.eval_shape` (zero FLOPs).

## Numerical Stability

### Zero Initialization

Conditioner networks are initialized with zero final-layer weights and biases:

$$
W_{\text{out}} = 0, \quad b_{\text{out}} = 0
$$

This means:
- Affine coupling: $s = 0, t = 0$ -> identity transform
- Spline coupling: uniform bins, unit derivatives -> near-identity

The flow starts as identity, avoiding extreme initial transformations.

### Bounded Log-Scale

For affine coupling, the log-scale is bounded:

$$
s = s_{\max} \cdot \tanh(s_{\text{raw}} / s_{\max})
$$

This prevents $\exp(s)$ from exploding or vanishing.

### LOFT Transform

The LOFT (Log-Soft) transform stabilizes high-dimensional flows by modifying the tails:

$$
\text{LOFT}(x; \tau) = \begin{cases}
x & \lvert x\rvert \leq \tau \\
\text{sign}(x) \cdot (\tau + \log(\lvert x\rvert - \tau + 1)) & \lvert x\rvert > \tau
\end{cases}
$$

Transitions from linear (near origin) to logarithmic (in tails), preventing extreme values from causing numerical issues [andrade2021].

## Density Evaluation

### log_prob

Uses the inverse map and change of variables:

```python
z, log_det_inv = inverse(params, x, context)
log_prob = base_dist.log_prob(z) + log_det_inv
```

### sample_and_log_prob

Avoids redundant inverse computation when both samples and densities are needed:

```python
z = base_dist.sample(key, shape)
x, log_det_fwd = forward(params, z, context)
log_prob = base_dist.log_prob(z) - log_det_fwd
```

Note the sign difference: `log_prob` uses `+ log_det_inv` while `sample_and_log_prob` uses `- log_det_fwd`.
Both are correct because $\log\lvert\det \partial z/\partial x\rvert = -\log\lvert\det \partial x/\partial z\rvert$.

See [REFERENCE.md](REFERENCE.md#forwardinverse-convention) for the full sign convention table.

## References

1. Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2017). "Density estimation using Real-NVP." ICLR.

2. Durkan, C., Bekasov, A., Murray, I., and Papamakarios, G. (2019). "Neural Spline Flows." NeurIPS.

3. Perez, E., Strub, F., De Vries, H., Dumoulin, V., and Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.

4. Andrade, D. (2021). "Stable Training of Normalizing Flows for High-dimensional Variational Inference."

5. Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., and Lakshminarayanan, B. (2021). "Normalizing Flows for Probabilistic Modeling and Inference." JMLR.
