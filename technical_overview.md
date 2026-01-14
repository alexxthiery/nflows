# Technical Overview

This document provides an in-depth explanation of the nflows library implementation.

## 1. Normalizing Flows: Mathematical Foundation

### 1.1 Change of Variables

A normalizing flow defines a probability distribution by transforming samples from a simple base distribution through an invertible map.

Let $z \sim p_0(z)$ be a sample from the base distribution (typically standard Gaussian), and let $f: \mathbb{R}^d \to \mathbb{R}^d$ be an invertible transformation. The transformed variable $x = f(z)$ has density:

$$p(x) = p_0(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

Taking logarithms:

$$\log p(x) = \log p_0(z) + \log \left| \det \frac{\partial z}{\partial x} \right|$$

where $z = f^{-1}(x)$.

### 1.2 Jacobian Determinant

The computational bottleneck is computing $\det \frac{\partial f}{\partial z}$, which is $O(d^3)$ for general transformations. The key insight of modern normalizing flows is to design transformations with tractable Jacobians.

---

## 2. Coupling Layers

### 2.1 Affine Coupling (RealNVP)

The affine coupling layer splits the input into two parts using a binary mask $m \in \{0,1\}^d$:

**Forward transformation** ($z \to x$):
$$x_{\text{masked}} = z \odot m$$
$$x_{\text{transformed}} = z \odot (1-m) \cdot \exp(s(z \odot m)) + t(z \odot m)$$
$$x = x_{\text{masked}} + x_{\text{transformed}}$$

where $s, t: \mathbb{R}^d \to \mathbb{R}^d$ are the scale and shift functions (outputs of a neural network conditioner).

**Jacobian**: The Jacobian is triangular, so its determinant is the product of diagonal elements:
$$\log \left| \det \frac{\partial x}{\partial z} \right| = \sum_{i: m_i=0} s_i(z \odot m)$$

**Inverse transformation** ($x \to z$):
$$z_{\text{masked}} = x \odot m$$
$$z_{\text{transformed}} = (x \odot (1-m) - t(x \odot m)) \cdot \exp(-s(x \odot m))$$
$$z = z_{\text{masked}} + z_{\text{transformed}}$$

The inverse is computed analytically without iterative methods.

### 2.2 Spline Coupling

Instead of affine transformations, spline coupling uses monotonic rational-quadratic splines (Durkan et al., 2019).

For each dimension $i$ where $m_i = 0$, the transformation is:
$$x_i = \text{RQSpline}(z_i; \theta_i)$$

where $\theta_i = (w_i, h_i, d_i)$ are the spline parameters (bin widths, heights, and knot derivatives) predicted by the conditioner network.

**Properties**:
- Monotonic and invertible by construction
- $C^1$ continuous
- More expressive than affine (can model multimodal conditionals)
- Identity outside $[-B, B]$ (linear tails)

**Parameterization** (per dimension, $K$ bins):
- Widths: $K$ parameters → softmax → min_width floor → scale to $2B$
- Heights: $K$ parameters → softmax → min_height floor → scale to $2B$
- Derivatives: $K-1$ internal derivatives (boundaries fixed to 1)

**Derivative parameterization**: Internal knot derivatives are bounded via sigmoid:
$$d = d_{\min} + (d_{\max} - d_{\min}) \cdot \sigma(d_{\text{raw}})$$

where $d_{\min}$ and $d_{\max}$ are hyperparameters (defaults: 0.01 and 10.0). This ensures derivatives stay in a stable range while remaining learnable.

Total: $3K - 1$ parameters per transformed dimension.

---

## 3. Composing Transformations

### 3.1 Composite Transform

Multiple transforms are composed sequentially:
$$f = f_n \circ f_{n-1} \circ \cdots \circ f_1$$

The log-determinant accumulates:
$$\log \left| \det \frac{\partial f}{\partial z} \right| = \sum_{i=1}^{n} \log \left| \det \frac{\partial f_i}{\partial z_{i-1}} \right|$$

### 3.2 Mask Alternation

A single coupling layer only transforms dimensions where $m_i = 0$. To ensure all dimensions are transformed, we alternate masks:

- Layer 1: mask = $[1, 0, 1, 0, \ldots]$
- Layer 2: mask = $[0, 1, 0, 1, \ldots]$
- ...

### 3.3 Permutations

Fixed permutations between coupling layers improve mixing.

**Implementation**: For a permutation $\pi$, the forward transform is:
$$y_i = x_{\pi(i)}$$

The Jacobian is a permutation matrix with determinant $\pm 1$, so $\log|\det| = 0$.

In this library, a simple reverse permutation $\pi(i) = d - 1 - i$ is used between coupling layers when enabled.

### 3.4 Linear Transform

An optional global linear transformation with efficient LU-style parameterization.

**Parameterization**: We factor an invertible matrix $W \in \mathbb{R}^{d \times d}$ as:
$$W = L \cdot T$$

where:
- $L = \text{tril}(L_{\text{raw}}, k=-1) + I$ — unit-diagonal lower triangular
- $T = \text{triu}(U_{\text{raw}}, k=1) + \text{diag}(\exp(\log s))$ — upper triangular with positive diagonal

**Parameters**:
- `lower`: $(d, d)$ raw lower-triangular entries
- `upper`: $(d, d)$ raw upper-triangular entries
- `log_diag`: $(d,)$ log of diagonal entries

**Complexity**:
- Forward/inverse: $O(d^2)$ via triangular solves
- Log-determinant: $O(d)$ — just sum of `log_diag`

$$\log|\det W| = \sum_{i=1}^{d} \log s_i = \sum_{i=1}^{d} \text{log\_diag}_i$$

**Initialization**: Identity transform ($L = I$, $T = I$) via zero initialization of all raw parameters.

---

## 4. Conditional Normalizing Flows

Conditional flows model $p(x | c)$ where $c$ is a conditioning variable.

### 4.1 Mathematical Formulation

The base distribution remains unconditional: $z \sim p_0(z)$.

The transformation becomes context-dependent: $x = f(z; c)$.

For fixed $c$, the map $f(\cdot; c)$ must be invertible in $z$.

**Conditional density**:
$$\log p(x | c) = \log p_0(z) + \log \left| \det \frac{\partial z}{\partial x} \right|$$

where $z = f^{-1}(x; c)$.

The Jacobian is taken with respect to $x$ (or $z$), **not** $c$. The conditioning variable affects the transformation parameters but does not participate in the change of variables.

### 4.2 Conditioning Strategy: Concatenation

The simplest approach concatenates context to the conditioner input:

$$[s, t] = \text{MLP}([z_{\text{masked}}, c])$$

**Implementation**:
```
input_dim = dim + context_dim
MLP: R^{input_dim} → R^{2·dim}  (for affine coupling)
```

**Advantages**:
- Simple to implement
- Works well for low-dimensional context
- No architectural changes needed

**Disadvantages**:
- Input dimension grows with context dimension
- Context influence may be diluted in deep networks

### 4.3 Context Broadcasting

Context can be:
- **Per-sample**: shape $(B, d_c)$ — different context per batch element
- **Shared**: shape $(d_c,)$ — same context for all samples

The implementation broadcasts shared context to match batch dimensions:
```python
if context.ndim < x.ndim:
    context = broadcast_to(context, x.shape[:-1] + (context.shape[-1],))
```

---

## 5. Numerical Stability

### 5.1 Zero Initialization

Conditioner networks are initialized so the final layer outputs zero:
$$W_{\text{out}} = 0, \quad b_{\text{out}} = 0$$

This means:
- Affine coupling: $s = 0, t = 0$ → identity transform
- Spline coupling: uniform bins, unit derivatives → near-identity

**Benefit**: The flow starts as identity, avoiding extreme initial transformations that can cause training instability.

### 5.2 Bounded Log-Scale

For affine coupling, we bound the log-scale:
$$s = s_{\max} \cdot \tanh(s_{\text{raw}} / s_{\max})$$

This prevents $\exp(s)$ from exploding or vanishing.

### 5.3 LOFT Transform

The LOFT (Log-Soft) transform stabilizes high-dimensional flows by modifying the tails:

$$\text{LOFT}(x; \tau) = \begin{cases}
x & |x| \leq \tau \\
\text{sign}(x) \cdot (\tau + \log(|x| - \tau + 1)) & |x| > \tau
\end{cases}$$

**Effect**: Transitions from linear (near origin) to logarithmic (in tails), preventing extreme values from causing numerical issues.

---

## 6. Implementation Details

### 6.1 Parameter Structure

```python
params = {
    "base": {
        # Empty for StandardNormal
        # {"loc": ..., "log_scale": ...} for DiagNormal
    },
    "transform": [
        # Optional LinearTransform (if use_linear=True):
        # {"lower": (d,d), "upper": (d,d), "log_diag": (d,)},
        {"mlp": {...}},      # AffineCoupling or SplineCoupling params
        {},                   # Permutation (no params, if use_permutation=True)
        {"mlp": {...}},      # AffineCoupling or SplineCoupling params
        {},                   # LoftTransform (no params)
    ]
}
```

### 6.2 Forward vs Inverse Convention

| Method | Direction | Log-det sign |
|--------|-----------|--------------|
| `forward(z)` | $z \to x$ | $+\log|\det \partial x/\partial z|$ |
| `inverse(x)` | $x \to z$ | $+\log|\det \partial z/\partial x|$ |

Note: `inverse` returns the log-det of the inverse map, which is the negative of the forward log-det.

### 6.3 Density Evaluation

```python
def log_prob(params, x, context=None):
    z, log_det_inv = inverse(params, x, context)
    return base_dist.log_prob(z) + log_det_inv
```

### 6.4 Efficient Sampling with Log-Prob

When sampling, we can compute log-prob without calling inverse:

```python
def sample_and_log_prob(params, key, shape, context=None):
    z = base_dist.sample(key, shape)
    x, log_det_fwd = forward(params, z, context)
    log_prob = base_dist.log_prob(z) - log_det_fwd
    return x, log_prob
```

This avoids redundant computation when both samples and their densities are needed.

---

## 7. References

1. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). "Density estimation using Real-NVP." ICLR.

2. Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). "Neural Spline Flows." NeurIPS.

3. Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.

4. Andrade, D. (2021). "Stable Training of Normalizing Flows for High-dimensional Variational Inference."

5. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). "Normalizing Flows for Probabilistic Modeling and Inference." JMLR.
