# Basic calculus

## 1. Why Calculus Matters

As an AI Engineer, you will rarely solve calculus problems by hand. But you will constantly work with systems _built on_calculus. Understanding these foundations helps you:

- **Debug training** — understand why a model isn't converging, why loss explodes, or why gradients vanish.
- **Tune hyperparameters** — learning rates, momentum, and optimizers are all interpretations of calculus concepts.
- **Understand optimizers** — Adam, RMSProp, SGD are all calculus-based algorithms.
- **Read research papers** — ML papers are dense with gradient notation, partial derivatives, and loss function math.
- **Design architectures** — activation function choices, normalization, and regularization all trace back to calculus.

At its core, the vast majority of ML training is this single goal: _find the set of model parameters that minimizes a cost function_. Calculus — specifically differential calculus — is the mathematical machinery that makes this possible.

---

## 2. Foundational Prerequisites

Before diving into calculus concepts, make sure these building blocks are solid:

### Functions

A function `f(x)` maps an input `x` to an output `y`. In ML, functions are everywhere:

- A neural network is a massive composite function mapping input data to predictions.
- A loss function maps predictions to a scalar error value.
- An activation function maps a neuron's pre-activation value to its output.

### Limits (Conceptual Understanding Only)

You don't need to compute limits formally, but understand the concept: a limit describes the value a function _approaches_as its input approaches some value. Derivatives are formally defined through limits, so having the intuition is useful when you see notation like `lim(h→0)`.

### Slopes and Rates of Change

A slope tells you how fast something is changing. The derivative is simply a generalized, continuous version of slope. This intuition is everything.

---

## 3. Functions and Their Behavior

### Continuous vs. Differentiable Functions

A function must be **continuous** to be differentiable at a point. Most activation functions are designed to be differentiable almost everywhere (the one exception you'll encounter is **ReLU**, which is not differentiable at exactly `x=0`, but works fine in practice).

### Minima and Maxima

Training a model is an optimization problem. You need to understand:

- **Global minimum** — the lowest possible value of the loss function across all parameter space.
- **Local minimum** — a point lower than its immediate neighbors, but not necessarily the lowest overall.
- **Saddle point** — a point where the gradient is zero but it's neither a minimum nor a maximum.

For neural networks, the loss landscape is high-dimensional and non-convex, meaning gradient descent is _not_ guaranteed to find the global minimum. In practice, local minima in high-dimensional spaces are often nearly as good as the global minimum, which is why deep learning still works well.

### Convexity

A **convex function** is bowl-shaped — gradient descent on a convex function is guaranteed to find the global minimum. Simple models like linear and logistic regression have convex loss functions. Neural networks do not.

```
Convex:      f(x) = x²          → single bowl, one minimum
Non-convex:  f(x) = x⁴ - 2x²   → multiple local minima
```

---

## 4. Derivatives: The Core Tool

### What a Derivative Is

The derivative of `f(x)` at a point `x` tells you the instantaneous rate of change — the slope of the tangent line to the curve at that point.

**Notation you'll see:**

```
f'(x)        — "f prime of x"    (Lagrange notation)
df/dx        — "d f d x"         (Leibniz notation)
∂f/∂x        — partial derivative (when multiple variables exist)
∇f           — gradient          (vector of all partial derivatives)
```

### Geometric Intuition

```
If f'(x) > 0   → function is increasing at x
If f'(x) < 0   → function is decreasing at x
If f'(x) = 0   → function has a flat point (candidate minimum, maximum, or saddle)
```

This is exactly what gradient descent exploits: it moves parameters in the direction where `f'` is negative (downhill).

### The Derivative Formula (for intuition)

```
f'(x) = lim(h→0) [ f(x + h) - f(x) ] / h
```

This is the slope between two points on the curve as the gap between them shrinks to zero. You won't compute this directly — you'll use differentiation rules.

---

## 5. Key Differentiation Rules

These are the rules you need to know. Master these and you can differentiate almost any function you'll encounter in AI engineering.

### Power Rule

```
f(x) = xⁿ  →  f'(x) = n·xⁿ⁻¹

Example:
f(x) = x³   →  f'(x) = 3x²
f(x) = x²   →  f'(x) = 2x
f(x) = x    →  f'(x) = 1
f(x) = 5    →  f'(x) = 0   (constant has zero derivative)
```

### Sum Rule

```
(f + g)' = f' + g'

Example:
f(x) = x³ + x²   →  f'(x) = 3x² + 2x
```

### Product Rule

```
(f·g)' = f'·g + f·g'

Example:
f(x) = x² · sin(x)
f'(x) = 2x · sin(x) + x² · cos(x)
```

### Quotient Rule

```
(f/g)' = (f'·g - f·g') / g²
```

You'll encounter this occasionally, but in AI engineering the chain rule is far more central.

### Chain Rule (Critical — covered in depth in Section 8)

```
If h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)
```

### Exponential and Logarithm Rules

These appear constantly in loss functions and activation functions:

```
d/dx [eˣ]     = eˣ          (the exponential function is its own derivative)
d/dx [aˣ]     = aˣ · ln(a)
d/dx [ln(x)]  = 1/x
d/dx [log_a(x)] = 1 / (x · ln(a))
```

### Trigonometric Rules (for reference)

```
d/dx [sin(x)]  =  cos(x)
d/dx [cos(x)]  = -sin(x)
d/dx [tanh(x)] = 1 - tanh²(x)   ← used in RNN activation functions
```

---

## 6. Partial Derivatives and Multivariable Calculus

### Why This Matters in AI

Neural network models have thousands to billions of parameters. The loss function depends on _all_ of them simultaneously. You can't just take a single derivative — you need a derivative with respect to each parameter independently. This is where partial derivatives come in.

### What a Partial Derivative Is

A partial derivative measures how a function changes when you vary _one_ variable while holding all others constant.

```
f(w₁, w₂) = w₁² + 3·w₁·w₂ + w₂²

∂f/∂w₁ = 2w₁ + 3w₂    (treat w₂ as a constant)
∂f/∂w₂ = 3w₁ + 2w₂    (treat w₁ as a constant)
```

### Notation

```
∂f/∂x   — "partial derivative of f with respect to x"
```

The ∂ symbol (called "del" or "partial") signals that we're doing a partial derivative, not a total one.

### Practical Example: Loss Function

```
Mean Squared Error: L(w, b) = (1/n) · Σ (y_pred - y_true)²
                             = (1/n) · Σ (w·x + b - y_true)²

∂L/∂w = (2/n) · Σ x·(w·x + b - y_true)
∂L/∂b = (2/n) · Σ (w·x + b - y_true)
```

These two partial derivatives tell us exactly how to adjust `w` and `b` to reduce the loss.

---

## 7. Gradients

### What a Gradient Is

The **gradient** is a vector that collects all partial derivatives of a function into one object. If your loss function `L` depends on parameters `θ = [w₁, w₂, ..., wₙ]`, then:

```
∇L = [ ∂L/∂w₁,  ∂L/∂w₂,  ...,  ∂L/∂wₙ ]
```

The gradient is the generalization of the derivative to multiple dimensions.

### The Critical Geometric Meaning

**The gradient points in the direction of steepest ascent.**

This is the most important property in all of ML optimization:

- To _increase_ a function → move in the direction of the gradient.
- To _minimize_ a function (what we want in training) → move in the _opposite_ direction of the gradient (negative gradient).

This is exactly what gradient descent does.

### Gradient Magnitude

The magnitude (length) of the gradient tells you _how steep_ the slope is. Large gradient → steep terrain → large update step. Small gradient → near-flat terrain → small update step.

---

## 8. The Chain Rule

### Why This Is the Most Important Rule in Deep Learning

Every neural network is a **composition of functions** — the output of one layer becomes the input to the next. The chain rule is the calculus tool for differentiating composite functions. It is the mathematical engine behind backpropagation.

### The Basic Form

```
If h(x) = f(g(x)), then:

h'(x) = f'(g(x)) · g'(x)

Or in Leibniz notation, if y = f(u) and u = g(x):
dy/dx = (dy/du) · (du/dx)
```

**Intuition:** "The rate of change of the composite function equals the product of the rates of change of each individual function in the chain."

### Single-Variable Example

```
h(x) = (3x + 1)⁵

Let u = 3x + 1, so h = u⁵

dh/dx = (dh/du) · (du/dx)
      = 5u⁴   ·   3
      = 15(3x + 1)⁴
```

### Multi-Variable Chain Rule (what ML actually uses)

When a function depends on multiple intermediate variables, you sum up the contributions from each path:

```
If L depends on z, and z depends on w through multiple paths:

∂L/∂w = Σ (∂L/∂zᵢ) · (∂zᵢ/∂w)
```

### Neural Network Example (one layer)

```
z = w·x + b          (linear transformation)
a = σ(z)             (activation function, e.g., sigmoid)
L = (a - y)²         (loss)

To find ∂L/∂w:
∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)
       = 2(a-y)  ·  σ'(z)  ·    x
```

Each term in this product is one "link" in the chain. This is precisely how PyTorch and TensorFlow compute gradients internally via `.backward()`.

---

## 9. Gradient Descent: Calculus in Action

### The Algorithm

Gradient descent is the optimization algorithm that uses gradients to iteratively update model parameters toward a minimum:

```
θ = θ - α · ∇L(θ)

Where:
  θ      = model parameters (weights and biases)
  α      = learning rate (step size)
  ∇L(θ)  = gradient of loss with respect to parameters
```

At each step, you compute the gradient and take a small step in the opposite direction.

### The Learning Rate: A Calculus Consequence

The learning rate `α` directly controls how large each parameter update is. This is where your calculus understanding pays off practically:

```
α too large  → overshoot the minimum, loss oscillates or diverges
α too small  → extremely slow convergence, may get stuck
α just right → smooth descent toward minimum
```

Understanding that the gradient points in a specific direction with a specific magnitude helps you reason about why adaptive learning rate methods (like Adam) work — they scale the step size based on the history of gradient magnitudes.

### Variants You'll Use

**Batch Gradient Descent** Compute the gradient over the entire dataset. Accurate but slow for large datasets.

**Stochastic Gradient Descent (SGD)** Compute the gradient on one random sample at a time. Fast but noisy. The noise can actually help escape local minima.

**Mini-batch Gradient Descent** The standard in practice. Use a small batch (typically 32–512 samples) per update. Balances efficiency and stability.

### Python Implementation

```python
import numpy as np

def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    """
    Simple gradient descent for linear regression: y_pred = w * x + b
    Loss = Mean Squared Error
    """
    n = len(y)
    w, b = 0.0, 0.0  # initialize parameters

    for epoch in range(epochs):
        # Forward pass: compute predictions
        y_pred = w * X + b

        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)

        # Compute gradients (partial derivatives of loss w.r.t. w and b)
        dL_dw = (2 / n) * np.sum((y_pred - y) * X)  # ∂L/∂w
        dL_db = (2 / n) * np.sum(y_pred - y)         # ∂L/∂b

        # Update parameters (move opposite to gradient)
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b

# Example usage
np.random.seed(42)
X = np.random.randn(100)
y = 3 * X + 2 + np.random.randn(100) * 0.1   # true: w=3, b=2

w, b = gradient_descent_linear_regression(X, y, learning_rate=0.1, epochs=500)
print(f"\nFinal: w={w:.4f} (true: 3.0), b={b:.4f} (true: 2.0)")
```

---

## 10. Backpropagation: The Chain Rule at Scale

### What Backpropagation Is

Backpropagation is the algorithm that efficiently computes the gradient of the loss function with respect to every weight and bias in a neural network. It is "just" the chain rule applied systematically, layer by layer, from the output back to the input.

### Forward vs. Backward Pass

**Forward pass:** Compute the prediction by passing data through each layer in sequence. Cache intermediate values.

**Backward pass:** Using the chain rule, compute the gradient of the loss with respect to each parameter, working backwards from the output layer to the input layer.

### Why Backward (Not Forward)

Computing gradients from back to front is dramatically more efficient because intermediate gradient values computed in deeper layers can be reused for shallower layers. Without this, you would need to recompute many redundant products.

### Step-by-Step: One Neuron Example

```
Network: x → [linear: z=wx+b] → [sigmoid: a=σ(z)] → [loss: L=(a-y)²]

Forward pass values (given x=2.0, w=0.5, b=0.1, y=1.0):
  z = 0.5 * 2.0 + 0.1 = 1.1
  a = σ(1.1) = 1 / (1 + e^(-1.1)) ≈ 0.7503
  L = (0.7503 - 1.0)² ≈ 0.0623

Backward pass (chain rule):
  ∂L/∂a = 2(a - y)              = 2(0.7503 - 1.0) = -0.4994
  ∂a/∂z = σ(z)(1 - σ(z))       = 0.7503 * 0.2497 ≈ 0.1874
  ∂z/∂w = x                     = 2.0
  ∂z/∂b = 1

  ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w = -0.4994 · 0.1874 · 2.0 ≈ -0.1873
  ∂L/∂b = ∂L/∂a · ∂a/∂z · ∂z/∂b = -0.4994 · 0.1874 · 1.0 ≈ -0.0936

Update (with lr=0.1):
  w_new = 0.5 - 0.1 * (-0.1873) = 0.5187
  b_new = 0.1 - 0.1 * (-0.0936) = 0.1094
```

### Python Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)   # this is ∂σ/∂z — key derivative used in backprop

def backprop_single_neuron(x, y_true, w, b, lr=0.1):
    # --- Forward pass ---
    z = w * x + b
    a = sigmoid(z)
    loss = (a - y_true) ** 2

    # --- Backward pass (chain rule) ---
    dL_da = 2 * (a - y_true)          # ∂L/∂a
    da_dz = sigmoid_derivative(z)     # ∂a/∂z
    dz_dw = x                         # ∂z/∂w
    dz_db = 1                         # ∂z/∂b

    dL_dw = dL_da * da_dz * dz_dw    # chain rule: ∂L/∂w
    dL_db = dL_da * da_dz * dz_db    # chain rule: ∂L/∂b

    # --- Parameter update ---
    w -= lr * dL_dw
    b -= lr * dL_db

    return w, b, loss

# Train for 200 steps
x, y_true = 2.0, 1.0
w, b = 0.5, 0.1

for step in range(200):
    w, b, loss = backprop_single_neuron(x, y_true, w, b)
    if step % 50 == 0:
        print(f"Step {step}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
```

---

## 11. Loss Functions and Their Derivatives

The loss (or cost) function is the function gradient descent is trying to minimize. As an AI Engineer, you must know the derivatives of the standard loss functions.

### Mean Squared Error (MSE) — Regression

```
L = (1/n) · Σ (ŷᵢ - yᵢ)²

∂L/∂ŷᵢ = (2/n) · (ŷᵢ - yᵢ)
```

Used for regression tasks. Penalizes large errors quadratically.

### Binary Cross-Entropy — Binary Classification

```
L = -(1/n) · Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

∂L/∂ŷᵢ = -(yᵢ/ŷᵢ) + (1-yᵢ)/(1-ŷᵢ)
```

Used with sigmoid output for binary classification. The derivative simplifies beautifully when paired with sigmoid, which is by design.

### Categorical Cross-Entropy — Multi-class Classification

```
L = -(1/n) · Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)

∂L/∂ŷᵢₖ = -yᵢₖ / ŷᵢₖ
```

Used with softmax output for multi-class classification.

### Python: Computing Loss and Its Derivative

```python
import numpy as np

def mse_loss(y_pred, y_true):
    n = len(y_true)
    loss = np.mean((y_pred - y_true) ** 2)
    grad = (2 / n) * (y_pred - y_true)   # ∂L/∂ŷ
    return loss, grad

def binary_cross_entropy(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
    n = len(y_true)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    grad = (1 / n) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))
    return loss, grad

# Example
y_pred = np.array([0.9, 0.2, 0.8])
y_true = np.array([1.0, 0.0, 1.0])

loss, grad = mse_loss(y_pred, y_true)
print(f"MSE Loss: {loss:.4f}")
print(f"MSE Gradient: {grad}")
```

---

## 12. Activation Functions and Their Derivatives

Activation functions introduce non-linearity. Their derivatives are essential for backpropagation — you must know both the function _and_ its derivative.

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)
σ'(x) = σ(x) · (1 - σ(x))
```

**AI Engineering note:** Prone to **vanishing gradients** in deep networks. When `|x|` is large, `σ'(x)` approaches 0, which causes gradients to shrink to near-zero as they propagate backward through many layers, making early layers very slow to train.

### Tanh

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
tanh'(x) = 1 - tanh²(x)
```

Outputs in range (-1, 1). Also suffers from vanishing gradients, but is zero-centered, which can be advantageous.

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
ReLU'(x) = 1  if x > 0
            0  if x ≤ 0
```

**AI Engineering note:** Default choice for most hidden layers. Simple derivative eliminates the vanishing gradient problem for positive values. The **dying ReLU problem** occurs when neurons get stuck outputting 0 permanently if their pre-activation is always negative.

### Leaky ReLU

```
LeakyReLU(x) = x     if x > 0
               α·x   if x ≤ 0  (where α is a small value, e.g., 0.01)

LeakyReLU'(x) = 1    if x > 0
                α     if x ≤ 0
```

Solves the dying ReLU problem by allowing a small gradient when the input is negative.

### Softmax (for output layer, multi-class)

```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ

∂softmax(xᵢ)/∂xⱼ = softmax(xᵢ) · (δᵢⱼ - softmax(xⱼ))
```

where `δᵢⱼ` is the Kronecker delta. This has a more complex Jacobian, but frameworks handle this automatically.

### Python: Activation Functions and Derivatives

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)  # 1 where x>0, 0 elsewhere

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Visualizing the vanishing gradient problem
x = np.array([-5.0, -2.0, -0.5, 0.5, 2.0, 5.0])
print("x              :", x)
print("sigmoid(x)     :", np.round(sigmoid(x), 4))
print("sigmoid'(x)    :", np.round(sigmoid_derivative(x), 4))
# Notice: at x=-5 and x=5, derivative is nearly 0
# This is the vanishing gradient problem

print("\nrelu(x)        :", relu(x))
print("relu'(x)       :", relu_derivative(x))
# ReLU derivative is either 0 or 1 — no vanishing for positive inputs
```

---

## 13. Integrals: Where They Show Up

You'll encounter integral calculus less frequently than differential calculus as an AI Engineer, but it appears in several important areas.

### Probability Distributions

Probability density functions (PDFs) require integration to compute probabilities over ranges:

```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

And the requirement that all probabilities sum to 1:

```
∫₋∞^∞ f(x) dx = 1
```

### Expected Values

The expected value of a function under a distribution uses an integral:

```
E[f(X)] = ∫ f(x) · p(x) dx
```

This appears in variational inference, policy gradients in reinforcement learning, and the derivation of cross-entropy loss.

### KL Divergence

KL divergence, which measures how different two probability distributions are, is defined as an integral (or sum for discrete distributions):

```
KL(P || Q) = ∫ P(x) · log(P(x) / Q(x)) dx
```

Used extensively in VAEs (Variational Autoencoders) and other generative models.

### Attention Mechanism (Continuous Analogy)

The softmax attention in transformers can be thought of as a discrete approximation of a continuous integral over an attention distribution.

---

## 14. Jacobian and Hessian Matrices

### Jacobian Matrix

The Jacobian generalizes the gradient to vector-valued functions. If a function maps a vector input to a vector output `f: Rⁿ → Rᵐ`, the Jacobian is an m×n matrix of all partial derivatives:

```
J = [ ∂fᵢ/∂xⱼ ]

For f: R² → R²:
J = | ∂f₁/∂x₁   ∂f₁/∂x₂ |
    | ∂f₂/∂x₁   ∂f₂/∂x₂ |
```

**In AI Engineering:** The Jacobian appears when computing gradients through layers that transform vectors (like fully connected and convolutional layers). Frameworks like PyTorch use Jacobian-vector products internally for efficient backpropagation.

### Hessian Matrix

The Hessian is a matrix of second-order partial derivatives — it captures the curvature of the loss surface:

```
H = [ ∂²L/∂wᵢ∂wⱼ ]
```

**In AI Engineering:**

- A positive definite Hessian at a point confirms a local minimum.
- Second-order optimizers (like L-BFGS and Newton's method) use the Hessian to take better-informed steps than first-order methods like SGD.
- For large models, computing the full Hessian is computationally infeasible (an n×n matrix for n parameters), so approximations are used.

---

## 15. Practical Python Examples

### Example 1: Numerical Gradients for Debugging

When building custom loss functions or debugging, **numerical differentiation** lets you verify your analytical gradients:

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    Compute gradient numerically using central difference.
    Useful for verifying analytical gradients.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Define a loss function
def my_loss(params):
    w, b = params
    x_data = np.array([1.0, 2.0, 3.0])
    y_data = np.array([2.0, 4.0, 6.0])
    y_pred = w * x_data + b
    return np.mean((y_pred - y_data) ** 2)

params = np.array([1.5, 0.5])

# Numerical gradient (ground truth)
num_grad = numerical_gradient(my_loss, params)

# Analytical gradient (our calculation)
w, b = params
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])
y_pred = w * x_data + b
residuals = y_pred - y_data
analytical_dw = (2 / len(y_data)) * np.sum(residuals * x_data)
analytical_db = (2 / len(y_data)) * np.sum(residuals)

print(f"Numerical gradient:   dw={num_grad[0]:.6f}, db={num_grad[1]:.6f}")
print(f"Analytical gradient:  dw={analytical_dw:.6f}, db={analytical_db:.6f}")
# These should match closely — this is called a "gradient check"
```

### Example 2: Understanding Gradient Descent Convergence

```python
import numpy as np

def visualize_gradient_descent():
    """
    Demonstrates how learning rate affects convergence.
    Minimizing f(x) = x² (global minimum at x=0).
    """
    def f(x): return x ** 2
    def df(x): return 2 * x   # derivative of x²

    x_start = 5.0
    learning_rates = [0.01, 0.1, 0.9, 1.1]  # last two cause issues

    for lr in learning_rates:
        x = x_start
        history = [x]
        for _ in range(20):
            x = x - lr * df(x)   # gradient descent update
            history.append(x)
            if abs(x) > 100:     # diverged
                break
        status = "converged" if abs(x) < 0.01 else ("diverged" if abs(x) > 10 else "slow")
        print(f"lr={lr}: final x={x:.4f} ({status})")

visualize_gradient_descent()
```

### Example 3: PyTorch Autograd — Calculus Under the Hood

PyTorch uses **automatic differentiation** (autograd) to compute derivatives without you writing them by hand:

```python
import torch

# Define parameters with gradient tracking
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

# Input and target
x = torch.tensor(2.0)
y_true = torch.tensor(1.0)

# Forward pass — PyTorch records all operations for the chain rule
z = w * x + b                        # linear transform
a = torch.sigmoid(z)                 # activation
loss = (a - y_true) ** 2             # MSE loss

# Backward pass — autograd applies the chain rule automatically
loss.backward()   # computes ∂loss/∂w and ∂loss/∂b

print(f"Loss: {loss.item():.4f}")
print(f"∂loss/∂w: {w.grad.item():.4f}")   # gradient w.r.t. weight
print(f"∂loss/∂b: {b.grad.item():.4f}")   # gradient w.r.t. bias

# PyTorch's autograd is doing exactly what we computed manually in Section 10
# The chain rule is applied layer by layer through the computational graph
```

### Example 4: Diagnosing Vanishing Gradients

```python
import numpy as np

def simulate_gradient_flow(depth=10, activation='sigmoid'):
    """
    Simulate how gradients shrink through many sigmoid layers.
    Illustrates the vanishing gradient problem.
    """
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
    def relu_grad(x): return float(x > 0)

    gradient = 1.0   # start with gradient of 1 at the output
    print(f"\nGradient flow through {depth} layers ({activation} activation):")
    for layer in range(depth):
        pre_activation = np.random.randn()   # random pre-activation value

        if activation == 'sigmoid':
            local_grad = sigmoid_grad(pre_activation)
        else:  # relu
            local_grad = relu_grad(pre_activation)

        gradient *= local_grad
        print(f"  Layer {layer+1}: local_grad={local_grad:.4f}, cumulative={gradient:.6f}")

simulate_gradient_flow(depth=8, activation='sigmoid')
# Gradient vanishes quickly — each sigmoid layer multiplies by a value < 0.25
# max of σ'(x) = 0.25 at x=0

simulate_gradient_flow(depth=8, activation='relu')
# ReLU maintains gradient magnitude (1.0 for active neurons)
```

---

## 16. Summary: The AI Engineer's Calculus Cheatsheet

|Concept|What It Is|Where It Shows Up|
|---|---|---|
|**Derivative**|Rate of change of a function|Loss sensitivity to parameters|
|**Partial Derivative**|Rate of change w.r.t. one variable|Multi-parameter models|
|**Gradient (∇L)**|Vector of all partial derivatives|Gradient descent update step|
|**Chain Rule**|Derivative of composed functions|Backpropagation in every layer|
|**Gradient Descent**|Iterative minimization via gradient|Training any ML model|
|**Learning Rate (α)**|Step size in gradient descent|Most important hyperparameter|
|**Backpropagation**|Efficient chain rule application|Training neural networks|
|**Jacobian**|Matrix of partial derivatives (vector→vector)|Layer-wise gradient computation|
|**Hessian**|Second-order derivatives (curvature)|Advanced optimizers (L-BFGS)|
|**Vanishing Gradient**|Gradients near zero in deep networks|Sigmoid/tanh in deep networks|
|**Exploding Gradient**|Gradients become very large|RNNs, very deep networks|
|**Convexity**|Bowl-shaped loss surface|Linear/logistic regression|
|**Saddle Point**|Zero gradient, neither min nor max|Common in neural network loss landscapes|

### Key Identities to Memorize

```
Power:         d/dx [xⁿ]    = n·xⁿ⁻¹
Exponential:   d/dx [eˣ]    = eˣ
Logarithm:     d/dx [ln x]  = 1/x
Sigmoid:       σ'(x)        = σ(x)(1 - σ(x))
Tanh:          tanh'(x)     = 1 - tanh²(x)
ReLU:          ReLU'(x)     = 1 if x>0, else 0
Chain Rule:    d/dx[f(g(x))] = f'(g(x)) · g'(x)
```

### The Mental Model That Ties It All Together

```
Training a model = Finding the minimum of a loss function

How to find the minimum:
  1. Compute the gradient (which direction is uphill?)
  2. Step in the opposite direction (downhill)
  3. Repeat

How to compute the gradient across many layers:
  → Chain rule, applied backwards through each layer
  → This is backpropagation

What can go wrong:
  → Gradient too small (vanishing) → early layers don't learn
  → Gradient too large (exploding) → training diverges
  → Learning rate too big → overshoot minimum
  → Learning rate too small → too slow, may get stuck
```

---

_This document covers the calculus foundations most relevant to practical AI Engineering. The concepts of linear algebra (matrix operations, eigenvectors), probability theory, and statistics are equally essential and complement this material in the broader mathematical foundation of AI._