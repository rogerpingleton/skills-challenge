# Basic linear algebra

## Why Linear Algebra Matters

As an AI Engineer, linear algebra is the _language your models speak_. It is not optional background knowledge — it is the direct mathematical substrate of every model you will build, fine-tune, deploy, or debug. Here is where it shows up concretely:

- **Embeddings** are vectors. When you retrieve from a vector database, you are computing dot products.
- **Neural network layers** are matrix multiplications. A forward pass is a chain of them.
- **Attention mechanisms** (transformers, RAG rerankers) are scaled dot-product operations between query, key, and value matrices.
- **Dimensionality reduction** (PCA, UMAP) uses eigendecomposition and SVD.
- **Fine-tuning techniques** like LoRA work by decomposing weight matrices using low-rank approximations.
- **Context windows and token batches** are represented as tensors with explicit shape semantics.

You do not need to derive proofs. You need to reason fluently about shapes, transformations, and what operations mean geometrically and computationally.

---

## Scalars, Vectors, and Matrices

### Scalars

A scalar is a single number — a magnitude with no direction. In AI, scalars appear as loss values, learning rates, temperature parameters, and similarity scores.

```python
loss = 0.342
temperature = 0.7
learning_rate = 1e-4
```

### Vectors

A vector is an ordered list of numbers. Geometrically, it represents a point or direction in n-dimensional space. In AI, vectors are everywhere:

- A token embedding: a vector of 768, 1024, or 4096 floats
- A document representation in a vector database
- A hidden state in a neural network layer
- A row of features for a tabular ML model

```python
import numpy as np

# A 4-dimensional word embedding (toy example)
token_embedding = np.array([0.21, -0.45, 0.88, 0.12])

# Shape tells you dimensionality
print(token_embedding.shape)  # (4,)
```

**Column vs. row vectors** — by convention, vectors are column vectors (shape `(n, 1)`), but NumPy defaults to 1D arrays (shape `(n,)`). This distinction matters when you multiply matrices. Always check `.shape`.

### Matrices

A matrix is a 2D array of numbers with shape `(rows, columns)`. In AI:

- A **weight matrix** in a neural network: shape `(input_dim, output_dim)`
- A **batch of embeddings**: shape `(batch_size, embedding_dim)`
- An **attention score matrix**: shape `(seq_len, seq_len)`
- An **image**: shape `(height, width)` or `(height, width, channels)`

```python
# A weight matrix mapping 4-dim input to 3-dim output
W = np.array([
    [0.1,  0.4, -0.2],
    [0.3, -0.1,  0.5],
    [-0.2, 0.2,  0.1],
    [0.5,  0.3, -0.4]
])
print(W.shape)  # (4, 3)

# A batch of 3 embeddings, each 4-dimensional
embeddings = np.array([
    [0.21, -0.45, 0.88, 0.12],
    [0.55,  0.10, 0.30, -0.80],
    [-0.11, 0.67, 0.22,  0.45]
])
print(embeddings.shape)  # (3, 4)
```

---

## Matrix Operations

### Transpose

The transpose flips a matrix over its diagonal: rows become columns. If `A` has shape `(m, n)`, then `A.T` has shape `(n, m)`.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)    # (2, 3)
print(A.T.shape)  # (3, 2)
```

You use transposes constantly for:

- Fixing shape mismatches in matrix multiplication
- Computing similarity matrices: `embeddings @ embeddings.T` gives a `(batch, batch)` similarity grid
- Expressing the relationship between weight matrix and its gradient

### Element-wise Operations

These apply a function to each element independently. Shape must match (or be broadcastable).

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.5, 2.5])

print(a + b)   # [1.5, 3.5, 5.5]
print(a * b)   # [0.5, 3.0, 7.5]  — NOT matrix multiplication
print(a ** 2)  # [1.0, 4.0, 9.0]
```

Activation functions (ReLU, sigmoid, GELU) are element-wise. When you apply `torch.relu(x)`, every element is independently clipped to 0 if negative.

### Broadcasting

Broadcasting lets NumPy/PyTorch apply operations across arrays of different (but compatible) shapes without explicit loops. This is critical for efficiently adding biases to batches.

```python
# Add a bias vector (shape 3) to a batch of 5 vectors (shape 5x3)
batch = np.ones((5, 3))
bias  = np.array([0.1, 0.2, 0.3])  # shape (3,)

result = batch + bias  # broadcasts bias across all 5 rows
print(result.shape)    # (5, 3) — works because trailing dims match
```

Broadcasting rule: dimensions align from the right. Sizes must either match or one must be 1.

---

## The Dot Product and Similarity

The dot product is one of the most important operations in all of AI engineering. Given two vectors **a** and **b** of the same length:

```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```

Geometrically, it equals `|a| × |b| × cos(θ)` where θ is the angle between the vectors.

**Key insight:** when both vectors are unit vectors (length 1), the dot product equals `cos(θ)` — the cosine similarity. This is the exact operation used by every embedding-based retrieval system.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Toy embeddings
apple  = np.array([0.9, 0.1, 0.2])
banana = np.array([0.8, 0.2, 0.1])
car    = np.array([0.0, 0.1, 0.9])

print(cosine_similarity(apple, banana))  # ~0.97 — similar (both fruits)
print(cosine_similarity(apple, car))     # ~0.29 — dissimilar
```

This is what vector databases (Pinecone, Weaviate, pgvector) do at scale: for a given query embedding, find the stored embeddings with the highest dot product.

**Attention scores** use the same idea. In a transformer, the raw attention score between query vector **q** and key vector **k** is `q · k`. The full scaled dot-product attention formula is:

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

---

## Matrix Multiplication

Matrix multiplication is the core operation of neural networks. When you call a linear layer, you are computing a matrix multiplication.

For matrices `A` of shape `(m, n)` and `B` of shape `(n, p)`, the result `C = A @ B` has shape `(m, p)`. The inner dimensions must match.

```
(m × n) @ (n × p) = (m × p)
```

```python
# A linear layer: 4 input features → 3 output features
x = np.array([1.0, 0.5, -1.0, 0.8])  # shape (4,) — one input sample
W = np.random.randn(4, 3)              # weight matrix shape (4, 3)
b = np.array([0.1, -0.1, 0.2])        # bias shape (3,)

output = x @ W + b  # shape (3,)
print(output.shape)  # (3,)
```

**Batched matrix multiplication** — in practice, you process entire batches at once:

```python
# Batch of 32 samples, each 4-dimensional
X = np.random.randn(32, 4)   # shape (32, 4)
W = np.random.randn(4, 128)  # weight matrix

out = X @ W   # shape (32, 128) — all 32 samples processed simultaneously
```

**Why shape errors happen** — the most common runtime error in AI engineering is a shape mismatch. Always reason about shapes before multiplying:

```python
# WRONG — dimensions don't align
A = np.random.randn(5, 3)
B = np.random.randn(5, 4)
# A @ B raises ValueError: (3,) != (5,)

# RIGHT — transpose B so inner dims match
C = A @ B.T  # (5,3) @ (3,5) = (5,5) — only if this is what you intend
# OR reshape A/B appropriately for your use case
```

---

## Tensors

A tensor is a generalization of scalars, vectors, and matrices to arbitrary dimensions. In PyTorch and TensorFlow, everything is a tensor.

|Rank|Name|Example shape|AI example|
|---|---|---|---|
|0|Scalar|`()`|Loss value|
|1|Vector|`(768,)`|Single embedding|
|2|Matrix|`(32, 768)`|Batch of embeddings|
|3|Tensor|`(32, 128, 768)`|Batch of sequences with embeddings|
|4|Tensor|`(16, 3, 224, 224)`|Batch of RGB images|

```python
import torch

# A batch of 8 sequences, each 64 tokens long, with 512-dim embeddings
# This is a typical transformer hidden state
hidden_states = torch.randn(8, 64, 512)
print(hidden_states.shape)  # torch.Size([8, 64, 512])

# Access the embedding for the 3rd token of the 1st sequence
token_vec = hidden_states[0, 2, :]
print(token_vec.shape)  # torch.Size([512])
```

**Understanding tensor shapes in transformers** — when you work with a language model, learning to read tensor shapes is essential:

```
(batch_size, seq_len, hidden_dim) — standard transformer hidden state
(batch_size, num_heads, seq_len, head_dim) — after attention head split
(batch_size, seq_len, vocab_size) — logits before softmax
```

---

## Linear Transformations

A linear transformation is what a matrix does to a vector: it stretches, rotates, reflects, and projects it into a new space. Every linear layer in a neural network is a linear transformation.

```python
# This matrix rotates 2D vectors by 90 degrees
R = np.array([[0, -1],
              [1,  0]])

v = np.array([1.0, 0.0])  # pointing right
rotated = R @ v
print(rotated)  # [0. 1.] — now pointing up
```

**Why this matters for AI:** When a neural network learns, it is learning which linear transformation (weight matrix) to apply at each layer, combined with non-linear activations to handle complex patterns. The depth of the network allows compositions of these transformations to approximate arbitrarily complex functions.

**LoRA and low-rank transformations** — Low-Rank Adaptation, a dominant fine-tuning technique, is directly grounded in this concept. Instead of updating a full weight matrix `W` of shape `(d, d)`, LoRA represents the update as:

```
ΔW = A × B   where A is (d, r) and B is (r, d), r << d
```

This dramatically reduces the number of trainable parameters because two small matrices can approximate the change in the large one. Understanding that a rank-r matrix can only transform vectors within an r-dimensional subspace is what makes LoRA make sense.

---

## Norms and Distance

A norm is a measure of a vector's "length" or "magnitude." Different norms have different properties and are used in different contexts.

### L2 Norm (Euclidean Norm)

The standard notion of length. Used in cosine similarity, gradient clipping, weight regularization.

```
||v||₂ = sqrt(v[0]² + v[1]² + ... + v[n]²)
```

```python
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))       # 5.0
print(np.linalg.norm(v, ord=2)) # 5.0 — explicit L2
```

### L1 Norm (Manhattan Norm)

Sum of absolute values. Used in sparse regularization (Lasso), and for distance metrics when outliers are a concern.

```python
print(np.linalg.norm(v, ord=1))  # 7.0 (3 + 4)
```

### Unit Normalization

Dividing a vector by its norm produces a unit vector (length 1). This is essential before computing cosine similarity and is a standard preprocessing step for embeddings.

```python
def normalize(v):
    return v / np.linalg.norm(v)

embedding = np.array([3.0, 4.0, 0.0])
unit_vec  = normalize(embedding)
print(np.linalg.norm(unit_vec))  # 1.0
```

### Gradient Clipping

A direct application of norms in training: if the L2 norm of the gradient vector exceeds a threshold, rescale it. This prevents exploding gradients in deep networks and RNNs.

```python
import torch

# In PyTorch training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# This computes ||all_grads||₂ and scales them down if > 1.0
```

---

## Eigenvalues and Eigenvectors

An eigenvector of a matrix `A` is a special vector that, when multiplied by `A`, only scales — it does not change direction:

```
A × v = λ × v
```

where `v` is the eigenvector and `λ` (lambda) is the corresponding eigenvalue (a scalar).

```python
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [3. 2.]
print(eigenvectors)  # columns are the eigenvectors
```

### Where AI Engineers Encounter This

**Principal Component Analysis (PCA)** — the principal components are the eigenvectors of the covariance matrix. The corresponding eigenvalues tell you how much variance each component explains. PCA is used for dimensionality reduction before clustering, visualization (projecting to 2D), and as a preprocessing step.

```python
from sklearn.decomposition import PCA
import numpy as np

# 100 embeddings of dimension 512 — reduce to 2D for visualization
embeddings = np.random.randn(100, 512)
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

print(reduced.shape)  # (100, 2)
print(pca.explained_variance_ratio_)  # how much variance each component captures
```

**Attention and covariance** — eigenanalysis of attention weight matrices can reveal which directions in embedding space the model attends to most strongly. This is a technique used in model interpretability research.

**Stability and conditioning** — the ratio of the largest to smallest eigenvalue of a matrix (the "condition number") determines how numerically stable operations on that matrix are. Poorly conditioned matrices lead to training instability.

---

## Singular Value Decomposition

SVD decomposes any matrix `A` (shape `m × n`) into three matrices:

```
A = U × Σ × V^T
```

- `U` — shape `(m, m)`, columns are left singular vectors
- `Σ` — shape `(m, n)`, diagonal matrix of singular values (non-negative, descending)
- `V^T` — shape `(n, n)`, rows are right singular vectors

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, sigma, Vt = np.linalg.svd(A)
print(sigma)  # [1.684e+01, 1.068e+00, 1.198e-15] — last ≈ 0, rank-2 matrix
```

### Why SVD Is Critical for AI Engineers

**Low-rank approximation** — keep only the top-k singular values to get the best rank-k approximation of the matrix. This is the mathematical foundation of LoRA, matrix compression, and noise reduction.

```python
k = 2  # keep top-2 components
U_k     = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k    = Vt[:k, :]

A_approx = U_k @ sigma_k @ Vt_k
print(A_approx)  # very close to A for well-structured matrices
```

**LoRA in practice** — when you add LoRA adapters to a model, the weight update `ΔW = A × B` is exactly a rank-r matrix. SVD is how you would analyze what subspace that update occupies. It's also used to initialize LoRA weights to approximate a given target update.

**Recommendation systems** — collaborative filtering via matrix factorization is SVD applied to user-item interaction matrices.

**Embedding compression** — SVD can compress a large embedding matrix by keeping only the most important singular directions, reducing memory and compute without large accuracy loss.

---

## Projections and Orthogonality

A projection maps a vector onto a subspace — it finds the "shadow" of the vector in that direction.

The projection of vector **a** onto vector **b** is:

```
proj_b(a) = (a · b / ||b||²) × b
```

```python
a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])  # x-axis

projection = (np.dot(a, b) / np.dot(b, b)) * b
print(projection)  # [3. 0.] — shadow on the x-axis
```

**Orthogonality** — two vectors are orthogonal if their dot product is zero. Orthogonal vectors are "perpendicular" and carry independent information.

### AI Engineering Applications

**Attention projections** — in a transformer, queries, keys, and values are created by projecting the hidden state through learned weight matrices: `Q = X @ W_Q`. These projections map embeddings into a space where the attention pattern makes semantic sense.

**Orthogonal initialization** — initializing weight matrices to be orthogonal (columns mutually perpendicular, unit length) helps gradient flow in deep networks. PyTorch supports this directly:

```python
linear = torch.nn.Linear(128, 128)
torch.nn.init.orthogonal_(linear.weight)
```

**Gram-Schmidt and QR decomposition** — these procedures orthogonalize a set of vectors, producing a basis where each vector is perpendicular to the others. Used in numerical stability analysis and some optimizer implementations.

**KV-cache and context compression** — recent techniques for compressing the KV-cache (the stored key-value pairs in transformer inference) use projection to lower-dimensional subspaces, reducing memory without losing critical information.

---

## Working with NumPy and PyTorch

Most of your daily linear algebra as an AI engineer happens through these two libraries.

### NumPy — the Foundation

```python
import numpy as np

# Creating matrices
zeros  = np.zeros((3, 4))
ones   = np.ones((3, 4))
eye    = np.eye(4)          # 4×4 identity matrix
rand   = np.random.randn(3, 4)  # Gaussian random

# Key operations
A = np.random.randn(4, 5)
B = np.random.randn(5, 3)

C        = A @ B                      # matrix multiply → (4,3)
A_T      = A.T                        # transpose → (5,4)
norm_A   = np.linalg.norm(A)          # Frobenius norm
trace_sq = np.linalg.matrix_rank(A)  # rank of A
inv_sq   = np.linalg.inv(np.eye(4))  # inverse (square matrices only)
```

### PyTorch — for Models and Gradients

PyTorch extends NumPy-style operations with automatic differentiation and GPU support.

```python
import torch

# Most NumPy patterns transfer directly
A = torch.randn(4, 5)
B = torch.randn(5, 3)

C = A @ B        # matrix multiply
C = torch.mm(A, B)  # equivalent for 2D
C = torch.matmul(A, B)  # general, handles batched and broadcast cases

# Batched matrix multiply — critical for transformer ops
# Query: (batch=8, heads=12, seq=64, head_dim=64)
# Key:   (batch=8, heads=12, head_dim=64, seq=64)
Q = torch.randn(8, 12, 64, 64)
K = torch.randn(8, 12, 64, 64)
scores = torch.matmul(Q, K.transpose(-2, -1))  # → (8, 12, 64, 64)

# Einsum — expressive multi-dimensional operations
# Equivalent to above attention scores
scores2 = torch.einsum('bhqd,bhkd->bhqk', Q, K)

# Norms
l2_norm  = torch.norm(A)
row_norm = torch.norm(A, dim=1)  # norm of each row

# SVD in PyTorch
U, S, Vh = torch.linalg.svd(A)
```

### Einsum — The Power Tool

`einsum` (Einstein summation) is a concise notation for expressing complex tensor operations. It is widely used in transformer implementations and custom model code.

```python
# Dot product of two vectors
a = torch.randn(5)
b = torch.randn(5)
dot = torch.einsum('i,i->', a, b)  # sum over i

# Matrix multiply
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ij,jk->ik', A, B)  # i,k free; j summed

# Batch matrix multiply (common in attention)
Q = torch.randn(2, 3, 4)  # batch=2, seq=3, dim=4
K = torch.randn(2, 3, 4)
scores = torch.einsum('bqd,bkd->bqk', Q, K)  # (2, 3, 3) attention matrix

# Outer product
a = torch.randn(4)
b = torch.randn(5)
outer = torch.einsum('i,j->ij', a, b)  # (4, 5) matrix
```

Reading einsum notation: letters on the left of `->` are the input indices, right side are output indices. Any letter that appears in inputs but not in output is summed over.

---

## Quick Reference

### Shape Rules

|Operation|Input shapes|Output shape|
|---|---|---|
|`A @ B`|`(m,n)` and `(n,p)`|`(m,p)`|
|`A.T`|`(m,n)`|`(n,m)`|
|`a · b` (dot)|`(n,)` and `(n,)`|scalar|
|`np.outer(a,b)`|`(m,)` and `(n,)`|`(m,n)`|
|Batched `@`|`(b,m,n)` and `(b,n,p)`|`(b,m,p)`|

### Key Identities

```
(AB)^T = B^T A^T
||Ax|| ≤ ||A|| · ||x||
A @ np.eye(n) = A
np.linalg.inv(A) @ A = np.eye(n)
```

### Common Operations by AI Task

|Task|Operation|
|---|---|
|Embedding retrieval|Cosine similarity / dot product|
|Linear layer forward|`X @ W + b`|
|Attention scores|`Q @ K.T / sqrt(d_k)`|
|Gradient clipping|L2 norm + rescaling|
|LoRA weight update|`A @ B` (low-rank matrix product)|
|PCA / compression|SVD, keep top-k singular values|
|Embedding normalization|Divide by L2 norm|
|Batch similarity matrix|`embeddings @ embeddings.T`|

---

_This document covers the linear algebra that an AI Engineer will actively use. For deeper mathematical treatment — proofs, abstract vector spaces, spectral theory — see Gilbert Strang's "Introduction to Linear Algebra" or the mathematics appendix in "Deep Learning" (Goodfellow, Bengio, Courville)._