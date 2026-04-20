# Overfitting and underfitting

These are two of the most fundamental concepts in machine learning — understanding them is core to your role as an AI Engineer.

---

## The Core Idea: The Bias-Variance Tradeoff

At the heart of both problems is a tension between two types of error:

- **Bias** — error from wrong assumptions in the model (too simple)
- **Variance** — error from sensitivity to small fluctuations in training data (too complex)

A well-trained model balances both. Overfitting and underfitting are what happen when that balance breaks.

---

## Underfitting (High Bias)

Underfitting occurs when a model is **too simple** to capture the underlying patterns in the data. It performs poorly on _both_training data and unseen data.

**Analogy:** A student who barely studied and fails both the practice test and the real exam.

**Example in Python (linear model on non-linear data):**

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Data has a quadratic pattern
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Underfitting: forcing a straight line onto curved data
model = LinearRegression()
model.fit(X, y)
print(model.score(X, y))  # Low R² — poor fit
```

**Signs of underfitting:**

- High training loss _and_ high validation loss
- Model is too constrained (e.g., linear model on non-linear data)
- Low complexity relative to the problem

---

## Overfitting (High Variance)

Overfitting occurs when a model **memorizes the training data** — including its noise — rather than learning the generalizable pattern. It performs well on training data but poorly on new, unseen data.

**Analogy:** A student who memorized every practice exam answer verbatim but can't handle any new question phrasing.

**Example in Python:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Overfitting: no depth limit allows the tree to memorize training data
model = DecisionTreeClassifier(max_depth=None)
model.fit(X_train, y_train)

print(f"Train accuracy: {model.score(X_train, y_train):.2f}")  # ~1.00
print(f"Test accuracy:  {model.score(X_test, y_test):.2f}")    # Much lower
```

**Signs of overfitting:**

- Very low training loss, but high validation loss (the "gap" is the red flag)
- Model has too many parameters relative to training samples
- Training accuracy >> Test accuracy

---

## Visualizing the Problem

```
Loss
  |
  |  Underfitting zone     Sweet spot     Overfitting zone
  |  ~~~~~~~~~~~~~~~~|___________________|~~~~~~~~~~~~~~~~~~
  |                        /‾‾‾‾ Validation loss
  |                   ____/
  |              ____/
  |  ___________/                         Training loss ___
  |____________________________________________ Model Complexity →
```

The goal is to find the model complexity where validation loss is minimized.

---

## What You're Expected to Know as an AI Engineer

### 1. Detection

- Always split data into **train / validation / test** sets (or use cross-validation)
- Monitor the **loss curves** during training — a growing gap between train and val loss is the hallmark of overfitting
- Use metrics like accuracy, F1, RMSE on held-out data — never trust training metrics alone

### 2. Fixing Underfitting

|Technique|Why it helps|
|---|---|
|Increase model complexity|More capacity to learn patterns|
|Add more/better features|Give the model more signal|
|Reduce regularization|Less constraint on the model|
|Train longer|Let the model converge more fully|

### 3. Fixing Overfitting

|Technique|Why it helps|
|---|---|
|**Regularization** (L1/L2)|Penalizes large weights, smooths the model|
|**Dropout** (neural nets)|Randomly disables neurons during training|
|**Early stopping**|Stop training when val loss starts rising|
|**More training data**|Harder to memorize a larger, diverse dataset|
|**Data augmentation**|Artificially increases dataset diversity|
|**Reduce model complexity**|Fewer parameters = less room to overfit|
|**Cross-validation**|More robust evaluation of generalization|

### 4. Regularization in Practice (Python)

```python
from sklearn.linear_model import Ridge, Lasso

# L2 regularization (Ridge) — penalizes large coefficients
ridge = Ridge(alpha=1.0)

# L1 regularization (Lasso) — drives some coefficients to zero (feature selection)
lasso = Lasso(alpha=0.1)
```

### 5. The Deep Learning Context

In neural networks, overfitting is especially common with large models and small datasets. Key tools you'd use:

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.4)  # Dropout regularization
        self.fc2 = nn.Linear(64, 10)
```

You'd also be expected to know about **weight decay** (L2 in optimizers), **batch normalization**, and **learning rate scheduling** as they all interact with generalization.

### 6. LLMs and Foundation Models

In modern AI engineering, you'll also encounter these concepts at a higher level:

- **Fine-tuning** a pretrained model on a small dataset is extremely prone to overfitting — techniques like **LoRA**, **freezing layers**, and **small learning rates** are standard mitigations
- **Prompt engineering / RAG** can sometimes substitute for fine-tuning precisely _to avoid_ overfitting on limited data

---

## Quick Mental Model to Keep

> If your model **can't even fit the training data** → underfitting → make it more powerful. If your model **fits training data perfectly but fails on new data** → overfitting → constrain it or get more data.

These aren't just theoretical concepts — you'll be debugging them constantly in real training runs.