# Training - validation - test splits

The three-way data split is one of the most fundamental concepts in ML — it's the mechanism by which you ensure a model actually _learns_ generalizable patterns rather than memorizing your dataset.

---

## The Core Problem Being Solved

When you train a model, it optimizes its parameters to minimize loss on the data it sees. Without held-out data, you have no way of knowing whether the model has _learned_ or simply _memorized_. The split strategy is your defense against this.

---

## The Three Splits

### 🟦 Training Set (~60–80% of data)

This is the data the model **directly learns from**. Weights/parameters are updated based on this data during backpropagation (or equivalent). Everything the model "knows" comes from here.

**What can go wrong:** If you only use this, your model will overfit — achieving near-perfect training accuracy while failing on new data.

### 🟨 Validation Set (~10–20% of data)

This is held out during training but used **by you, the engineer**, to make decisions:

- Tuning hyperparameters (learning rate, depth, regularization)
- Choosing when to stop training (early stopping)
- Comparing model architectures

The key insight is that every time you look at validation metrics and make a decision, you're indirectly "leaking" information about the validation set into your model. This is why you need a third split.

### 🟥 Test Set (~10–20% of data)

This is your **final, untouched benchmark** — used exactly once, after all training and tuning decisions are finalized. It gives you an honest estimate of real-world performance. If you evaluate on the test set repeatedly and adjust based on it, it becomes a second validation set and loses its integrity.

---

## How They Work Together

```
Raw Data
   │
   ├──► Train Set ──► Model learns weights
   │
   ├──► Validation Set ──► You tune hyperparameters & architecture
   │         ▲                        │
   │         └────────────────────────┘  (iterative loop)
   │
   └──► Test Set ──► Final honest evaluation (once, at the end)
```

---

## Python Example

```python
from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.array(...), np.array(...)  # your features and labels

# Step 1: carve out the test set first — lock it away
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Step 2: split the remainder into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    # 0.176 of 85% ≈ 15% of total, giving a ~70/15/15 split
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

> **Note:** Always split _before_ any preprocessing (scaling, encoding). Fit your scalers/encoders on `X_train` only, then `transform` val and test. Doing it the other way is **data leakage**.

---

## What You're Expected to Know as an AI Engineer

### Fundamentals

- **Why** the splits exist and what failure mode each one prevents (overfitting, optimistic bias)
- The difference between **model parameters** (learned from train) and **hyperparameters** (tuned on val)
- **Data leakage** — the subtle ways information from val/test can contaminate training, and how to prevent it

### Intermediate Concepts

- **Stratified splitting** — preserving class balance across splits, critical for imbalanced datasets
- **Cross-validation (k-fold)** — when your dataset is small, rotating the validation fold to get a more reliable estimate
- **Group/time-based splits** — when data points aren't i.i.d. (e.g., multiple records per patient, time series data), random splits are wrong; you must split by group or time boundary

### Advanced / Production Concerns

- **Distribution shift** — train/val/test may not reflect production data distribution; monitoring this is an ongoing job
- **Temporal splits for time-series** — never randomly shuffle time-series data; always use a chronological cutoff
- **Holdout strategy for LLMs/fine-tuning** — the same principles apply, but contamination risks are higher given how much data LLMs have already seen during pretraining
- **Benchmark leakage** — a real-world problem where test sets become "solved" over time because the community optimizes against them (e.g., ImageNet, GLUE)

---

## Common Pitfalls to Avoid

|Mistake|Consequence|
|---|---|
|Preprocessing before splitting|Data leakage — val/test statistics bleed into training|
|Tuning hyperparameters on test set|Optimistically biased final metrics|
|Random split on time-series data|Future data leaks into the past; model appears better than it is|
|Too small a test set|High variance in your final metric estimate|
|Ignoring class imbalance in splits|Skewed splits misrepresent true performance|

---

The mental model to internalize: **train set teaches, validation set guides decisions, test set tells the truth.** Any deviation from keeping those roles strictly separate undermines the validity of your evaluation.