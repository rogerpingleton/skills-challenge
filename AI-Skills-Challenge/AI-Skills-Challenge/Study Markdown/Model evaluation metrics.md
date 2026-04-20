# Model evaluation metrics

Model evaluation metrics are how you measure whether a model is actually doing what you want it to do. Choosing the wrong metric is one of the most common ways ML projects go wrong — a model can look great on paper and fail badly in production.

---

## The Core Idea

Every metric answers a specific question about model behavior. The right metric depends on:

- **What type of problem** you're solving (classification, regression, ranking, generation)
- **What failure modes matter most** (false positives vs. false negatives)
- **What the business/product actually cares about**

---

## Classification Metrics

These apply when your model predicts a discrete label (spam/not-spam, fraud/not-fraud, cat/dog/bird).

### The Confusion Matrix

Everything in classification flows from this 2x2 table:

||Predicted Positive|Predicted Negative|
|---|---|---|
|**Actual Positive**|True Positive (TP)|False Negative (FN)|
|**Actual Negative**|False Positive (FP)|True Negative (TN)|

### Key Metrics

**Accuracy** — `(TP + TN) / Total`

- The most intuitive metric, but often misleading.
- _Problem:_ If 99% of transactions are legitimate, a model that predicts "not fraud" every time has 99% accuracy — and is completely useless.

**Precision** — `TP / (TP + FP)`

- "Of everything the model flagged positive, how many actually were?"
- _Use when false positives are costly_ — e.g., spam filters (you don't want to delete real emails).

**Recall (Sensitivity)** — `TP / (TP + FN)`

- "Of all actual positives, how many did the model catch?"
- _Use when false negatives are costly_ — e.g., cancer screening (missing a real case is dangerous).

**F1 Score** — `2 * (Precision * Recall) / (Precision + Recall)`

- The harmonic mean of precision and recall. Use when you need a single number that balances both.
- Preferred over accuracy on **imbalanced datasets**.

**ROC-AUC**

- Plots the true positive rate vs. false positive rate at every decision threshold.
- AUC = 1.0 is perfect; 0.5 is random guessing.
- Good for comparing models independent of threshold, but can be optimistic on imbalanced data.

**PR-AUC (Precision-Recall AUC)**

- Better than ROC-AUC when classes are heavily imbalanced (e.g., fraud detection where 0.1% of cases are fraud).

---

## Regression Metrics

When your model predicts a continuous value (price, temperature, sales volume).

| Metric                        | Formula               | Notes                                     |
| ----------------------------- | --------------------- | ----------------------------------------- |
| **MAE** (Mean Absolute Error) | `mean(\|y - ŷ\|)`     | Robust to outliers; easy to interpret     |
| **MSE** (Mean Squared Error)  | `mean((y - ŷ)²)`      | Penalizes large errors heavily            |
| **RMSE**                      | `sqrt(MSE)`           | Same units as target; most common default |
| **R² (R-squared)**            | `1 - SS_res/SS_tot`   | 1.0 = perfect; 0 = baseline mean model    |
| **MAPE**                      | `mean(\|y - ŷ\| / y)` | Percentage error; breaks when y ≈ 0       |

**As an AI Engineer**, the key judgment call is: _does your use case penalize large errors more than small ones?_ If yes, prefer MSE/RMSE. If errors are roughly equal in cost, use MAE.

---

## Ranking & Recommendation Metrics

Used in search engines, recommenders, and LLM retrieval (RAG pipelines).

- **MRR (Mean Reciprocal Rank):** Average of 1/rank of the first correct result. Good for "find one right answer" tasks.
- **NDCG (Normalized Discounted Cumulative Gain):** Rewards highly relevant results appearing at the top. Standard for search quality.
- **Precision@K / Recall@K:** Precision or recall considering only the top K results.

---

## NLP / Generative Model Metrics

This is increasingly central to AI Engineering today.

|Metric|Used For|Limitation|
|---|---|---|
|**BLEU**|Machine translation|Measures n-gram overlap; poor proxy for quality|
|**ROUGE**|Summarization|Recall-oriented n-gram overlap|
|**Perplexity**|Language model fluency|Doesn't measure factual correctness|
|**BERTScore**|Text generation quality|Uses embeddings; better semantic match|
|**LLM-as-Judge**|Open-ended generation|Scalable but inherits model biases|

In modern LLM/RAG systems, you'll often define **task-specific evals** (e.g., "does the answer contain the correct entity?") rather than relying on generic metrics.

---

## The Precision-Recall Tradeoff

This is a concept you _must_ understand intuitively. Most classifiers output a probability score; you pick a **threshold** to convert it to a label. Adjusting that threshold moves you along a tradeoff curve:

- **Lower threshold** → catch more positives → higher recall, lower precision
- **Higher threshold** → only flag when confident → higher precision, lower recall

A concrete example: a medical diagnosis model. You might deliberately lower the threshold (accept more false alarms) to ensure you never miss a real case. A content moderation model might raise the threshold to avoid incorrectly removing legitimate posts.

---

## What You're Expected to Know as an AI Engineer

### Foundational (must know cold)

- The confusion matrix and all four derived metrics
- Why accuracy fails on imbalanced data, and what to use instead
- MAE vs. RMSE tradeoffs
- How to tune a decision threshold for a business requirement

### Intermediate (know and apply)

- ROC-AUC vs. PR-AUC and when each is appropriate
- Cross-validation (k-fold) for reliable metric estimation
- Train/validation/test split discipline — never evaluate on training data
- Overfitting signals: high train metric, low val metric

### Applied / Production-level

- **Offline vs. online metrics:** A model's offline eval metrics often don't match production behavior. You're expected to connect offline metrics to business KPIs.
- **Metric selection as a design decision:** You should be able to justify _why_ you chose a metric to a stakeholder.
- **Evaluation datasets:** Know how to construct a held-out test set that reflects real-world distribution, and recognize **data leakage** when you see it.
- **LLM evals:** For GenAI work — building eval harnesses, using LLM-as-Judge, and understanding benchmark limitations (e.g., models trained on benchmark data).

---

## A Practical Python Example

```python
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
import numpy as np

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_prob = np.array([0.1, 0.4, 0.8, 0.6, 0.9, 0.35, 0.75, 0.2])
y_pred = (y_prob >= 0.5).astype(int)  # threshold at 0.5

# Full breakdown: precision, recall, F1 per class
print(classification_report(y_true, y_pred))

# AUC metrics
print(f"ROC-AUC:  {roc_auc_score(y_true, y_prob):.3f}")
print(f"PR-AUC:   {average_precision_score(y_true, y_prob):.3f}")

# Confusion matrix
print(confusion_matrix(y_true, y_pred))
```

---

## The Mental Model to Carry

> **Metrics are a proxy for what you actually care about.** Your job as an AI Engineer is to choose metrics that align model behavior with real-world outcomes — and to know when your metrics are lying to you.

The most dangerous situation in ML is a model that scores well on your chosen metric but fails at the actual task. Metric selection, evaluation design, and understanding tradeoffs are where engineering judgment separates strong practitioners from people who just run `model.fit()`.