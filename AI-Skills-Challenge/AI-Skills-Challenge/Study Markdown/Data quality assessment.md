
# Data quality assessment

## Why Data Quality Is the Foundation of AI

As Andrew Ng famously observed, if 80% of the work in ML is data preparation, then ensuring data quality is the most critical task for a machine learning team. This isn't hyperbole — roughly 85% of all AI projects fail, and many AI professionals flag poor data quality as a major source of concern: without high-quality data, even the most sophisticated models and AI agents can go awry.

Machine learning models "learn" directly from the datasets they are given, making high-quality, governed data an essential precondition for sustained AI success. The classic principle applies: garbage in, garbage out.

---

## The Core Dimensions of Data Quality

Data quality assessment is not a single check — it is a multi-dimensional evaluation. For AI/ML systems, the traditional dimensions are extended and deepened beyond what was needed in conventional data warehousing.

### 1. Accuracy

In AI systems, accuracy extends beyond whether data values correctly represent real-world entities. It also encompasses how label noise (incorrectly or ambiguously labeled training examples), measurement error, and proxy variables affect model training.

**Example (Python):** Detecting label noise using a cross-validation confidence approach:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

def detect_label_noise(X, y, threshold=0.3):
    """
    Uses a model's predicted probabilities to flag potentially mislabeled samples.
    Samples where the model is very confident in a different class are flagged.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')
    
    flagged = []
    for i, (true_label, probs) in enumerate(zip(y, proba)):
        predicted_label = np.argmax(probs)
        confidence_in_true = probs[true_label]
        if predicted_label != true_label and confidence_in_true < threshold:
            flagged.append({
                'index': i,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence_in_true': confidence_in_true
            })
    return flagged
```

### 2. Completeness

For AI data quality, completeness extends beyond checking whether required fields or records are missing — it includes whether the data sufficiently covers the full range of cases the model is expected to encounter, such as edge cases, rare events, and minority populations. Gaps in coverage can result in brittle models that perform well on average but fail in underrepresented scenarios, increasing both fairness and operational risks.

**Example:** Auditing class coverage and missing value rates:

```python
import pandas as pd

def completeness_report(df: pd.DataFrame, label_col: str) -> dict:
    total = len(df)
    
    missing_rates = (df.isnull().sum() / total * 100).to_dict()
    class_distribution = df[label_col].value_counts(normalize=True).to_dict()
    
    # Flag underrepresented classes (less than 1% of data)
    underrepresented = {k: v for k, v in class_distribution.items() if v < 0.01}
    
    return {
        'total_rows': total,
        'missing_rates_pct': missing_rates,
        'class_distribution': class_distribution,
        'underrepresented_classes': underrepresented
    }
```

### 3. Consistency

Consistency ensures that data follows a standard format and structure, facilitating efficient processing and analysis. Inconsistent data can lead to confusion and misinterpretation, impairing the performance of AI systems.

This means checking for conflicting values across records, schema drift across time-partitioned data, and format inconsistencies (e.g., dates stored as strings in multiple formats).

**Example:** Detecting schema drift between training and serving data:

```python
def check_schema_consistency(train_df: pd.DataFrame, serve_df: pd.DataFrame) -> list:
    issues = []
    
    for col in train_df.columns:
        if col not in serve_df.columns:
            issues.append(f"Missing column in serving data: {col}")
            continue
        
        if train_df[col].dtype != serve_df[col].dtype:
            issues.append(
                f"Type mismatch in '{col}': "
                f"train={train_df[col].dtype}, serve={serve_df[col].dtype}"
            )
        
        # Check for unexpected new categories in categoricals
        if train_df[col].dtype == 'object':
            train_cats = set(train_df[col].dropna().unique())
            serve_cats = set(serve_df[col].dropna().unique())
            new_cats = serve_cats - train_cats
            if new_cats:
                issues.append(f"New unseen categories in '{col}': {new_cats}")
    
    return issues
```

### 4. Relevance

Assessing data relevance in AI means determining whether each feature and example provides information that supports the system's intended function — including whether data improves predictive performance, supports robustness across different conditions, reduces sensitivity to noise or spurious correlations, and facilitates downstream interpretability.

Irrelevant features add noise and can cause models to overfit to spurious patterns.

### 5. Timeliness / Freshness

Freshness loses value naturally as data ages. For production ML systems, models trained on stale data will degrade as the real-world distribution shifts. This is especially critical for RAG pipelines, recommendation systems, and fraud detection.

### 6. Diversity and Balance

Eight key dimensions for data perception in deep learning and LLMs include quality, difficulty, diversity, uncertainty, value, redundancy, balance, and others. Diversity and balance are often underweighted until a model fails in production on a demographic group or scenario that wasn't well represented in training.

**Example:** Measuring distributional coverage with entropy:

```python
from scipy.stats import entropy
import numpy as np

def distribution_balance_score(labels: list) -> dict:
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    
    uniform_entropy = np.log(len(values))  # Max possible entropy
    actual_entropy = entropy(probs)
    balance_ratio = actual_entropy / uniform_entropy if uniform_entropy > 0 else 0
    
    return {
        'num_classes': len(values),
        'class_probs': dict(zip(values.tolist(), probs.tolist())),
        'entropy': round(actual_entropy, 4),
        'balance_ratio': round(balance_ratio, 4),  # 1.0 = perfectly balanced
        'is_imbalanced': balance_ratio < 0.7
    }
```

---

## Automated Quality Assessment Methods

Different types of models can be used for quality filtering: N-gram based classifiers (like fastText) are the simplest approach and excel in efficiency; BERT-style classifiers offer better quality through Transformer-based architectures; LLMs provide the most sophisticated assessment but have significant computational requirements and are best suited for smaller-scale applications like fine-tuning datasets; and reward models are specialized for evaluating conversational data quality but are also computationally expensive. The optimal selection should consider both dataset scale and available computational resources.

### Heuristic / Rule-Based Checks

Fast, cheap, and highly scalable. These run first in any pipeline:

```python
import re

def heuristic_quality_check(text: str) -> dict:
    """Quick heuristic checks for text data quality."""
    issues = []
    
    # Too short or too long
    words = text.split()
    if len(words) < 5:
        issues.append("too_short")
    if len(words) > 2000:
        issues.append("too_long")
    
    # Repetition: a sign of low-quality web content
    unique_words = set(words)
    dedup_ratio = len(unique_words) / len(words) if words else 0
    if dedup_ratio < 0.3:
        issues.append("highly_repetitive")
    
    # Gibberish / excessive punctuation
    punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if punct_ratio > 0.3:
        issues.append("excessive_punctuation")
    
    # Boilerplate patterns
    boilerplate_patterns = [
        r"click here to",
        r"terms and conditions",
        r"cookie policy",
        r"all rights reserved"
    ]
    for pattern in boilerplate_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(f"boilerplate:{pattern}")
    
    return {
        'passes': len(issues) == 0,
        'issues': issues,
        'word_count': len(words),
        'dedup_ratio': round(dedup_ratio, 3)
    }
```

### Model-Based Quality Scoring

NVIDIA's Nemotron-4 reward model assesses data quality through five key attributes: Helpfulness, Correctness, Coherence, Complexity, and Verbosity. By setting appropriate thresholds for these attribute scores, the filtering process ensures only high-quality synthetic data is retained.

For LLM fine-tuning datasets specifically, you want to score instruction-response pairs on these dimensions before including them in training.

### LLM-as-Judge

Pipelines that follow a sequence of generation, ranking, and selection use an auxiliary LLM as a judge to evaluate candidate outputs against specific rubrics. For instance, the UltraFeedback framework evaluates outputs based on criteria such as helpfulness and honesty, effectively shifting the control mechanism from manual prompt engineering to scalable, automated preference filtering.

```python
import anthropic

def llm_quality_judge(instruction: str, response: str) -> dict:
    """Use an LLM to evaluate instruction-response pair quality."""
    client = anthropic.Anthropic()
    
    prompt = f"""Evaluate this instruction-response pair on a scale of 1-5 for each dimension.
Return JSON only with keys: helpfulness, correctness, coherence, complexity, verbosity, overall.

Instruction: {instruction}
Response: {response}

Return ONLY valid JSON, no explanation."""
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    return json.loads(message.content[0].text)
```

---

## The Data Trust Score

A Data Trust Score quantifies dataset reliability using a weighted combination of quality signals. Freshness loses value naturally as data ages, and lineage ensures a dataset cannot appear more reliable than its inputs. Trust scoring creates measurable, auditable data health indicators, and different applications prioritize different quality attributes.

Here is a practical implementation:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DataTrustScore:
    accuracy_score: float       # 0-1: label correctness estimates
    completeness_score: float   # 0-1: coverage of expected cases
    consistency_score: float    # 0-1: schema/format uniformity
    freshness_score: float      # 0-1: recency relative to domain
    diversity_score: float      # 0-1: balance across classes/groups
    
    # Weights can be tuned per use-case
    weights: dict = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'accuracy': 0.30,
                'completeness': 0.25,
                'consistency': 0.15,
                'freshness': 0.15,
                'diversity': 0.15
            }
    
    @property
    def composite_score(self) -> float:
        w = self.weights
        return (
            self.accuracy_score * w['accuracy'] +
            self.completeness_score * w['completeness'] +
            self.consistency_score * w['consistency'] +
            self.freshness_score * w['freshness'] +
            self.diversity_score * w['diversity']
        )
    
    @property
    def grade(self) -> str:
        s = self.composite_score
        if s >= 0.85: return "A — Production Ready"
        if s >= 0.70: return "B — Needs Minor Fixes"
        if s >= 0.55: return "C — Needs Significant Work"
        return "D — Not Suitable for Training"


def freshness_score(last_updated: datetime, decay_days: int = 90) -> float:
    """Exponential decay freshness score."""
    age_days = (datetime.now() - last_updated).days
    return float(np.exp(-age_days / decay_days))
```

---

## Deduplication: The Hidden Quality Problem

Duplicate data causes models to memorize rather than generalize. Researchers have formally identified redundancy as one of eight key data perception dimensions, noting that high redundancy in training data can hurt both model performance and generalization.

There are two levels of deduplication you must handle:

**Exact deduplication** — hash-based, fast:

```python
import hashlib

def exact_dedup(texts: list[str]) -> list[str]:
    seen = set()
    unique = []
    for text in texts:
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(text)
    return unique
```

**Fuzzy / semantic deduplication** — using MinHash or embedding similarity for near-duplicates:

```python
from datasketch import MinHash, MinHashLSH

def build_minhash_index(texts: list[str], threshold: float = 0.85):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in text.lower().split():
            m.update(word.encode('utf8'))
        lsh.insert(f"doc_{i}", m)
        minhashes[f"doc_{i}"] = m
    
    return lsh, minhashes
```

---

## Bias Detection and Fairness Auditing

Datasets may underrepresent certain demographics, languages, or use cases, leading to models that perform poorly on underrepresented groups. Bias in training data is one of the most consequential quality issues.

StereoSet detects stereotype reinforcement across gender, race, and religion through masked language modeling tasks, while BBQ probes intersectional fairness through binary and multiple-choice QA pairs. Real-world scenarios often require continuous bias monitoring, prompting the emergence of frameworks like DynamicBiasEval, which introduces drift-based fairness diagnostics across model updates.

For tabular/structured datasets, you can audit at the feature level:

```python
def fairness_audit(df: pd.DataFrame, sensitive_cols: list, label_col: str) -> dict:
    """Audit label distribution across sensitive attribute groups."""
    results = {}
    for col in sensitive_cols:
        group_stats = df.groupby(col)[label_col].value_counts(normalize=True).unstack()
        # Compute max disparity ratio across groups
        col_max = group_stats.max()
        col_min = group_stats.min()
        disparity = (col_max / col_min.replace(0, float('nan'))).max()
        results[col] = {
            'distribution': group_stats.to_dict(),
            'max_disparity_ratio': round(disparity, 3),
            'flag': disparity > 2.0  # Flag if any group is 2x over/under-represented
        }
    return results
```

---

## Annotation Quality: Inter-Annotator Agreement

For supervised datasets, label quality depends on who labeled the data and how consistently. Annotator agreement and consistency prove difficult to maintain across large evaluation efforts. Different annotators may interpret evaluation criteria differently, leading to inconsistent labels that undermine dataset quality. Measuring inter-annotator agreement through metrics like Cohen's kappa helps identify consistency issues.

```python
from sklearn.metrics import cohen_kappa_score

def annotation_quality_report(annotations: dict[str, list]) -> dict:
    """
    annotations: dict mapping annotator_id -> list of labels for each sample
    """
    annotators = list(annotations.keys())
    kappas = {}
    
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a1, a2 = annotators[i], annotators[j]
            kappa = cohen_kappa_score(annotations[a1], annotations[a2])
            kappas[f"{a1}_vs_{a2}"] = round(kappa, 4)
    
    avg_kappa = sum(kappas.values()) / len(kappas) if kappas else 0
    
    return {
        'pairwise_kappas': kappas,
        'average_kappa': round(avg_kappa, 4),
        # Interpretation: <0.4 poor, 0.4-0.6 moderate, 0.6-0.8 good, >0.8 excellent
        'quality': 'good' if avg_kappa >= 0.6 else 'needs_review'
    }
```

---

## Continuous Data Quality Monitoring

Quality assessment is not a one-time gate — it must run continuously as data and the world change.

Solutions like Monte Carlo and WhyLabs are at the forefront of observability, offering real-time monitoring of data quality, lineage, and drift, thereby ensuring the accuracy and reliability of AI models.

Evidently is an open-source Python library for data scientists and ML engineers to evaluate, test, and monitor ML models and data quality. It works with tabular data, text, and embeddings in NLP and LLM tasks. It supports building custom reports from individual metrics and its key strength is monitoring ability throughout the ML lifecycle by tracking model features over time.

A practical drift detection example using population stability index (PSI):

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index — measures feature distribution shift.
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change, investigate
    PSI > 0.2: Major shift, model likely degraded
    """
    expected_perc, bin_edges = np.histogram(expected, bins=bins, density=True)
    actual_perc, _ = np.histogram(actual, bins=bin_edges, density=True)
    
    # Avoid log(0)
    expected_perc = np.clip(expected_perc, 1e-6, None)
    actual_perc = np.clip(actual_perc, 1e-6, None)
    
    psi_value = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return round(float(psi_value), 5)
```

---

## AI-Augmented Quality Assessment

The field is rapidly moving beyond hand-written rules. AI-augmented data quality engineering shifts data quality from deterministic, Boolean checks to probabilistic, generative, and self-learning systems. The result is a self-healing data ecosystem that adapts to concept drift and scales alongside growing enterprise complexity.

Key advances include:

- **Semantic type inference**: Tools like Sherlock (MIT) analyze over 1,500 statistical, lexical, and embedding features to classify column semantic types with extremely high accuracy, examining distribution patterns, character entropy, word embeddings, and contextual behavior — not just simple rules like "five digits = ZIP code."
    
- **Generative remediation**: Generative AI allows automated remediation, not just detection. Instead of engineers writing correction rules, AI learns how the data should behave. Jellyfish is an instruction-tuned LLM for data cleaning and transformation tasks, reducing hallucinations by integrating domain constraints during inference.
    
- **Reinforcement learning for pipeline ordering**: ReClean frames data cleaning as a sequential decision process where an RL agent decides the optimal next cleaning action, receiving rewards based on downstream ML performance rather than arbitrary quality rules.
    

---

## The Data Quality Pipeline: Putting It All Together

Here is how the stages fit together in an AI engineering context:

```
Raw Data
    ↓
[1] Heuristic Filters (fast, cheap: length, boilerplate, encoding)
    ↓
[2] Exact + Fuzzy Deduplication
    ↓
[3] Schema & Consistency Checks
    ↓
[4] Completeness Audit (missing values, coverage gaps)
    ↓
[5] Bias / Fairness Audit (sensitive attributes, class balance)
    ↓
[6] Label Quality Check (noise detection, inter-annotator agreement)
    ↓
[7] Model-Based Quality Scoring (classifier, reward model, or LLM-as-judge)
    ↓
[8] Data Trust Score — Pass/Fail gate for training
    ↓
[9] Continuous Monitoring (PSI, drift detection) post-deployment
```

Four foundational practices for improving and sustaining AI data quality throughout this lifecycle are: data profiling and exploration early in the lifecycle; robust data validation processes; continuous data quality monitoring as data and usage patterns evolve; and tracking data lineage so that provenance is fully understood at every stage of the pipeline.

---

## Key Tools Reference (Python Ecosystem)

|Tool|Purpose|
|---|---|
|`pandas-profiling` / `ydata-profiling`|Fast EDA and data profiling|
|`great_expectations`|Declarative data validation and test suites|
|`evidently`|Drift detection, data quality reports for ML|
|`whylogs`|Statistical logging for data pipelines|
|`cleanlab`|Label noise detection and correction|
|`datasketch`|MinHash-based fuzzy deduplication|
|`scipy.stats`|Statistical divergence (KL, Jensen-Shannon, entropy)|
|`sklearn.metrics`|Cohen's kappa, confusion matrices|
|`Detoxify`|Toxicity scoring for text datasets|

---

## Summary

Data quality assessment in AI engineering is a layered discipline spanning accuracy, completeness, consistency, relevance, freshness, balance, and safety. It requires a combination of fast heuristic rules for scale, model-based scoring for nuance, statistical tools for drift, and human judgment for edge cases. Since most GenAI models depend upon large datasets to function, data engineers are needed to process and structure data so that it's clean, labeled, and relevant — ensuring reliable AI outputs. Building a repeatable, automated quality pipeline is not overhead — it is the core of any AI system that will survive contact with the real world.