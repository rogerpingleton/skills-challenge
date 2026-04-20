
# Annotation guidelines creation

## What Annotation Guidelines Are and Why They Matter

Annotation guidelines are the authoritative reference document that tells human (or automated) labelers _what_ to label, _how_ to label it, and _what each label means_. They are the contract between your data requirements and the people executing the labeling. Every data annotation initiative should begin with a clear question: what decision do we want this model to make? Without a clear use case, labeling efforts become unfocused and wasteful.

The goal of creating annotation guidelines is twofold: to act as reference documentation to learn the specific use case, and to serve as a lookup tool annotators can scan for answers to specific questions.

Poor guidelines are one of the most common root causes of dataset failure. At least 30% of generative AI projects will be abandoned after proof of concept by end of 2025, due to poor data quality, inadequate risk controls, or unclear business value, according to Gartner.

---

## The Anatomy of a Well-Structured Guideline Document

A guideline document is not a bullet list of labels. It is a structured artifact with distinct sections. Here is the standard structure you should follow.

### 1. Project Context and Objectives

Begin by grounding the annotator in the _why_. Define challenges your product addresses for customers, explain adverse effects if those challenges go unaddressed, outline how your product addresses customer pain points, and spell out the role high-quality annotation plays in achieving your goals. Annotators who understand the purpose behind their tasks perform significantly better.

### 2. Annotator Assumptions

Be explicit about what your labelers already know versus what they don't. Annotators are tech-savvy but not machine learning engineers. Many understand concepts like Intersection over Union, but the vast majority will best understand requirements written in plain language. Never assume domain knowledge you haven't explicitly conveyed.

### 3. General Guidelines (Task-Level)

Start with guidelines that apply to all classes and labels to be created. Good practice is to outline: what annotation types are relevant (e.g., instance segmentation with masks), what objects to label and not label, and what to do in cases of ambiguous scenarios or uncertainty independent of object class.

### 4. Class-Specific Guidelines

Then go into detail for each class: provide a 1–2 sentence explanation of each class, visual examples of the class, and examples of correct vs. incorrect labels.

### 5. Edge Cases and Decision Trees

This is where most guidelines fail. Edge cases need explicit treatment — don't leave annotators to improvise. Highlight edge cases and errors to minimize initial mistakes. Clearly communicate the evaluation criteria to annotators, preventing potential issues during reviews.

### 6. Gold Standard Examples

Summarize the guidelines by providing examples of a "gold standard" to assist in understanding complex tasks. These aren't just illustrative — they become your calibration benchmark.

### 7. Version History and Change Log

Implement version control for guidelines to adapt to the ML project lifecycle. Guidelines drift is a real problem in long-running projects. Treat your annotation guideline document the same way you treat code — version it, diff it, and document why changes were made.

---

## Quality Mechanics: Inter-Annotator Agreement (IAA)

IAA is the quantitative backbone of annotation quality. It measures how consistently different annotators apply the same guidelines to the same data.

Measuring agreement between annotators is the most direct way to assess label quality. Metrics like Cohen's Kappa and Krippendorff's Alpha quantify how consistently different annotators apply the same guidelines. Low agreement scores indicate ambiguous instructions, insufficient training, or genuinely subjective tasks that require revised rubrics.

Even with detailed guidelines, two qualified annotators will often disagree on the same example. This is not a failure of the annotators — it is a property of human judgment. Disagreement arises from interpretation differences (annotators apply thresholds differently) and genuinely ambiguous cases.

In practice:

- **Cohen's Kappa** is used for two annotators on categorical labels (target: κ > 0.7 for good agreement, κ > 0.8 for strong agreement)
- **Krippendorff's Alpha** handles multiple annotators, missing data, and ordinal/continuous scales
- **Fleiss' Kappa** is used for multiple annotators on categorical data

Rather than treating disagreement as noise to be eliminated, modern annotation pipelines can measure, use, and control disagreement to produce more reliable, auditable, and model-ready datasets.

Here is a Python implementation of both key IAA metrics:

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff  # pip install krippendorff

# Cohen's Kappa - two annotators
annotator_1 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
annotator_2 = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's Kappa: {kappa:.3f}")
# κ < 0.4: poor, 0.4-0.6: moderate, 0.6-0.8: good, > 0.8: excellent

# Krippendorff's Alpha - multiple annotators, handles missing data (np.nan)
# Shape: (n_annotators, n_items)
reliability_data = np.array([
    [1, 0, 1, 1, 0, 1, np.nan, 0, 1, 1],  # Annotator A
    [1, 0, 1, 0, 0, 1,      1, 0, 1, 1],  # Annotator B
    [1, 1, 1, 1, 0, 0,      1, 0, 1, 0],  # Annotator C
])

alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
print(f"Krippendorff's Alpha: {alpha:.3f}")
# α > 0.8 is reliable; 0.667–0.8 is acceptable for tentative conclusions
```

When IAA is low, the root cause is almost always a guideline problem — ambiguous class definitions, missing edge case coverage, or conflicting decision rules.

---

## The Pilot Annotation Round

Before full-scale labeling, always run a calibration pilot. Conduct pilot annotation rounds before full-scale labeling begins to catch inconsistencies early. Quality assurance mechanisms such as spot checks, inter-annotator agreement metrics, and gold-standard benchmarks should be woven into daily operations.

A typical pilot workflow:

1. Select ~100–200 samples covering the full range of difficulty
2. Have 3+ annotators label the same samples independently
3. Calculate IAA scores
4. Conduct a calibration session — review every disagreement
5. Update guidelines based on what actually confused annotators
6. Re-run pilot until IAA reaches target threshold
7. Only then scale to the full dataset

---

## Multi-Level Annotation Schemes

For LLM-related tasks (RLHF, instruction tuning, preference datasets), guidelines need to capture multiple dimensions simultaneously. Combine surface-level tagging (e.g., sentiment, correctness) with deeper assessments (e.g., reasoning quality, factual accuracy) to capture the full spectrum of model performance.

A practical rubric for response quality annotation might look like:

```python
# Example multi-dimensional annotation schema for RLHF preference data
annotation_schema = {
    "response_id": str,
    "dimensions": {
        "factual_accuracy": {
            "scale": [1, 2, 3, 4, 5],
            "definition": "Is the information correct and verifiable?",
            "anchors": {
                1: "Contains clear factual errors",
                3: "Mostly accurate with minor issues",
                5: "Fully accurate and verifiable"
            }
        },
        "completeness": {
            "scale": [1, 2, 3, 4, 5],
            "definition": "Does the response fully address the prompt?",
        },
        "reasoning_quality": {
            "scale": [1, 2, 3, 4, 5],
            "definition": "Is the reasoning sound and well-structured?",
        },
        "safety": {
            "scale": ["safe", "borderline", "unsafe"],
            "definition": "Does the response contain harmful content?",
        }
    },
    "overall_preference": {
        "type": "ranking",
        "note": "Rank responses A/B/C from best to worst"
    },
    "flags": ["factual_error", "off_topic", "harmful", "incomplete"],
    "annotator_confidence": [1, 2, 3]  # Low/Medium/High
}
```

---

## Governance: Version Control and Lineage

Annotated datasets evolve through iterations, and tracking those changes is essential for reproducibility. Every model trained on a given dataset should be traceable back to the specific data version, guidelines version, and annotator performance metrics that produced it. Treat annotation guidelines, tool configurations, and metadata schemas as living artifacts maintained alongside code and model documentation.

In practice, store your guidelines in Git alongside your dataset code and use DVC (Data Version Control) or similar tools to link dataset snapshots to specific guideline versions.

---

## Automating Annotation Guidelines

There are two distinct automation opportunities here: automating _labeling itself_, and automating _guideline improvement_. Both are active areas in 2025.

### 1. LLM-Assisted Pre-Labeling

Using LLMs to generate draft annotations before human review is the most effective way to reduce labeling costs without sacrificing quality. The model proposes labels for straightforward cases, and human annotators focus their attention on uncertain or complex instances.

LLMs can assist with pre-labeling or draft annotations, but they should not serve as the sole source of labels. Models inherit biases from training data, exhibit overconfidence, and struggle with implicit or latent concepts. Human review remains essential for safety-critical and domain-specific labeling tasks.

```python
import anthropic
import json

client = anthropic.Anthropic()

def auto_prelabel_sentiment(texts: list[str], guidelines: str) -> list[dict]:
    """
    Use Claude to pre-label a batch of texts for sentiment.
    Returns structured labels for human review.
    """
    results = []
    
    for text in texts:
        prompt = f"""You are an annotation assistant. Follow these guidelines exactly:

{guidelines}

Annotate the following text. Respond ONLY with valid JSON, no other text:
{{
  "label": "positive|negative|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "flag_for_review": true|false,
  "flag_reason": "reason if flagged, else null"
}}

Text to annotate:
{text}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        annotation = json.loads(raw)
        annotation["source_text"] = text
        annotation["prelabeled_by"] = "claude-sonnet-4-6"
        results.append(annotation)
    
    return results

# Example usage
guidelines = """
Sentiment Guidelines v1.2
- POSITIVE: text expresses satisfaction, praise, or enthusiasm
- NEGATIVE: text expresses dissatisfaction, criticism, or frustration  
- NEUTRAL: factual, ambiguous, or mixed sentiment
- Flag for review if: sarcasm is possible, sentiment is mixed across sentences,
  or confidence < 0.7
"""

texts = [
    "This product completely changed how I work. Absolutely love it.",
    "Arrived two days late and the packaging was damaged.",
    "The software updated to version 3.2 last Tuesday."
]

labels = auto_prelabel_sentiment(texts, guidelines)
for l in labels:
    print(f"[{l['label'].upper()} | conf={l['confidence']}] {l['source_text'][:50]}...")
    if l['flag_for_review']:
        print(f"  ⚠ Flagged: {l['flag_reason']}")
```

### 2. Active Learning for Selective Annotation

Active learning algorithms identify the most informative unlabeled examples and route them to human annotators first. Rather than labeling randomly, you label the samples where the model is most uncertain — dramatically reducing annotation volume needed.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def uncertainty_sampling(
    model,
    unlabeled_embeddings: np.ndarray,
    n_samples: int = 50
) -> np.ndarray:
    """
    Select the most uncertain samples for human annotation.
    Uses least-confidence sampling strategy.
    """
    # Get predicted probabilities across all classes
    proba = model.predict_proba(unlabeled_embeddings)
    
    # Least confidence: 1 - max class probability
    uncertainty_scores = 1 - proba.max(axis=1)
    
    # Return indices of most uncertain samples
    top_uncertain_idx = np.argsort(uncertainty_scores)[::-1][:n_samples]
    return top_uncertain_idx, uncertainty_scores[top_uncertain_idx]

# Usage in an active learning loop
# 1. Start with small labeled seed set
# 2. Train initial model
# 3. Score all unlabeled data by uncertainty
# 4. Send top-N uncertain samples to human annotators
# 5. Add newly labeled samples to training set
# 6. Retrain and repeat
```

### 3. Automating Guideline _Improvement_ with LLMs

This is a cutting-edge technique from a 2025 ACL paper. LLMs can help produce guidelines of high quality — improving inter-annotator agreement from 0.593 to 0.84 on the WNUT-17 entity recognition benchmark — while being faster and cheaper than using crowdsource workers to iterate guidelines.

The core loop: give an LLM your current guideline draft and a sample of disagreements, ask it to identify ambiguities, and generate a revised guideline. Then measure IAA again.

```python
import anthropic
import json

client = anthropic.Anthropic()

def improve_guidelines_with_llm(
    current_guidelines: str,
    disagreements: list[dict],
    task_description: str
) -> str:
    """
    Use Claude to analyze annotation disagreements and suggest
    concrete improvements to the guidelines.
    
    disagreements: list of {"text": str, "labels": [str, str, ...], "annotators": [str, ...]}
    """
    disagreement_examples = "\n".join([
        f"- Text: '{d['text']}'\n  Labels given: {d['labels']}"
        for d in disagreements[:10]  # Cap at 10 for token budget
    ])
    
    prompt = f"""You are an expert in dataset engineering and annotation guideline design.

Task: {task_description}

Current annotation guidelines:
---
{current_guidelines}
---

The following examples caused disagreement among annotators (different labels assigned):
{disagreement_examples}

Analyze these disagreements and:
1. Identify the root cause of each disagreement (ambiguous definition, missing edge case, conflicting rules)
2. Propose specific, concrete guideline revisions that would resolve each ambiguity
3. Add any missing edge case rules
4. Ensure revisions do not introduce new ambiguities

Return ONLY the fully revised guidelines document. Do not include explanations outside the document itself.
Use the same structure as the input guidelines."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Example disagreements from a pilot round
disagreements = [
    {
        "text": "Great product but shipping took forever",
        "labels": ["positive", "negative", "neutral"],
        "annotators": ["A1", "A2", "A3"]
    },
    {
        "text": "Not bad for the price",
        "labels": ["positive", "neutral"],
        "annotators": ["A1", "A2"]
    }
]

current_guidelines = """
Sentiment Labels:
- POSITIVE: expresses satisfaction or praise
- NEGATIVE: expresses dissatisfaction or criticism
- NEUTRAL: factual or ambiguous
"""

revised = improve_guidelines_with_llm(
    current_guidelines,
    disagreements,
    task_description="Classify customer product reviews by sentiment"
)
print(revised)
```

### 4. LLM-as-Judge for Quality Assurance

Beyond labeling, LLMs can be used to _audit_ completed annotations — flagging likely errors before they enter your training set.

```python
def audit_annotations_with_llm(
    annotations: list[dict],
    guidelines: str,
    sample_size: int = 100
) -> list[dict]:
    """
    Use an LLM to QA a random sample of completed annotations.
    Returns a list of potential errors with explanations.
    """
    import random
    sample = random.sample(annotations, min(sample_size, len(annotations)))
    flagged = []

    for item in sample:
        prompt = f"""You are a data quality auditor. Given these annotation guidelines:

{guidelines}

Review this completed annotation:
Text: "{item['text']}"
Assigned label: "{item['label']}"
Annotator reasoning: "{item.get('reasoning', 'none provided')}"

Does the label correctly follow the guidelines? Respond with JSON only:
{{
  "correct": true|false,
  "confidence": 0.0-1.0,
  "issue": "description of problem if incorrect, else null"
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = json.loads(response.content[0].text)
        if not result["correct"] and result["confidence"] > 0.8:
            flagged.append({**item, "audit_issue": result["issue"]})
    
    return flagged
```

---

## The Tiered Human+Automation Pipeline

Organizations can adopt a tiered approach: use automation to handle clear-cut, high-volume cases such as transcribing clean audio or tagging common objects. Route complex or ambiguous data to expert annotators for manual review. Automation should be viewed not as a replacement for human intelligence but as a force multiplier — machines handling scale, humans ensuring meaning.

A production-ready tiered pipeline looks like:

```
Raw Data
    │
    ▼
LLM Pre-Labeler (auto-label high-confidence cases)
    │
    ├─── Confidence > 0.9 ──► Auto-accept queue (spot-check 5%)
    │
    ├─── 0.7–0.9 ──────────► Human review queue
    │
    └─── < 0.7 or flagged ──► Expert annotator queue
                                    │
                              Disagreements ──► Guideline Improvement Loop
```

LLM-assisted workflows reduce annotation time 40–70% when humans verify low-confidence predictions and edge cases.

---

## Key Principles Summary

To recap what every AI Engineer needs to internalize about annotation guidelines:

**Design phase:** Start from model objective → define label taxonomy → write general rules → write class-specific rules → enumerate edge cases → provide gold standard examples → version the document.

**Calibration phase:** Always run a pilot. Measure IAA with Cohen's Kappa or Krippendorff's Alpha. Hold a calibration session on every disagreement. Update guidelines. Re-pilot.

**Scale phase:** Use LLM pre-labeling for speed, active learning for efficiency, tiered routing for cost control, and LLM-as-judge for QA.

**Governance:** Track every model back to the specific data version, guideline version, and annotator performance metrics that produced it. Documentation is part of governance.

When defining guidelines, consider frequently overlooked aspects: managing sensitive data, the role of annotator bias and subjective interpretation (demographics and personal views can subtly influence judgments in nuanced tasks like toxicity detection), and clarifying class overlap by providing distinct examples to distinguish similar categories.

The annotation guideline is not a one-time document — it is a living engineering artifact that evolves with your data, your model, and the disagreements you discover along the way.