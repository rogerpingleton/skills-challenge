# Feedback system design

### A Comprehensive Guide to User Feedback Integration

---

## 1. Introduction

In AI engineering, a **feedback system** is the architectural machinery that captures, stores, processes, and applies signals from users or automated evaluators to continuously improve an AI model's behavior. Rather than treating a deployed model as a static artifact, feedback systems create a living loop between real-world usage and model improvement.

This is critical because:

- AI models degrade over time as user needs, language, and context drift.
- Static training data cannot anticipate all edge cases encountered in production.
- Human values, preferences, and goals are too nuanced for simple reward functions to encode at training time.

Modern feedback systems are at the heart of techniques like **Reinforcement Learning from Human Feedback (RLHF)**, **Direct Preference Optimization (DPO)**, and **Online Iterative RLHF** — the same techniques used to align large-scale production models like Claude and GPT-4.

---

## 2. Core Concept: The Feedback Loop Architecture

A feedback system in AI engineering is composed of four fundamental stages:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Collect    │────▶│    Store     │────▶│   Process    │────▶│    Apply     │
│   Feedback   │     │  & Annotate  │     │  & Analyze   │     │  to Model    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ▲                                                                │
       └────────────────────────────────────────────────────────────────┘
                              Continuous Improvement Loop
```

**Collect** — Capture explicit signals (thumbs up/down, ratings, corrections) and implicit signals (dwell time, re-prompts, abandonment).

**Store & Annotate** — Persist feedback with context (session, prompt, response, user segment) and optionally enrich it with human annotator labels.

**Process & Analyze** — Aggregate signals, filter noise, detect distribution shifts, and derive reward or preference labels.

**Apply** — Update the model via fine-tuning, reward model retraining, or prompt/system-level adjustments.

---

## 3. Common Features of AI Feedback Systems

### 3.1 Explicit vs. Implicit Feedback Collection

**Explicit feedback** is user-initiated: thumbs up/down, star ratings, corrections, preference rankings between two responses.

**Implicit feedback** is inferred from behavior: response regeneration, copy-paste frequency, follow-up clarification requests, session abandonment.

Both signal types are important. Explicit feedback is more reliable per instance; implicit feedback scales to millions of interactions without user friction.

### 3.2 Pairwise Preference / Ranking

Instead of rating responses on an absolute scale, RLHF systems present annotators (human or AI) with multiple candidate responses to the same prompt and ask which is preferred. This relative comparison is more consistent than absolute scoring and is the foundation for training reward models.

The preference dataset takes the form:

```python
{"prompt": "...", "chosen": "Response A", "rejected": "Response B"}
```

### 3.3 Reward Modeling

A **reward model** is a neural network trained to predict the scalar quality of any model response, acting as a proxy for human judgment at scale. Once trained on human preference data, it can score millions of candidate outputs during reinforcement learning without requiring a human in the loop for every example.

### 3.4 Online vs. Offline Feedback Pipelines

- **Offline RLHF**: Feedback is collected in batches, a reward model is trained, and then the policy is updated — periodic, predictable, but slow to adapt.
- **Online Iterative RLHF**: Feedback is continuously collected from the live system and the model is updated in rolling cycles, enabling dynamic adaptation to evolving user preferences.

### 3.5 Data Drift and Concept Drift Monitoring

Production feedback systems must monitor for:

- **Data drift**: The distribution of incoming prompts shifts away from training data.
- **Concept drift**: User expectations and preferences change over time.
- **Reward hacking**: The model finds shortcuts to maximize reward without genuinely improving quality.

KL divergence monitoring between the current policy and the SFT baseline is a standard technique for detecting drift early.

### 3.6 Human-in-the-Loop (HITL) Integration

Not all feedback can or should be automated. High-stakes decisions, safety-critical outputs, or subjective assessments require human annotators. Well-designed systems route uncertain or flagged outputs to human review queues automatically.

### 3.7 Feedback Aggregation and Noise Filtering

Individual feedback signals are noisy. Aggregation strategies include:

- Majority vote across annotators
- Inter-annotator agreement scoring (e.g., Cohen's Kappa)
- Confidence-weighted averaging
- Outlier rejection before dataset construction

---

## 4. System Architecture Patterns

### 4.1 The Reflect-and-Critique Pattern

Described by Google's Antonio Gulli as one of the highest-impact agentic patterns, a primary LLM generates a response while a secondary evaluator LLM critiques it. This internal feedback loop catches errors before the user sees them and is analogous to automated peer review.

```
User Prompt ──▶ Generator LLM ──▶ Response Draft ──▶ Evaluator LLM
                      ▲                                    │
                      └─────── Refined Response ◀──────────┘
                                (if critique fails)
```

A maximum iteration limit (typically 3–5 cycles) prevents infinite loops.

### 4.2 The Self-Improving Agent Pattern

Agents with built-in feedback loops and retraining pipelines that adapt continuously. Key requirements include:

- Transparent and auditable updates
- Monitoring for data drift, concept drift, and performance decay
- Rollback mechanisms for regression
- Governed release cycles with checkpoints

### 4.3 The Reward Shaping Pipeline

```
Human Annotations ──▶ Reward Model Training ──▶ Policy Optimization (PPO/DPO)
        ▲                                                    │
        └──────────── New Preference Data ◀──────────────────┘
```

---

## 5. Python Implementation Examples

### 5.1 Feedback Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"          # 1-5 Likert scale
    PREFERENCE = "preference"  # pairwise: "a" or "b"
    CORRECTION = "correction"  # free-text improvement


@dataclass
class FeedbackRecord:
    """Core data structure representing a single user feedback event."""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    prompt: str = ""
    response: str = ""
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP
    value: Optional[float | str] = None   # scalar, label, or free text
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "response": self.response,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
```

### 5.2 Feedback Collector

```python
import json
import sqlite3
from pathlib import Path


class FeedbackCollector:
    """
    Collects and persists feedback records to a local SQLite store.
    In production this would write to a distributed event store (Kafka, Kinesis, etc.)
    """

    def __init__(self, db_path: str = "feedback.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id  TEXT,
                prompt      TEXT,
                response    TEXT,
                feedback_type TEXT,
                value       TEXT,
                metadata    TEXT,
                timestamp   TEXT
            )
        """)
        self.conn.commit()

    def collect(self, record: FeedbackRecord) -> None:
        """Persist a FeedbackRecord."""
        d = record.to_dict()
        self.conn.execute(
            """INSERT OR IGNORE INTO feedback VALUES
               (:feedback_id, :session_id, :prompt, :response,
                :feedback_type, :value, :metadata, :timestamp)""",
            {**d, "value": str(d["value"]), "metadata": json.dumps(d["metadata"])},
        )
        self.conn.commit()

    def fetch_recent(self, limit: int = 100) -> list[dict]:
        """Retrieve the most recent feedback records."""
        cursor = self.conn.execute(
            "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
```

### 5.3 Preference Dataset Builder (for RLHF)

```python
from dataclasses import dataclass


@dataclass
class PreferencePair:
    """A single pairwise preference example for reward model training."""
    prompt: str
    chosen: str    # preferred response
    rejected: str  # dispreferred response


class PreferenceDatasetBuilder:
    """
    Converts raw pairwise feedback records into a clean preference dataset
    suitable for RLHF reward model training.
    """

    def __init__(self, min_agreement: float = 0.6):
        self.min_agreement = min_agreement
        self._pairs: list[PreferencePair] = []

    def add_comparison(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        annotator_votes: list[str],  # each vote is "a" or "b"
    ) -> None:
        """
        Add a comparison. Only include if annotator agreement exceeds threshold.
        Majority vote determines chosen vs rejected.
        """
        if not annotator_votes:
            return

        vote_a = annotator_votes.count("a")
        vote_b = annotator_votes.count("b")
        total = len(annotator_votes)
        agreement = max(vote_a, vote_b) / total

        if agreement < self.min_agreement:
            return  # too noisy — skip this example

        if vote_a >= vote_b:
            chosen, rejected = response_a, response_b
        else:
            chosen, rejected = response_b, response_a

        self._pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))

    def export(self) -> list[dict]:
        """Export the dataset as a list of dicts (ready for HuggingFace datasets, etc.)."""
        return [
            {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            for p in self._pairs
        ]

    def __len__(self) -> int:
        return len(self._pairs)


# --- Example usage ---
builder = PreferenceDatasetBuilder(min_agreement=0.66)

builder.add_comparison(
    prompt="Explain gradient descent in simple terms.",
    response_a="Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent.",
    response_b="It's like rolling a ball downhill until it reaches the lowest point.",
    annotator_votes=["b", "b", "a", "b"],  # 3/4 prefer response_b — accepted
)

dataset = builder.export()
print(f"Dataset size: {len(builder)} examples")
print(dataset[0])
```

### 5.4 Reward Model Scorer (Lightweight Example)

```python
import torch
import torch.nn as nn


class SimpleRewardModel(nn.Module):
    """
    A minimal reward model that scores a (prompt, response) pair.
    In practice this would be a fine-tuned transformer (e.g. Llama-3 with a
    classification head), but this illustrates the interface.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # scalar reward score
        )

    def forward(self, prompt_emb: torch.Tensor, response_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([prompt_emb, response_emb], dim=-1)
        return self.encoder(combined).squeeze(-1)


def preference_loss(
    reward_model: SimpleRewardModel,
    prompt_emb: torch.Tensor,
    chosen_emb: torch.Tensor,
    rejected_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry preference loss: maximize P(chosen > rejected).
    Standard loss function for RLHF reward model training.
    """
    r_chosen = reward_model(prompt_emb, chosen_emb)
    r_rejected = reward_model(prompt_emb, rejected_emb)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    return loss
```

### 5.5 Feedback Aggregator with Drift Detection

```python
import statistics
from collections import defaultdict


class FeedbackAggregator:
    """
    Aggregates feedback signals and detects anomalies or drift
    in user satisfaction over time.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._scores: list[float] = []          # rolling satisfaction scores
        self._segment_scores: dict[str, list[float]] = defaultdict(list)

    def ingest(self, score: float, segment: str = "global") -> None:
        """Ingest a normalized satisfaction score in [0, 1]."""
        self._scores.append(score)
        self._segment_scores[segment].append(score)

        # Keep a rolling window
        if len(self._scores) > self.window_size:
            self._scores.pop(0)

    def mean_satisfaction(self) -> float:
        return statistics.mean(self._scores) if self._scores else 0.0

    def detect_drift(self, threshold: float = 0.1) -> bool:
        """
        Compares the satisfaction of the first vs second half of the window.
        Returns True if a meaningful drop is detected — a signal to trigger retraining.
        """
        if len(self._scores) < self.window_size:
            return False  # not enough data yet

        mid = len(self._scores) // 2
        early = statistics.mean(self._scores[:mid])
        recent = statistics.mean(self._scores[mid:])

        drop = early - recent
        if drop > threshold:
            print(f"[DRIFT ALERT] Satisfaction dropped {drop:.2%} "
                  f"({early:.2%} → {recent:.2%}). Retraining recommended.")
            return True
        return False


# --- Example usage ---
agg = FeedbackAggregator(window_size=10)

# Simulate 10 feedback scores — good early, declining later
scores = [0.9, 0.85, 0.88, 0.87, 0.9, 0.72, 0.68, 0.65, 0.60, 0.55]
for s in scores:
    agg.ingest(s)

drift_detected = agg.detect_drift(threshold=0.15)
print(f"Overall satisfaction: {agg.mean_satisfaction():.2%}")
```

### 5.6 Human-in-the-Loop Review Queue

```python
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class ReviewTask:
    """A prioritized task routed to human reviewers."""
    priority: float          # lower = higher urgency (min-heap)
    feedback_id: str = field(compare=False)
    prompt: str = field(compare=False)
    response: str = field(compare=False)
    reason: str = field(compare=False)


class HITLReviewQueue:
    """
    Routes low-confidence or flagged outputs to a human review queue.
    Priority is determined by model confidence and feedback signal strength.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold
        self._heap: list[ReviewTask] = []

    def submit(
        self,
        feedback_id: str,
        prompt: str,
        response: str,
        model_confidence: float,
        reason: str = "low confidence",
    ) -> bool:
        """
        Submit an output for human review if confidence is below threshold.
        Returns True if the task was enqueued.
        """
        if model_confidence < self.threshold:
            task = ReviewTask(
                priority=model_confidence,  # lower confidence = higher urgency
                feedback_id=feedback_id,
                prompt=prompt,
                response=response,
                reason=reason,
            )
            heapq.heappush(self._heap, task)
            return True
        return False

    def next_task(self) -> ReviewTask | None:
        """Pop the highest-priority (lowest confidence) task."""
        return heapq.heappop(self._heap) if self._heap else None

    def queue_depth(self) -> int:
        return len(self._heap)
```

---

## 6. End-to-End Pipeline Integration

Bringing all components together into a production-style pipeline:

```python
def run_feedback_pipeline(
    session_id: str,
    prompt: str,
    response_a: str,
    response_b: str,
    user_preference: str,        # "a" or "b"
    model_confidence: float,
) -> None:
    """
    Demonstrates a complete feedback ingestion pipeline:
    collect → store → aggregate → queue for human review.
    """
    collector = FeedbackCollector()
    aggregator = FeedbackAggregator(window_size=100)
    queue = HITLReviewQueue(confidence_threshold=0.6)
    dataset_builder = PreferenceDatasetBuilder(min_agreement=0.6)

    # 1. Collect explicit preference feedback
    record = FeedbackRecord(
        session_id=session_id,
        prompt=prompt,
        response=response_a if user_preference == "a" else response_b,
        feedback_type=FeedbackType.PREFERENCE,
        value=user_preference,
    )
    collector.collect(record)

    # 2. Update preference dataset
    dataset_builder.add_comparison(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        annotator_votes=[user_preference],
    )

    # 3. Aggregate satisfaction signal
    score = 1.0 if user_preference == "a" else 0.0
    aggregator.ingest(score)

    # 4. Route to HITL if model confidence is low
    queue.submit(
        feedback_id=record.feedback_id,
        prompt=prompt,
        response=record.response,
        model_confidence=model_confidence,
    )

    # 5. Check for drift and alert
    aggregator.detect_drift()

    print(f"Pipeline complete. Queue depth: {queue.queue_depth()}")
    print(f"Dataset size: {len(dataset_builder)} preference pairs")
```

---

## 7. Advanced Techniques (2025)

### 7.1 RLTHF — Targeted Human Feedback

Recent work (2025) introduced RLTHF, which uses a reward model's score distribution to identify the hardest-to-annotate samples and routes only those to human annotators. This achieves full-annotation-level alignment with just 6–7% of the human annotation effort, dramatically reducing cost without sacrificing quality.

### 7.2 Online Iterative RLHF

Unlike batch-based RLHF, online iterative RLHF collects feedback from the live production system and updates the policy in rolling cycles. This dynamic approach adapts to shifting user preferences in near real-time and has achieved state-of-the-art performance on benchmarks like AlpacaEval-2 and Arena-Hard.

### 7.3 RLAIF — Reinforcement Learning from AI Feedback

Rather than relying exclusively on human annotators, RLAIF uses a capable LLM as the preference judge to generate synthetic preference labels. This dramatically scales the feedback pipeline and reduces annotation costs, while human reviewers focus on calibration and edge-case adjudication.

### 7.4 Direct Preference Optimization (DPO)

DPO bypasses the reward model entirely by directly optimizing the policy on preference data via a reparameterized objective. Simpler to implement than PPO-based RLHF, it has been widely adopted (Meta's Llama 4 uses a multi-round SFT + PPO + DPO alignment pipeline).

---

## 8. Key Metrics for Feedback System Health

|Metric|Description|Target|
|---|---|---|
|Preference Agreement Rate|% of annotator votes matching the majority label|> 70%|
|Mean Satisfaction Score|Average normalized user feedback score|> 0.75|
|KL Divergence (policy vs SFT)|Measures policy drift from the aligned baseline|Monitor for spikes|
|HITL Queue Depth|Volume of outputs awaiting human review|< 500 backlog|
|Reward Model Accuracy|Accuracy of reward model on held-out preference pairs|> 75%|
|Feedback Latency|Time from user interaction to feedback ingestion|< 500ms|

---

## 9. Design Principles Summary

**Collect broadly, filter aggressively.** Cast a wide net for feedback signals, but invest in noise filtering and inter-annotator agreement before any signal enters training data.

**Design for drift.** User preferences and language evolve. Build monitoring for data drift, concept drift, and reward hacking from day one — not as an afterthought.

**Minimize annotation burden.** Techniques like RLTHF, RLAIF, and implicit signal collection reduce the human annotation cost that would otherwise bottleneck continuous improvement.

**Make feedback loops auditable.** Every update to a production model should trace back to specific feedback batches, reward model versions, and evaluation benchmarks. Auditability is a prerequisite for enterprise compliance and safety-critical deployments.

**Prefer online over offline where feasible.** Online iterative RLHF adapts faster to production distribution shifts. Offline pipelines are safer and more controlled for high-stakes systems.

---

## 10. References and Further Reading

- Lambert et al. (2024) — _Tülu 3: Pushing Frontiers in Open Language Model Post-Training_ — comprehensive open-source RLHF recipe.
- Rafailov et al. (2023) — _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_ — foundational DPO paper.
- Preprints.org (2025) — _Introduction to RLHF: A Review of Current Developments_ — survey of 2025 advances including RLTHF and online iterative RLHF.
- VentureBeat (2025) — _Agentic Design Patterns: The Missing Link Between AI Demos and Enterprise Value_ — practical patterns including Reflection and Critique.
- Google Cloud Architecture Center (2025) — _Choose a Design Pattern for Your Agentic AI System_.
- CMU ML Blog (2025) — _RLHF 101: A Technical Tutorial on RLHF_ — reproducible end-to-end RLHF tutorial.

---

_Report generated: April 2026 | Focus: AI Engineering — User Feedback Integration_