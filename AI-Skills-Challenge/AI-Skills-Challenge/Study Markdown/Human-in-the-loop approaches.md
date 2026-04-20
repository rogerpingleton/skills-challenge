# Human-in-the-loop approaches

## Overview

Human-in-the-Loop (HITL) is the intentional integration of human oversight, judgment, and feedback into AI/ML workflows at critical decision points. Rather than letting an AI system operate fully autonomously, HITL inserts humans at specific junctures — for approval, correction, labeling, or preference signaling — so that the system can remain accurate, safe, and aligned with real-world expectations.

As of 2026, with generative AI deeply embedded in production systems, HITL is no longer an optional quality-assurance step. It is a foundational engineering concern spanning model training, deployment, and continuous improvement.

> "The goal of HITL is to allow AI systems to achieve the efficiency of automation without sacrificing the precision, nuance and ethical reasoning of human oversight." — IBM Think, 2025

---

## Core HITL Approaches

### 1. Active Learning

**What it is:** The model identifies data points where it is uncertain or ambiguous and surfaces those specific samples to human annotators for labeling.

**Key idea:** Instead of labeling an entire dataset upfront, the model selects the _most informative_ examples, dramatically reducing annotation cost.

**Feedback mechanism:** Humans provide labels, classifications, or structured annotations on the flagged samples. Those labels are fed back to retrain or fine-tune the model.

**Python pattern:**

```python
# Pseudocode: active learning loop
uncertain_samples = model.get_uncertain_samples(pool, threshold=0.3)
human_labels = annotation_queue.send(uncertain_samples)
training_set.extend(human_labels)
model.retrain(training_set)
```

---

### 2. Reinforcement Learning from Human Feedback (RLHF)

**What it is:** Humans compare or rank model outputs (pairwise or grouped), and these preferences are used to train a _reward model_. The reward model then guides further fine-tuning of the main model via RL algorithms such as PPO (Proximal Policy Optimization).

**Phases:**

1. **Supervised Fine-Tuning (SFT):** The base model is primed on demonstration data — examples of ideal input/output pairs.
2. **Reward Model Training:** Humans rank different model outputs; these preferences train a reward model.
3. **RL Fine-Tuning:** The policy (main LLM) is optimized using the reward model as a scoring function, with a KL penalty to prevent the model from drifting too far from the SFT baseline.

**Key variants (2025):**

- **DPO (Direct Preference Optimization):** Skips the separate reward model; takes gradient steps directly on pairwise preference data. Simpler and widely adopted.
- **RLAIF (RL from AI Feedback):** Uses another model (e.g., GPT-4 or Claude) as the feedback provider, reducing human annotation burden.
- **Online Iterative RLHF:** Continuous feedback collection and model updates; enables dynamic adaptation rather than periodic re-training cycles.
- **RLTHF (Targeted Human Feedback):** Combines LLM-based initial alignment with selective human corrections — only humans review the cases where the model is most uncertain.

---

### 3. Interactive Machine Learning (IML)

**What it is:** Humans directly and iteratively participate in shaping the model's decision boundaries — in real time, rather than in a batch training pipeline.

**Feedback mechanism:** Users interact with model outputs as they are generated, correcting classifications, adjusting features, or providing structured feedback tags that are used to refine the model immediately.

**Example:** A student critiques an AI-generated explanation using predefined tags (`clarity`, `correctness`, `tone`). Those tags drive a RAG retrieval step to generate a better follow-up response.

---

### 4. Human Approval / Checkpoint Gates (Agentic Workflows)

**What it is:** In multi-step agentic pipelines, a human must explicitly approve or reject the AI's proposed action before execution continues. The system pauses and waits.

**Feedback mechanism:** Boolean or structured approval. The reviewer can also _edit_ the AI's output (e.g., refine an email draft) before approving.

**When triggered:**

- Action would modify persistent state (database writes, API calls with side effects)
- Model confidence falls below a defined threshold
- Domain rules require human sign-off (regulated industries)

---

### 5. Implicit Feedback / Behavioral Signal Collection

**What it is:** User interactions with the system — clicks, thumbs-up/down, content engagement, ignoring a suggestion — are treated as training signals without requiring explicit input.

**Feedback mechanism:** Engagement events (ratings, dwell time, ignores, rewrites) are logged and used to retrain or re-rank model outputs.

**Example:** A streaming platform's recommendation system updates user preference embeddings based on watch/skip/like actions.

---

### 6. Expert Review & Annotation Workflows

**What it is:** Domain specialists (radiologists, lawyers, security analysts) review and annotate AI outputs offline — typically async — to build ground-truth datasets or to validate decisions before they go live.

**Feedback mechanism:** Structured annotation tools (label studios, custom UIs), correction forms, audit logs. Outputs flow back into training pipelines or model evaluation benchmarks.

---

## Common Features & Implementation Patterns

|Feature|Description|Typical Implementation|
|---|---|---|
|**Thumbs up / down**|Lightweight binary signal on individual responses|Button event → feedback DB → periodic fine-tune|
|**Star ratings / sliders**|Granular preference signal (1–5 scale)|Rating widget → reward model input|
|**Annotation tags**|Structured labels (e.g., `clarity`, `bias`, `factual_error`)|Tag palette → RAG or fine-tune signal|
|**Edit-then-approve**|Human corrects AI draft before accepting|Diff tracking → SFT example generation|
|**Pairwise comparisons**|Human chooses preferred response from two options|Comparison UI → DPO / reward model training|
|**Confidence-based routing**|Below-threshold confidence → escalate to human|Threshold config → queue routing|
|**Audit logs**|All AI decisions and human interventions logged|Structured event logging → compliance / retraining|
|**Rejection + reason**|Human rejects output with a reason code|Reason taxonomy → targeted correction|
|**Gold standard tests**|Periodically inject known answers to measure annotator quality|Quality scoring → annotator calibration|

### Python: Minimal Feedback Collection Pipeline

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class FeedbackEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: str = ""
    prompt: str = ""
    response: str = ""
    rating: Optional[int] = None          # 1-5 or None
    thumbs: Optional[bool] = None         # True=up, False=down
    correction: Optional[str] = None      # human-edited version
    tags: list[str] = field(default_factory=list)

class FeedbackStore:
    def __init__(self):
        self._events: list[FeedbackEvent] = []

    def record(self, event: FeedbackEvent):
        self._events.append(event)

    def get_for_sft(self) -> list[dict]:
        """Return correction pairs suitable for SFT fine-tuning."""
        return [
            {"prompt": e.prompt, "completion": e.correction}
            for e in self._events
            if e.correction is not None
        ]

    def get_preference_pairs(self) -> list[dict]:
        """Return thumbs-up/down pairs for DPO training."""
        ups = {e.prompt: e.response for e in self._events if e.thumbs is True}
        downs = {e.prompt: e.response for e in self._events if e.thumbs is False}
        pairs = []
        for prompt in ups:
            if prompt in downs:
                pairs.append({
                    "prompt": prompt,
                    "chosen": ups[prompt],
                    "rejected": downs[prompt],
                })
        return pairs
```

---

## When to Use HITL

### Use HITL when:

|Condition|Rationale|
|---|---|
|**High-stakes, irreversible actions**|Database writes, financial transactions, medical orders — mistakes are costly|
|**Low model confidence**|Model uncertainty exceeds acceptable threshold for autonomous operation|
|**Regulatory or compliance requirements**|Legal, medical, and financial domains often mandate human sign-off|
|**Sparse or novel training data**|Edge cases not well represented in training require human ground-truth|
|**Subjective or context-dependent output**|Tone, ethics, nuance, and cultural context can't be fully automated|
|**Early model versions**|Before a model has accumulated sufficient production feedback|
|**Alignment-sensitive tasks**|Outputs that could be biased, harmful, or easily misinterpreted|
|**Agentic workflows with real-world side effects**|Any agent that can send emails, modify files, make API calls|

### Reduce or remove HITL when:

|Condition|Rationale|
|---|---|
|Model confidence is consistently high|Automation is safe and scalable|
|Volume is too large for human review|At massive scale, HITL becomes a bottleneck|
|Task is well-defined and testable|Automated evaluation (unit tests, evals) can replace human review|
|Latency requirements preclude human wait|Real-time systems (fraud detection, autocomplete) can't pause|

---

## Concrete Examples by Domain

### Healthcare — Medical Imaging

An AI flags ambiguous regions on a radiology scan it cannot confidently classify. A radiologist reviews only those flagged areas, provides a label (`benign` / `malignant` / `inconclusive`), and that label re-enters the training pipeline. This pattern keeps the radiologist focused on genuinely uncertain cases rather than reviewing every scan.

```python
# Simplified routing logic
def route_scan(scan, model, threshold=0.85):
    prediction, confidence = model.predict(scan)
    if confidence < threshold:
        return human_review_queue.enqueue(scan)
    return prediction
```

---

### Customer Service — AI-Assisted Agents

An LLM drafts a response to a support ticket. Before it is sent, a human agent reviews and edits it. The edited version — alongside the original draft — is logged as a correction pair for SFT. Over time, the model learns the company's tone, terminology, and escalation policies from these edit pairs.

---

### Content Operations

A content platform uses AI to draft articles covering 70–80% of the production process. Humans inject structured inputs (brand voice, ICP, messaging framework) before drafting begins, then review and approve the output before publication. Each review stage generates signal: approved content becomes positive SFT examples; rejected/heavily edited content drives targeted retraining.

---

### Agentic PTO / HR Automation (AWS Bedrock Example)

A Bedrock agent helps employees manage PTO. Reading PTO balances and listing past requests is fully automated. However, booking, modifying, or canceling a request triggers a user confirmation gate — a Boolean checkpoint that pauses the agent until the employee approves. This prevents irreversible database changes from occurring without explicit sign-off.

---

### Education — Adaptive Learning

Students using an AI tutoring system critique AI-generated explanations using predefined tags (`too fast`, `wrong answer`, `unclear notation`). These tags drive a RAG retrieval step, retrieving more targeted instructional material and adjusting the explanation's depth and style. Feedback is also accumulated to fine-tune the model toward better pedagogical responses.

---

### Recommendation Systems

A streaming service collects implicit signals — watch-time, skip events, ratings, playlist additions. These behavioural signals are used as reward signals to continuously update user preference models, without requiring users to fill in explicit surveys. The model updates more frequently than a batch retraining cycle would allow.

---

## HITL in LLM Post-Training

RLHF and its modern variants are the canonical HITL pattern for LLM alignment. As of 2025, by some estimates 70% of enterprises have adopted RLHF or DPO to align AI outputs, up from 25% in 2023.

### Standard RLHF Pipeline

```
[Pre-trained Base Model]
        |
        v
[Supervised Fine-Tuning (SFT)]
  - Human-written demonstrations
  - Teaches format & basic behaviour
        |
        v
[Reward Model Training]
  - Humans rank or compare outputs (pairwise / grouped)
  - Reward model learns to predict human preference scores
        |
        v
[RL Fine-Tuning (PPO)]
  - Policy (LLM) generates responses
  - Reward model scores them
  - KL penalty prevents reward hacking / drift
        |
        v
[Aligned LLM]
```

### DPO (Direct Preference Optimization)

DPO simplifies the pipeline by eliminating the separate reward model. Human preference data (chosen vs. rejected response pairs) is used directly to compute a classification-style loss on the policy model. Less infrastructure, fewer hyperparameters.

```python
# DPO loss (simplified concept)
def dpo_loss(policy_model, ref_model, prompt, chosen, rejected, beta=0.1):
    log_ratio_chosen = (
        policy_model.log_prob(prompt, chosen) - ref_model.log_prob(prompt, chosen)
    )
    log_ratio_rejected = (
        policy_model.log_prob(prompt, rejected) - ref_model.log_prob(prompt, rejected)
    )
    loss = -torch.log(torch.sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    return loss.mean()
```

### Online Iterative RLHF

Unlike traditional offline RLHF (batch retraining), online iterative RLHF continuously collects feedback and updates the model, enabling dynamic adaptation to evolving human preferences. This is particularly important for production LLMs that must track shifting user expectations, seasonal language changes, and new domain requirements.

---

## Agentic AI & Checkpoint Patterns

Agentic systems that can take actions in the world (browse the web, write files, call APIs, send messages) require especially careful HITL design.

### Key checkpoint patterns:

**1. Pre-execution confirmation** Before the agent executes a tool/action, the user confirms or rejects. Sufficient for most write operations.

**2. Confidence-based routing** The agent calculates a confidence or risk score for the planned action. Below threshold → human queue. Above threshold → auto-execute.

```python
def maybe_execute(action, agent_confidence, threshold=0.9):
    if action.is_destructive or agent_confidence < threshold:
        return HumanApprovalRequest(action=action)
    return action.execute()
```

**3. Edit-then-approve** The agent produces a draft (email, document, code). The human can edit the draft before approving. The edit delta is captured as a correction for future training.

**4. Structured rejection with reason** When a human rejects an agent's proposal, they select from a predefined reason taxonomy. These reasons feed targeted retraining, not just generic negative signal.

---

## Challenges & Trade-offs

|Challenge|Detail|Mitigation|
|---|---|---|
|**Scalability**|Human review does not scale linearly with data volume|Use active learning to select only high-value samples; use RLAIF for bulk cases|
|**Annotator bias**|Reviewers introduce cultural, demographic, or personal biases|Diverse annotator pools; inter-annotator agreement tracking; adversarial review|
|**Latency**|Human-in-the-loop adds wait time|Reserve HITL for async or non-real-time paths; use confidence routing to reduce volume|
|**Cost**|Domain expert annotation (medical, legal) is expensive|Tiered review: general annotators for bulk, experts for edge cases|
|**Feedback fatigue**|High review volume degrades quality over time|Rotate reviewers; simplify interfaces; use gold-standard injection to calibrate|
|**Reward hacking**|Model learns to game the reward signal rather than improve genuinely|KL divergence monitoring; diverse eval benchmarks; periodic human spot-checks|

---

## Tooling & Ecosystem (2025–2026)

|Tool / Framework|HITL Role|
|---|---|
|**Hugging Face TRL**|RLHF/DPO/PPO training pipelines for LLMs|
|**Label Studio**|General annotation platform (classification, NER, image, audio)|
|**Argilla**|LLM-specific feedback and preference data collection|
|**LangGraph / LangChain**|Agent frameworks with built-in human approval steps|
|**Amazon Bedrock Agents**|Managed agent HITL: user confirmation gates on tool use|
|**user-feedback MCP Server**|MCP-based feedback protocol for AI coding agents (e.g., Cursor)|
|**Zapier AI Agents**|No-code HITL checkpoint insertion in automation workflows|
|**MLflow / W&B**|Experiment tracking; useful for monitoring feedback loop effects on model metrics|

### Python: Minimal RLHF Feedback Collection with HF TRL

```python
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assume you've collected preference pairs from your feedback store
preference_data = feedback_store.get_preference_pairs()
dataset = Dataset.from_list(preference_data)

model = AutoModelForCausalLM.from_pretrained("your-sft-checkpoint")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-checkpoint")
tokenizer = AutoTokenizer.from_pretrained("your-sft-checkpoint")

training_args = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## Summary Decision Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                     HITL APPROACH SELECTOR                      │
├────────────────────┬──────────────────────┬─────────────────────┤
│ Situation          │ Recommended Approach  │ Feedback Signal     │
├────────────────────┼──────────────────────┼─────────────────────┤
│ LLM alignment      │ RLHF / DPO           │ Pairwise rankings   │
│ Fine-tuning (safe) │ SFT from corrections │ Edit-then-approve   │
│ Agentic side-effects│ Checkpoint gates    │ Boolean confirm     │
│ Scarce labels      │ Active learning      │ Targeted annotation │
│ High-volume, low   │                      │                     │
│   annotation cost  │ Implicit / IML       │ Behavioral signals  │
│ Domain expertise   │ Expert review queue  │ Structured labels   │
│ Real-time systems  │ Confidence routing   │ Post-hoc correction │
└────────────────────┴──────────────────────┴─────────────────────┘
```

---

_Report compiled April 2026. Sources include IBM Think, AWS ML Blog, CMU ML Blog, Preprints.org, Zapier, Splunk, Tredence, and IntuitionLabs._