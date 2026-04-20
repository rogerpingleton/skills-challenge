# Using AI judges and human evals

Evaluation is one of the most critical — and most underinvested — parts of AI engineering. Here's what you need to know.

---

## Why Evals Matter

LLMs are non-deterministic, hard to unit-test, and fail in subtle ways. Traditional software testing (assert x == y) rarely applies. Evals are your substitute for a test suite: they tell you whether your system is actually working, and whether changes make things better or worse.

The two dominant approaches are **human evaluation** and **LLM-as-judge (AI judges)**.

---

## Human Evals

### What They Are

A human rates, ranks, or labels model outputs — either directly or by comparing two outputs side-by-side (A/B).

### When to Use Them

- Establishing **ground truth** for a new task (before you have an AI judge)
- Evaluating **subjective quality** (tone, creativity, empathy)
- **Auditing** your AI judge for drift or bias
- High-stakes domains (medical, legal) where automated errors are costly

### Common Formats

|Format|Description|Best For|
|---|---|---|
|Likert scale|Rate 1–5 on a dimension|Quality, helpfulness|
|Binary pass/fail|Is this output acceptable?|Safety, correctness|
|Pairwise ranking|Which of A or B is better?|Comparing two models/prompts|
|Rubric-based|Score against explicit criteria|Structured tasks|

### Python Example — Collecting Human Labels

```python
import json
from datetime import datetime

def create_eval_task(prompt: str, response: str, eval_id: str) -> dict:
    return {
        "eval_id": eval_id,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
        "human_score": None,       # filled in by annotator
        "human_notes": None,
    }

def record_label(task: dict, score: int, notes: str = "") -> dict:
    assert 1 <= score <= 5, "Score must be 1–5"
    task["human_score"] = score
    task["human_notes"] = notes
    return task

# Build a small eval batch
tasks = [
    create_eval_task("Summarize quantum entanglement", "...<model output>...", "eval_001"),
    create_eval_task("Write a haiku about Python", "...<model output>...", "eval_002"),
]

# Simulate an annotator
tasks[0] = record_label(tasks[0], score=4, notes="Accurate but a bit verbose")
tasks[1] = record_label(tasks[1], score=5, notes="Nailed it")

print(json.dumps(tasks, indent=2))
```

### Pitfalls

- **Inter-annotator disagreement** — always compute agreement scores (Cohen's Kappa)
- **Annotator fatigue** — keep tasks short, rotate reviewers
- **Slow and expensive** at scale — use for calibration, not volume

---

## AI Judges (LLM-as-Judge)

### What They Are

You use a second LLM (the "judge") to evaluate the output of your primary LLM. The judge is given a rubric and asked to score or critique the output. This scales infinitely and can run in CI/CD pipelines.

### When to Use Them

- **Regression testing** after prompt or model changes
- **Automated pipelines** where human review at every step is impractical
- Scoring along dimensions that are hard to check programmatically (coherence, relevance, tone)

### The Core Pattern

```python
import anthropic
import json

client = anthropic.Anthropic()

def ai_judge(
    prompt: str,
    response: str,
    criteria: str = "accuracy, clarity, and completeness"
) -> dict:
    judge_prompt = f"""You are an expert evaluator. Score the following AI response.

## Task Given to the AI
{prompt}

## AI Response
{response}

## Evaluation Criteria
{criteria}

Return ONLY a JSON object with this structure:
{{
  "score": <int 1-5>,
  "reasoning": "<one sentence>",
  "pass": <true if score >= 3>
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = message.content[0].text.strip()
    return json.loads(raw)


# Example usage
result = ai_judge(
    prompt="Explain what a context window is in one sentence.",
    response="A context window is the maximum amount of text (tokens) an LLM can process at once.",
)
print(result)
# → {"score": 5, "reasoning": "Accurate and concise.", "pass": true}
```

### Multi-Criteria Judge

Real evals usually score along several dimensions independently:

```python
RUBRIC = {
    "accuracy":    "Is the factual content correct?",
    "conciseness": "Is the response appropriately brief without losing meaning?",
    "tone":        "Is the tone appropriate for the intended audience?",
}

def multi_criteria_judge(prompt: str, response: str, rubric: dict) -> dict:
    criteria_block = "\n".join(
        f"- **{k}**: {v}" for k, v in rubric.items()
    )
    keys = list(rubric.keys())
    score_schema = {k: "<int 1-5>" for k in keys}
    score_schema["overall_pass"] = "<bool>"

    judge_prompt = f"""You are an expert evaluator. Score the AI response below.

## Task
{prompt}

## Response
{response}

## Criteria
{criteria_block}

Return ONLY valid JSON matching this schema:
{json.dumps(score_schema, indent=2)}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    return json.loads(message.content[0].text.strip())

result = multi_criteria_judge(
    prompt="What is RAG?",
    response="RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with an LLM so the model can reference external documents when generating answers.",
    rubric=RUBRIC,
)
print(result)
# → {"accuracy": 5, "conciseness": 4, "tone": 5, "overall_pass": true}
```

### Pitfalls

- **Position bias** — judges favor the first option in pairwise evals. Randomize order and average.
- **Verbosity bias** — judges tend to prefer longer responses. Penalize unnecessary length explicitly in your rubric.
- **Self-serving bias** — a model judging its own outputs is unreliable. Use a different model or a stronger one.
- **Prompt sensitivity** — small rubric wording changes shift scores. Lock your judge prompt in version control.

---

## Combining Both: The Calibration Loop

The best eval systems use **human evals to calibrate AI judges**, then let AI judges run at scale:

```
1. Collect 100–200 human-labeled examples (ground truth)
2. Run your AI judge on the same examples
3. Measure agreement (Pearson r or Cohen's Kappa)
4. If agreement > 0.8 → trust the AI judge for scale
5. Periodically re-run human evals to check for drift
```

```python
from scipy.stats import pearsonr

human_scores  = [4, 5, 3, 2, 5, 4, 3, 4, 5, 2]
ai_scores     = [4, 5, 3, 3, 5, 4, 2, 4, 4, 2]

r, p = pearsonr(human_scores, ai_scores)
print(f"Pearson r = {r:.2f}, p = {p:.4f}")
# Pearson r = 0.94, p = 0.0001  → strong agreement, safe to use AI judge at scale
```

---

## What to Track in Production

|Metric|Why It Matters|
|---|---|
|Pass rate over time|Catch regressions after deploys|
|Score distribution|Detect mode collapse (everything scores 4)|
|Failure categories|Label _why_ outputs fail, not just _that_ they do|
|Judge agreement rate|Monitor your AI judge's reliability|
|Latency of eval pipeline|Slow evals don't get run|

---

## Key Takeaways

- **Start with human evals** to build ground truth. Don't skip this.
- **AI judges scale human judgment**, they don't replace it — calibrate them against humans first.
- **Version-control your eval prompts** like production code. A rubric change is a breaking change.
- **Treat your eval set as sacred** — don't train on it, don't leak it, add to it carefully.
- **Run evals in CI/CD** so every prompt or model change is automatically tested before shipping.