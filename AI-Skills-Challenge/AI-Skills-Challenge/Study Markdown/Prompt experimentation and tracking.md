
# Prompt experimentation and tracking

## 1. Why It Matters

LLMs are inherently complex and their responses can be influenced by slight changes in prompts. Prompt tracking serves as the backbone of systematic and reproducible research, enabling developers to understand the impact of different prompts on model behavior and performance.

The best practitioners treat prompting like an experimental process — they build, test, and fine-tune in small steps until the model produces the level of precision or creativity they're aiming for. Iteration is the real differentiator between casual users and skilled prompt engineers.

---

## 2. The Experimentation Loop

The core loop is: **Draft → Test → Evaluate → Iterate → Version → Deploy**. Each phase has structure.

### 2a. Start with a Hypothesis

Never change a prompt randomly. Frame it as an experiment:

- **Baseline:** `"Summarize the following article."`
- **Hypothesis:** Adding role + output format constraint will reduce hallucinations and improve structure.
- **Variant:** `"You are a senior analyst. Summarize the article below in 3 bullet points, each under 20 words. Focus only on facts stated in the text."`

Build constraints into your prompts by default. Even a short limit like "three examples max" or "use Markdown headings" improves clarity and repeatability.

### 2b. Isolate Variables (One Change at a Time)

This is the most violated rule. Common variables to isolate:

|Variable|Example|
|---|---|
|Role / Persona|"You are a..." vs no role|
|Output format|JSON vs prose vs bullets|
|Few-shot examples|0-shot vs 1-shot vs 3-shot|
|Chain-of-thought|"Think step by step" vs direct answer|
|Temperature|0.0 vs 0.7 vs 1.0|
|Context position|Instructions first vs last|

If you change role _and_ format _and_ temperature simultaneously, you cannot know what drove the improvement.

### 2c. Build a Test Dataset

Never evaluate on one input. Build a representative golden dataset:

```python
# golden_dataset.py
TEST_CASES = [
    {
        "id": "tc_001",
        "input": "Article: Tesla reported record earnings of $2.3B...",
        "expected_themes": ["earnings", "record", "Tesla"],
        "expected_format": "bullet_list",
    },
    {
        "id": "tc_002",
        "input": "Article: The FDA approved a new Alzheimer's drug...",
        "expected_themes": ["FDA", "approval", "Alzheimer's"],
        "expected_format": "bullet_list",
    },
    # ... 20+ cases covering edge cases, long inputs, ambiguous inputs
]
```

---

## 3. Tracking: Treating Prompts as Code

Just like code, prompts should be versioned and tracked to understand changes, roll back if needed, and iteratively experimented upon to find if there is any regression in the quality of responses.

### 3a. Prompt Versioning Schema

Store prompts as structured artifacts, not strings buried in code:

```python
# prompts/summarize_article_v3.yaml
name: summarize_article
version: "3.0"
author: "your_name"
date: "2026-04-09"
model: "claude-sonnet-4-20250514"
temperature: 0.3
description: "Summarizes news articles into structured bullets"
changelog: "v3: Added role + length constraint. v2: Added JSON output. v1: baseline"

system: |
  You are a senior news analyst. Your job is to summarize articles
  accurately, using only facts stated in the text. Never infer or assume.

user_template: |
  Summarize the following article in exactly 3 bullet points.
  Each bullet must be under 20 words.
  
  Article:
  {article_text}
  
  Respond in this JSON format:
  {{"summary": ["bullet1", "bullet2", "bullet3"]}}
```

```python
# prompt_loader.py
import yaml
from pathlib import Path

def load_prompt(name: str, version: str = "latest") -> dict:
    path = Path(f"prompts/{name}_v{version}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

prompt = load_prompt("summarize_article", version="3.0")
```

### 3b. MLflow for Prompt Experiment Tracking

MLflow acts like a lab notebook — but smarter. It logs every detail: prompts, model versions, system settings, and outputs. MLflow's role spans the full LLM development and deployment lifecycle and is model-agnostic, making it perfect for the LLM ecosystem.

```python
import mlflow
import anthropic
import json
from pathlib import Path

client = anthropic.Anthropic()

def run_prompt_experiment(
    prompt_version: str,
    test_cases: list,
    experiment_name: str = "article_summarizer"
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"prompt_v{prompt_version}"):
        prompt = load_prompt("summarize_article", version=prompt_version)

        # Log all prompt parameters
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("model", prompt["model"])
        mlflow.log_param("temperature", prompt["temperature"])
        mlflow.log_param("system_prompt_length", len(prompt["system"]))

        # Log the full prompt as an artifact
        mlflow.log_text(
            json.dumps(prompt, indent=2),
            f"prompt_v{prompt_version}.json"
        )

        results = []
        scores = []

        for tc in test_cases:
            user_msg = prompt["user_template"].format(
                article_text=tc["input"]
            )
            response = client.messages.create(
                model=prompt["model"],
                max_tokens=500,
                temperature=prompt["temperature"],
                system=prompt["system"],
                messages=[{"role": "user", "content": user_msg}]
            )

            output_text = response.content[0].text

            # Parse and evaluate
            try:
                parsed = json.loads(output_text)
                bullets = parsed.get("summary", [])
                format_score = 1.0 if len(bullets) == 3 else 0.0
                length_score = sum(
                    1 for b in bullets if len(b.split()) <= 20
                ) / 3
                theme_score = sum(
                    1 for theme in tc["expected_themes"]
                    if theme.lower() in output_text.lower()
                ) / len(tc["expected_themes"])
                score = (format_score + length_score + theme_score) / 3
            except json.JSONDecodeError:
                format_score = length_score = theme_score = score = 0.0

            results.append({
                "id": tc["id"],
                "output": output_text,
                "format_score": format_score,
                "length_score": length_score,
                "theme_score": theme_score,
                "overall_score": score,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })
            scores.append(score)

        # Log aggregate metrics
        mlflow.log_metric("mean_score", sum(scores) / len(scores))
        mlflow.log_metric("min_score", min(scores))
        mlflow.log_metric(
            "avg_input_tokens",
            sum(r["input_tokens"] for r in results) / len(results)
        )

        # Log full results as artifact
        mlflow.log_text(
            json.dumps(results, indent=2), "eval_results.json"
        )

        print(f"Prompt v{prompt_version} | Mean Score: {sum(scores)/len(scores):.3f}")
        return results
```

### 3c. LangSmith for Production Tracing

LangSmith helps developers build, debug, and deploy LLM applications. It offers tracing, evaluation, and monitoring tools to understand how prompts and agents behave in production. It's framework-agnostic and works with LangChain, LangGraph, or any custom code that calls LLM APIs directly.

```python
from langsmith import Client
from langsmith.wrappers import wrap_anthropic
import anthropic

# Wraps all calls for automatic tracing
ls_client = Client()
traced_client = wrap_anthropic(anthropic.Anthropic())

def summarize_with_tracing(article_text: str, prompt_version: str):
    prompt = load_prompt("summarize_article", version=prompt_version)
    
    # All calls are automatically traced in LangSmith
    response = traced_client.messages.create(
        model=prompt["model"],
        max_tokens=500,
        temperature=prompt["temperature"],
        system=prompt["system"],
        messages=[{
            "role": "user",
            "content": prompt["user_template"].format(
                article_text=article_text
            )
        }],
        # LangSmith metadata for filtering in the UI
        extra_headers={"x-langsmith-metadata": json.dumps({
            "prompt_version": prompt_version,
            "use_case": "article_summarizer"
        })}
    )
    return response.content[0].text
```

---

## 4. Evaluation Strategies

### 4a. LLM-as-Judge (Scalable Automated Eval)

```python
def llm_judge_eval(question: str, answer: str, context: str) -> dict:
    """Use Claude to score another Claude response."""
    judge_prompt = f"""You are an impartial evaluator. Score the answer on:
    - Faithfulness (0-1): Does it only use facts from the context?
    - Completeness (0-1): Does it address all parts of the question?
    - Conciseness (0-1): Is it free of unnecessary content?

    Context: {context}
    Question: {question}
    Answer: {answer}

    Respond ONLY as JSON: 
    {{"faithfulness": 0.0, "completeness": 0.0, "conciseness": 0.0, "reasoning": ""}}
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return json.loads(response.content[0].text)
```

### 4b. Key Metrics to Track

|Metric|What to Measure|
|---|---|
|**Format adherence**|Does the output match the required structure?|
|**Factual faithfulness**|No hallucinations vs source material|
|**Latency**|p50/p95 response times|
|**Token cost**|Input + output tokens per call|
|**Regression score**|Did the new prompt break passing test cases?|
|**LLM-as-judge score**|Automated quality scoring|

---

## 5. Best Practices Summary

**Versioning & Storage** Track authors, comments, diffs, and rollbacks for every change. Keep production-ready prompts stable while experimenting on branches. Store intent, dependencies, schemas, and evaluator configs together.

**Iteration Discipline** The process of prompt engineering is rarely static; it involves testing, evaluating, and revising prompts based on results. By regularly experimenting with different approaches and refining your prompts, you can identify which elements drive the most accurate and useful responses.

**Tooling** For most projects, the Anthropic or OpenAI SDK plus Pydantic plus a simple template loader covers 90% of use cases. Store your prompts in version control. Validate your outputs with schemas. Write tests. Instrument your calls.

**Staying Current** Every major model update changes how prompts are processed. Even small shifts in reasoning depth, context handling, or token limits can noticeably affect output quality. Re-run your eval suite after every model upgrade.

---

## 6. Tooling Quick Reference

|Tool|Best For|Python-Friendly?|
|---|---|---|
|**MLflow**|Experiment tracking, metric comparison, model registry|✅ First-class|
|**LangSmith**|Production tracing, LLM debugging, dataset eval|✅ SDK available|
|**Langfuse**|Open-source self-hosted observability|✅ SDK available|
|**Weights & Biases Weave**|Teams already on W&B for ML training|✅ One-line setup|
|**Braintrust**|Fast-moving teams needing PM + engineer collaboration|✅ SDK available|
|**YAML/Git**|Lightweight versioning for solo/small teams|✅ No dependencies|

The approach scales from a simple YAML file in a Git repo all the way to a full MLflow or LangSmith deployment — pick the complexity that matches your team size and production demands.