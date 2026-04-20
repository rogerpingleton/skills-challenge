# Testing

Testing in AI Engineering is both familiar and uniquely challenging compared to traditional software. You're dealing with probabilistic outputs, non-deterministic behavior, and systems that can degrade silently — which makes a disciplined testing culture _more_ important, not less.

---

## The Testing Pyramid (and why it looks different for AI)

The classic testing pyramid still applies, but each layer has AI-specific nuance:

```
         /\
        /  \   E2E / Integration Tests
       /----\
      /      \  Component / Unit Tests
     /--------\
    /          \  Data & Model Tests (AI-specific base)
   /____________\
```

**For AI systems, the pyramid gets a new foundation layer**: data quality and model behavior tests that don't exist in traditional software.

---

## The Four Testing Layers You Need to Know

### 1. Unit Testing

Testing individual functions and components in isolation.

```python
# Example: testing a prompt formatting function
import pytest
from your_module import format_prompt

def test_format_prompt_includes_context():
    context = "User is a beginner"
    question = "What is a list?"
    result = format_prompt(context, question)
    assert "User is a beginner" in result
    assert "What is a list?" in result

def test_format_prompt_raises_on_empty_question():
    with pytest.raises(ValueError):
        format_prompt("some context", "")
```

Unit tests are fast, cheap, and should be run on every commit. They test the _deterministic_ scaffolding around your AI system — parsers, formatters, routers, validators.

---

### 2. LLM / Model Output Testing (The AI-Specific Layer)

This is where AI Engineering diverges sharply from traditional software. You can't assert `output == "exact string"`. Instead you test _properties_ of outputs.

**Strategies:**

**a) Structural assertions** — Does the output have the right shape?

```python
import json

def test_llm_returns_valid_json(llm_client):
    response = llm_client.complete("Return a JSON object with keys 'name' and 'age'")
    try:
        parsed = json.loads(response)
        assert "name" in parsed
        assert "age" in parsed
    except json.JSONDecodeError:
        pytest.fail("LLM did not return valid JSON")
```

**b) Semantic assertions** — Does the output _mean_ the right thing? Use an LLM-as-judge pattern:

```python
def test_response_is_helpful(llm_client, judge_client):
    question = "What is gradient descent?"
    response = llm_client.complete(question)

    judgment = judge_client.complete(f"""
    Question: {question}
    Response: {response}
    
    Does this response correctly explain gradient descent? 
    Answer only YES or NO.
    """)
    
    assert "YES" in judgment.upper()
```

**c) Regression / golden set testing** — Maintain a set of known good input/output pairs:

```python
GOLDEN_SET = [
    {"input": "Summarize: The sky is blue.", "must_contain": ["sky", "blue"]},
    {"input": "Translate to French: hello", "must_contain": ["bonjour"]},
]

@pytest.mark.parametrize("case", GOLDEN_SET)
def test_golden_set(llm_client, case):
    response = llm_client.complete(case["input"]).lower()
    for keyword in case["must_contain"]:
        assert keyword in response
```

---

### 3. Integration Testing

Test how your components work _together_ — the LLM call + retrieval + parsing + downstream logic as a pipeline.

```python
def test_rag_pipeline_end_to_end(rag_pipeline, vector_db):
    # Seed the vector DB
    vector_db.upsert("doc_1", "Python was created by Guido van Rossum.")
    
    result = rag_pipeline.query("Who created Python?")
    
    assert result.answer is not None
    assert "guido" in result.answer.lower()
    assert len(result.sources) > 0  # Verify retrieval happened
```

These tests are slower and should run on PRs and pre-deployment, not every save.

---

### 4. Evaluation Pipelines (Evals)

Evals are the AI Engineering equivalent of a test suite for model _quality_ — not just correctness. This is a core skill expected of AI Engineers.

A typical eval framework:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    input: str
    expected_behavior: str  # Human-readable description
    scorer: Callable[[str], float]  # Returns 0.0 - 1.0

def run_evals(llm_client, eval_cases: list[EvalCase]) -> dict:
    results = []
    for case in eval_cases:
        output = llm_client.complete(case.input)
        score = case.scorer(output)
        results.append({"input": case.input, "score": score, "output": output})
    
    avg_score = sum(r["score"] for r in results) / len(results)
    return {"average_score": avg_score, "results": results}
```

You'll be expected to know frameworks like **Braintrust**, **LangSmith**, **Ragas** (for RAG evaluation), or **Promptfoo** for structured eval pipelines.

---

## When to Test: A Practical Timeline

|Trigger|What to run|
|---|---|
|Every file save|Unit tests (fast, local)|
|Every commit|Unit + linting + type checks|
|Every PR|Unit + integration tests + eval regression|
|Pre-deployment|Full eval suite + E2E tests|
|Post-deployment|Monitoring & online evals|
|Prompt/model change|Full eval suite + golden set comparison|

The key principle: **test as early and as often as the cost allows.** Unit tests are nearly free. Full evals can be expensive (API calls, time), so run them strategically.

---

## Integrating Continuous Testing into Your Workflow

### Local Development

Use `pytest` with a `--fast` marker to separate slow integration/eval tests from quick unit tests:

```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=False)

def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="skipped in fast mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
```

```bash
pytest --fast        # Quick feedback loop during development
pytest               # Full suite before committing
```

### CI/CD Pipeline (GitHub Actions example)

```yaml
name: AI Test Suite
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit --fast

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run integration tests
        run: pytest tests/integration
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  evals:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'  # Only on merge to main
    steps:
      - name: Run eval suite
        run: python evals/run_evals.py
```

### Monitoring in Production

Testing doesn't stop at deployment. AI Engineers are expected to instrument their systems:

```python
import logging
from datetime import datetime

def log_llm_call(input_text: str, output_text: str, latency_ms: float):
    logging.info({
        "timestamp": datetime.utcnow().isoformat(),
        "input_length": len(input_text),
        "output_length": len(output_text),
        "latency_ms": latency_ms,
        "flagged": run_safety_check(output_text)
    })
```

Tools like **LangSmith**, **Arize**, and **Weights & Biases** provide dashboards for this.

---

## What You're Expected to Know as an AI Engineer

|Area|Expectation|
|---|---|
|`pytest` fundamentals|Fixtures, parametrize, markers, conftest|
|Mocking LLM calls|`unittest.mock`, `pytest-mock` to avoid API calls in unit tests|
|Eval design|Building and maintaining golden datasets, scoring functions|
|LLM-as-judge|Using a model to evaluate another model's output|
|Regression testing|Catching prompt/model changes that degrade quality|
|CI/CD integration|Wiring tests into GitHub Actions or similar|
|Observability|Logging, tracing, and monitoring model outputs in production|
|Test data management|Curating representative, diverse test cases|

---

## The Golden Rules

1. **Mock your LLM in unit tests.** Don't make real API calls — they're slow, costly, and non-deterministic. Mock the client and test your logic.
2. **Maintain a golden dataset.** A curated set of representative inputs with expected behaviors is your most valuable testing asset.
3. **Test prompts like code.** Every prompt change is a potential regression. Treat it like a code change — version it, test it, review it.
4. **Separate fast from slow tests.** You need a feedback loop under 30 seconds for daily development. Keep expensive evals in a separate pipeline.
5. **Monitor in production.** Your test suite catches known failures. Production catches unknown ones — instrument accordingly.

The bottom line: as an AI Engineer, testing is not just about correctness — it's about **confidence in probabilistic systems**. The rigor you bring to testing is directly proportional to how much you can trust your system at scale.