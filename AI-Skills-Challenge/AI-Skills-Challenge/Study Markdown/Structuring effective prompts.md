# Structuring effective prompts

## 1. What Is Prompt Engineering?

Prompt engineering is the practice of crafting precise, structured instructions to direct an AI model toward producing desired outputs. Think of it as programming in natural language — the quality of the output is directly proportional to the quality of the input.

In 2025–2026, prompt engineering has matured beyond simple question-asking. It now encompasses **system design**, **context construction**, **output specification**, and **iterative evaluation**. Every instruction written into a system prompt is effectively a product decision: it shapes model behavior, tone, scope, and reliability at scale.

---

## 2. Anatomy of an Effective Prompt

A well-structured prompt is composed of several distinct layers. Not every prompt needs all of them, but knowing each layer allows you to deliberately choose what to include.

### The GOLD Framework

A field-tested ordering for important prompts:

```
Goal:       What is the objective and success criteria?
Output:     What format, length, and tone is expected?
Limits:     What is out of scope? What constraints apply?
Data:       What context, examples, or source material should be used?
```

### The Five Core Components

**1. Role / Persona** Tell the model _who it is_. This shapes expertise, tone, and the frame of reference for every response.

```
You are a senior Python developer with expertise in data engineering
and Apache Spark. You write clean, PEP-8 compliant code with
detailed inline comments.
```

**2. Context** Provide the background the model needs to respond appropriately. Ambiguous prompts produce ambiguous results. Context eliminates guesswork.

```
The codebase uses Python 3.11, pandas 2.x, and SQLAlchemy 2.0.
The database is PostgreSQL 15. The target audience for documentation
is junior engineers.
```

**3. Task / Instruction** State precisely what you want done. Use positive framing (say what TO do, not just what NOT to do).

```
Refactor the following function to use SQLAlchemy 2.0 ORM syntax,
replace all raw SQL strings, and add type annotations.
```

**4. Format Specification** Define the structure of the output. Format specification dramatically reduces the need for post-processing and revision.

```
Return your answer as a JSON object with the following keys:
- "refactored_code": the full updated function
- "changes": a list of strings summarizing each change made
- "breaking_changes": a boolean indicating if any changes are breaking
```

**5. Constraints / Guardrails** Define scope limits, style rules, and what to avoid.

```
Do not use deprecated SQLAlchemy 1.x syntax. Keep the function
signature identical. Do not add new dependencies.
```

---

## 3. Core Techniques

### 3.1 Zero-Shot Prompting

The model is given a task with no examples. It relies entirely on its pre-trained knowledge. Best suited for well-understood, common tasks.

**When to use:** Simple classification, summarization, common code tasks, general Q&A.

**Example:**

```python
prompt = """
Classify the sentiment of the following customer review as
POSITIVE, NEGATIVE, or NEUTRAL. Respond with only the label.

Review: "The product arrived on time but the packaging was damaged."
"""
```

**Output:** `NEGATIVE`

---

### 3.2 Few-Shot Prompting

Provide 3–5 worked examples that demonstrate the desired input → output pattern. This dramatically improves accuracy and consistency for complex or domain-specific tasks.

**When to use:** Structured data extraction, custom classification schemes, specific output formats, domain-specific tasks.

**Example:**

```python
prompt = """
Extract the key fields from each support ticket. Return JSON.

<examples>
<example>
Input: "Hi, my order #4821 hasn't arrived and it's been 2 weeks.
        I'm in Seattle."
Output: {"order_id": "4821", "issue": "missing delivery",
         "location": "Seattle", "urgency": "high"}
</example>
<example>
Input: "The color in my photo prints looks washed out on my
        Epson XP-5200."
Output: {"order_id": null, "issue": "print quality",
         "device": "Epson XP-5200", "urgency": "medium"}
</example>
</examples>

Now extract from this ticket:
"Order #9034 was delivered but the wrong item was sent.
 Customer is in Austin and requesting an exchange."
"""
```

**Key rules for few-shot examples:**

- Make examples **diverse** — cover different edge cases, not just easy ones
- Keep example structure **identical** to avoid confusing the model
- Use clear delimiters (XML tags like `<example>` work well for Claude models)
- Include 3–5 examples; more rarely helps and wastes tokens

---

### 3.3 Chain-of-Thought (CoT) Prompting

Instruct the model to reason step-by-step before providing a final answer. This technique dramatically improves accuracy on tasks requiring multi-step logic, math, or complex decision-making.

**Zero-Shot CoT:** Simply append a reasoning cue.

```python
prompt = """
A data pipeline processes 1,200 records per minute. It runs for
3.5 hours but is throttled to 60% capacity for the first 45 minutes.

How many total records were processed?

Let's think through this step by step.
"""
```

**Few-Shot CoT:** Provide examples that include the reasoning chain.

```python
prompt = """
<example>
Problem: A cache hit rate is 85%. If 4,000 requests come in,
         how many result in a cache miss?
Reasoning:
  - Cache miss rate = 100% - 85% = 15%
  - Cache misses = 4000 * 0.15 = 600
Answer: 600 cache misses
</example>

Problem: A microservice handles 500 requests/second. It has a
         2% error rate. How many errors occur in 10 minutes?
Reasoning:
"""
```

**Rule of thumb:** Use CoT for any task where the answer depends on intermediate logic. Simple lookups or classification do not benefit from it — and adding it wastes tokens.

---

## 4. What Makes a Good Prompt?

### ✅ Specificity Over Vagueness

Every degree of ambiguity is a degree of variance in the output. Specify the task, the format, the audience, the scope.

|Vague|Specific|
|---|---|
|"Explain climate change"|"Write a 3-paragraph summary of the primary causes of climate change for a high school audience, using neutral tone and no technical jargon"|
|"Write a function"|"Write a Python 3.11 function that takes a list of dicts and returns a new list sorted by the 'timestamp' key in ascending order. Include type annotations and a docstring."|
|"What are AI trends?"|"What are the most significant AI engineering trends in 2025–2026, specifically around agentic systems and LLM evaluation?"|

---

### ✅ Defined Output Format

If you need structured data, say so explicitly. Don't rely on the model to infer your downstream needs.

**Bad:**

```
Summarize the pros and cons of GraphQL vs REST.
```

**Good:**

```
Compare GraphQL and REST APIs. Structure your response as a
markdown table with columns: Criteria | GraphQL | REST.
Include rows for: Performance, Flexibility, Caching,
Learning Curve, and Best Use Case.
```

---

### ✅ Role Assignment

Assigning a role steers the model's vocabulary, depth of reasoning, and framing.

```python
system_prompt = """
You are a staff-level Python engineer performing a security-focused
code review. You identify vulnerabilities, suggest hardening
measures, and explain the OWASP category each issue maps to.
You are concise — you do not repeat the code back unless
highlighting a specific line.
"""
```

---

### ✅ Positive Framing

Telling the model what TO do is far more reliable than listing prohibitions.

**Bad:** "Don't be vague. Don't use bullet points. Don't write too long."

**Good:** "Write in prose paragraphs. Be specific and concrete. Limit your answer to 200 words."

---

### ✅ Delimiter Usage

For complex prompts mixing instructions, context, data, and examples — use XML-style delimiters to eliminate ambiguity. This is especially important for Claude models, which parse XML tags reliably.

```python
prompt = f"""
<instructions>
You are a data quality analyst. Review the CSV data below
and identify any anomalies. Return a JSON list of issues.
</instructions>

<data>
{csv_data}
</data>

<output_format>
[
  {{"row": int, "column": str, "issue": str, "severity": "low|medium|high"}}
]
</output_format>
"""
```

---

### ✅ Timeframes and Constraints

Adding specific constraints eliminates unnecessary guesswork and scopes the output.

```
"What are the AI trends for 2025–2026?" is better than "What are current AI trends?"
"Summarize in exactly 3 sentences" is better than "Keep it brief"
"Use only information from the provided document" prevents hallucination
```

---

## 5. What Makes a Bad Prompt?

### ❌ Over-Vagueness

The single most common failure. The more the model must infer, the higher the output variance.

```
# Bad
"Help me with my code."

# Better
"I have a Python function that reads a CSV and inserts rows into
PostgreSQL. It works but is slow for files over 100k rows.
Review the function below and suggest performance optimizations,
specifically around batching and connection pooling."
```

---

### ❌ Conflicting Instructions

Contradictory directives confuse the model and produce unpredictable results.

```
# Bad — the model can't satisfy both
"Be thorough and comprehensive. Keep your answer to one sentence."

"Respond in formal English. Use casual language to keep it friendly."
```

---

### ❌ Implicit Assumptions

Don't assume the model knows your domain conventions, your codebase style, or your audience unless you've stated them.

```
# Bad — what "best practices" means is undefined
"Refactor this to follow best practices."

# Good — best practices are explicit
"Refactor this function to follow PEP-8 style, use type annotations,
replace mutable default arguments, and add a docstring following
Google Python Style Guide format."
```

---

### ❌ Prompt Stuffing

Cramming too many tasks into a single prompt degrades quality on all of them. If a task has multiple distinct sub-tasks, use prompt chaining (separate API calls).

```python
# Bad — doing too much in one shot
"Summarize this document, extract all action items, translate them
to Spanish, evaluate sentiment, and generate a follow-up email."

# Better — chain these as separate calls:
# 1. summarize(document)
# 2. extract_action_items(document)
# 3. translate(action_items, target="es")
# 4. evaluate_sentiment(document)
# 5. generate_email(summary, action_items)
```

---

### ❌ Missing Output Specification

Leaving format undefined means the model will improvise — producing inconsistent structure across runs.

```
# Bad — will produce inconsistent JSON structure
"List the top 5 Python libraries for data engineering."

# Good
"""List the top 5 Python libraries for data engineering.
Return a JSON array. Each item should have:
- "name": library name
- "primary_use": one sentence
- "github_stars_approx": integer
- "best_for": list of 2-3 use case strings
"""
```

---

### ❌ Negative-Only Constraints

A list of prohibitions without positive direction leaves the model without clear guidance.

```
# Bad
"Don't be too technical. Don't write too much. Don't use jargon."

# Good
"Write for a non-technical executive audience. Use plain language.
Keep the total response to 150 words or less."
```

---

## 6. Advanced Techniques

### 6.1 System Prompt vs User Prompt

For production LLM applications, separate concerns cleanly:

- **System prompt:** Role, personality, output rules, persistent constraints, response format
- **User prompt:** The actual dynamic input — the query, the data, the document

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are a Python code reviewer specializing in security and performance.

Rules:
- Identify issues by category: Security, Performance, Readability, Correctness
- For each issue, specify: line number (if applicable), severity (critical/high/medium/low), description, and suggested fix
- Return your findings as a JSON array
- Do not reproduce the full code in your response
- If no issues are found, return an empty array []
"""

def review_code(code: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"<code>\n{code}\n</code>"}
        ]
    )
    return response
```

---

### 6.2 Prompt Chaining

Break complex workflows into sequential, focused calls. Each call's output feeds the next. This makes intermediate outputs inspectable and allows branching logic.

```python
import anthropic
import json

client = anthropic.Anthropic()

def extract_requirements(user_story: str) -> list[str]:
    """Step 1: Extract functional requirements from a user story."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="You are a senior business analyst. Extract requirements as JSON.",
        messages=[{
            "role": "user",
            "content": f"""
            Extract a list of testable functional requirements from this user story.
            Return ONLY a JSON array of requirement strings. No other text.

            User story: {user_story}
            """
        }]
    )
    return json.loads(response.content[0].text)


def generate_test_cases(requirements: list[str]) -> list[dict]:
    """Step 2: Generate test cases from requirements."""
    requirements_str = "\n".join(f"- {r}" for r in requirements)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="You are a QA engineer. Generate pytest test cases.",
        messages=[{
            "role": "user",
            "content": f"""
            Generate unit test cases for the following requirements.
            Return a JSON array of objects with keys:
            "requirement", "test_name", "test_description", "expected_outcome"

            Requirements:
            {requirements_str}
            """
        }]
    )
    return json.loads(response.content[0].text)


# Usage
user_story = """
As a user, I want to log in with my email and password,
so that I can access my account securely.
"""
requirements = extract_requirements(user_story)
test_cases = generate_test_cases(requirements)
```

---

### 6.3 Self-Consistency

For high-stakes reasoning tasks, run the same prompt multiple times (with temperature > 0) and take the majority answer. This reduces variance significantly.

```python
from collections import Counter

def self_consistent_answer(prompt: str, n: int = 5) -> str:
    """Run a prompt n times and return the most common answer."""
    answers = []
    for _ in range(n):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
            # Note: temperature control depends on your API setup
        )
        answers.append(response.content[0].text.strip())

    return Counter(answers).most_common(1)[0][0]
```

---

### 6.4 Meta-Prompting

Use the LLM to evaluate and improve your own prompts. This is particularly effective when you know _what_ you want but are unsure _how_ to ask for it.

```python
meta_prompt = """
I want to improve this prompt:
"{original_prompt}"

Analyze it using the TCRTE framework:
- Task clarity: What is missing or ambiguous about the task?
- Context: What background information would help?
- References: What examples or format specs would improve outputs?
- Testing criteria: How would I evaluate if the output is correct?
- Enhancement: What specific changes would make this prompt more reliable?

Then provide an improved version of the prompt.
"""
```

---

## 7. Testing & Evaluating Prompts for Consistency

Writing a good prompt is only half the work. The other half is _verifying_ it works reliably at scale. Production-grade prompt engineering treats prompts like software: they require version control, test suites, and regression testing.

### 7.1 Why Consistency Testing Matters

LLMs are non-deterministic. The same prompt can produce different outputs on repeated runs. Without systematic evaluation, you are operating on vibes — which breaks the moment you go to production.

The three most common failure modes are:

1. **Inconsistent formatting** — output structure changes between runs
2. **Factual drift** — answers vary in accuracy depending on phrasing
3. **Regression** — a prompt that worked on model version X breaks on version Y

---

### 7.2 Building a Golden Dataset

A golden dataset is a curated set of `(input, expected_output)` pairs that represents your real use cases, including edge cases.

```python
# golden_dataset.py

GOLDEN_CASES = [
    {
        "id": "sentiment_001",
        "input": "The product arrived on time but the packaging was damaged.",
        "expected": "NEGATIVE",
        "tags": ["sentiment", "mixed_signal"]
    },
    {
        "id": "sentiment_002",
        "input": "Absolutely love this product! Fast shipping, great quality.",
        "expected": "POSITIVE",
        "tags": ["sentiment", "clear_positive"]
    },
    {
        "id": "sentiment_003",
        "input": "It works as described.",
        "expected": "NEUTRAL",
        "tags": ["sentiment", "minimal_signal"]
    },
    # Add adversarial and edge cases
    {
        "id": "sentiment_004",
        "input": "Not bad, I guess. Could be worse.",
        "expected": "NEUTRAL",
        "tags": ["sentiment", "edge_case", "ambiguous"]
    },
]
```

---

### 7.3 Automated Evaluation in Python

```python
# eval_runner.py
import anthropic
import json
from dataclasses import dataclass
from typing import Callable

client = anthropic.Anthropic()

PROMPT_V1 = """
Classify the sentiment of the following review as POSITIVE, NEGATIVE, or NEUTRAL.
Respond with only the label.

Review: {input}
"""

PROMPT_V2 = """
You are a sentiment analysis engine. Classify the sentiment of customer reviews.

Rules:
- Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL
- Mixed reviews that lean negative → NEGATIVE
- Mixed reviews with no clear lean → NEUTRAL
- Do not include punctuation or explanation

Review: {input}
"""


@dataclass
class EvalResult:
    case_id: str
    input: str
    expected: str
    actual: str
    passed: bool
    tags: list[str]


def run_prompt(prompt_template: str, input_text: str) -> str:
    """Run a single prompt and return the raw text response."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": prompt_template.format(input=input_text)
        }]
    )
    return response.content[0].text.strip().upper()


def evaluate_prompt(
    prompt_template: str,
    dataset: list[dict],
    exact_match: bool = True
) -> list[EvalResult]:
    """Evaluate a prompt against a golden dataset."""
    results = []
    for case in dataset:
        actual = run_prompt(prompt_template, case["input"])
        passed = (actual == case["expected"]) if exact_match else True
        results.append(EvalResult(
            case_id=case["id"],
            input=case["input"],
            expected=case["expected"],
            actual=actual,
            passed=passed,
            tags=case.get("tags", [])
        ))
    return results


def print_eval_report(results: list[EvalResult], label: str = "Prompt"):
    """Print a summary report of evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    print(f"\n{'='*50}")
    print(f"EVAL REPORT: {label}")
    print(f"{'='*50}")
    print(f"Score: {passed}/{total} ({100*passed/total:.1f}%)")

    if failed:
        print(f"\nFailed Cases:")
        for r in failed:
            print(f"  [{r.case_id}] tags={r.tags}")
            print(f"    Input:    {r.input[:60]}...")
            print(f"    Expected: {r.expected}")
            print(f"    Got:      {r.actual}")


# Run evaluation
from golden_dataset import GOLDEN_CASES

results_v1 = evaluate_prompt(PROMPT_V1, GOLDEN_CASES)
results_v2 = evaluate_prompt(PROMPT_V2, GOLDEN_CASES)

print_eval_report(results_v1, label="Prompt V1 (baseline)")
print_eval_report(results_v2, label="Prompt V2 (with rules)")
```

---

### 7.4 LLM-as-a-Judge

For open-ended outputs (summaries, explanations, code quality), exact-match evaluation is impossible. Instead, use a second LLM call to score the output.

```python
JUDGE_PROMPT = """
You are an impartial evaluator assessing the quality of an AI-generated response.

Evaluate the response on the following criteria. Return a JSON object only.

Criteria:
- accuracy (0-5): Is the information factually correct and complete?
- relevance (0-5): Does it directly address the question asked?
- clarity (0-5): Is it easy to understand for the intended audience?
- format_compliance (0 or 1): Does it match the specified output format?

<question>{question}</question>
<expected_behavior>{expected_behavior}</expected_behavior>
<actual_response>{actual_response}</actual_response>

Return ONLY JSON. Example:
{{"accuracy": 4, "relevance": 5, "clarity": 3, "format_compliance": 1, "notes": "..."}}
"""


def judge_response(
    question: str,
    expected_behavior: str,
    actual_response: str
) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                expected_behavior=expected_behavior,
                actual_response=actual_response
            )
        }]
    )
    return json.loads(response.content[0].text)
```

**Important tips for LLM-as-a-judge:**

- Use binary criteria where possible (pass/fail) — they are more reliable than 1–10 scales
- Set temperature to 0 (or as low as possible) for the judge model
- Run the judge multiple times and average for high-stakes evaluation
- Never use the same model to judge its own output (verbosity and style bias)

---

### 7.5 Versioning Your Prompts

Treat prompts like code. Store them in version control. Never edit a production prompt in place.

```python
# prompt_registry.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptVersion:
    version: str
    created_at: str
    description: str
    template: str
    eval_score: float | None = None


PROMPT_REGISTRY = {
    "sentiment_classifier": [
        PromptVersion(
            version="1.0.0",
            created_at="2026-01-10",
            description="Initial baseline",
            template="Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.\n\n{input}",
            eval_score=0.72
        ),
        PromptVersion(
            version="1.1.0",
            created_at="2026-02-15",
            description="Added role + explicit rules for mixed signals",
            template="""You are a sentiment analysis engine...
(full template here)""",
            eval_score=0.91
        ),
    ]
}

def get_prompt(name: str, version: str = "latest") -> PromptVersion:
    versions = PROMPT_REGISTRY[name]
    if version == "latest":
        return versions[-1]
    return next(v for v in versions if v.version == version)
```

---

### 7.6 Production Monitoring

After deployment, your prompts encounter inputs that your golden dataset never anticipated. Monitor production traffic for:

- **Format failures** — output didn't match expected structure (causes downstream errors)
- **Low-confidence outputs** — when using classification, flag near-boundary cases
- **User feedback signals** — thumbs down, correction requests, session abandonment
- **Latency drift** — longer outputs can indicate prompt regression or scope creep

Tools in this space (as of 2026): **Braintrust**, **PromptLayer**, **LangSmith**, **Helicone**, **Opik (open source)**, **DeepEval (open source)**.

---

## 8. Python Examples: Putting It All Together

### A Complete, Production-Style Prompt for Code Review

```python
import anthropic
import json

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are a senior Python engineer conducting a pull request review.

Your review covers four areas:
1. Correctness — logic errors, off-by-one errors, unhandled exceptions
2. Security — injection risks, hardcoded secrets, unsafe deserialization
3. Performance — N+1 queries, unnecessary loops, memory inefficiency
4. Style — PEP-8 violations, unclear naming, missing type annotations

Output format (JSON only, no prose before or after):
{
  "overall_verdict": "approve" | "request_changes" | "nitpick",
  "issues": [
    {
      "category": "Correctness" | "Security" | "Performance" | "Style",
      "severity": "critical" | "high" | "medium" | "low",
      "line": <int or null>,
      "description": "<what the problem is>",
      "suggestion": "<specific fix>"
    }
  ],
  "summary": "<2-sentence overall assessment>"
}

If there are no issues, return an empty "issues" array and verdict "approve".
"""

def review_python_code(code: str, context: str = "") -> dict:
    context_block = f"\n<context>{context}</context>" if context else ""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{context_block}\n<code>\n{code}\n</code>"
        }]
    )
    
    return json.loads(response.content[0].text)


# Example usage
code_to_review = """
import pickle
import os

def load_user_data(user_id):
    path = f"/data/users/{user_id}.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_user_orders(db, user_id):
    orders = []
    user = db.query(f"SELECT * FROM users WHERE id = {user_id}")
    for item in user['order_ids']:
        order = db.query(f"SELECT * FROM orders WHERE id = {item}")
        orders.append(order)
    return orders
"""

result = review_python_code(
    code_to_review,
    context="Internal admin tool. Users are authenticated admins, not end users."
)
print(json.dumps(result, indent=2))
```

---

## 9. Quick Reference Checklist

Use this before shipping any prompt to production.

### Prompt Construction

- [ ] Role/persona is clearly defined
- [ ] Context provides all necessary background
- [ ] Task instruction is unambiguous and positively framed
- [ ] Output format is explicitly specified
- [ ] Constraints and scope limits are stated
- [ ] Complex prompts use XML delimiters to separate sections
- [ ] No conflicting instructions

### Few-Shot Examples (if used)

- [ ] 3–5 examples provided
- [ ] Examples are diverse and cover edge cases
- [ ] Example structure is identical across all examples
- [ ] Examples are wrapped in `<example>` / `<examples>` tags

### Reasoning Tasks

- [ ] Chain-of-thought is used for multi-step logic
- [ ] Reasoning steps are separated from the final answer
- [ ] CoT is NOT used for simple lookups (wastes tokens)

### Testing & Evaluation

- [ ] Golden dataset created with representative + edge cases
- [ ] Automated eval script measures exact-match or LLM-as-judge score
- [ ] Prompt is versioned with a description and eval score
- [ ] Regression tested before shipping updates to production
- [ ] Production monitoring in place for format failures and drift

---

_References: Anthropic Claude Prompting Best Practices (2026), OpenAI Prompt Engineering Guide, promptingguide.ai, Braintrust Prompt Evaluation (2025), Caylent LLM Evaluations Guide (2025), AWS Chain-of-Thought Prompting Reference (2026)_