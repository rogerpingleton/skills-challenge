
# In-context learning techniques

## What Is In-Context Learning?

In-context learning refers to a model's ability to temporarily learn from prompts by providing the model with a few examples to learn from. Unlike training and fine-tuning, in-context learning is temporary — the learned patterns disappear once the conversation context resets.

ICL is an emergent ability of large language models, emerging as a property of model scale where its efficacy increases at a different rate in larger models than in smaller models. This means your choice of model matters as much as your prompt design.

The key insight for engineers: **you are not changing model weights**. The model does not adjust its weights; instead, it learns on the fly by drawing on the context from few-shot examples.

---

## The Core Spectrum: Zero-Shot → One-Shot → Few-Shot

When giving AI models instructions, we can improve performance by providing examples. This technique — in-context learning — lets the AI learn from examples embedded directly in the prompt, rather than needing additional training or fine-tuning. By including examples, we guide the AI to better understand the task and expected output, leveraging its pattern recognition abilities.

### Zero-Shot

No examples — the model relies entirely on its pretrained knowledge and your instruction.

```python
prompt = """
Classify the sentiment of this text as positive, negative, or neutral.
Text: "The API latency was terrible and the docs were confusing."
Sentiment:
"""
```

**When to use it:** Simple, well-defined tasks (summarization, translation, factual Q&A) where the task boundary is unambiguous. Fast and cheap — no example curation overhead.

**Watch out for:** Ambiguous output formats. Without examples, the model may respond with "The sentiment is negative." instead of just `negative`, breaking downstream parsing.

---

### One-Shot

A single example to anchor format and behavior.

```python
prompt = """
Classify the sentiment of this text as positive, negative, or neutral.

Text: "This library is incredibly well-documented."
Sentiment: positive

Text: "The API latency was terrible and the docs were confusing."
Sentiment:
"""
```

**When to use it:** When you need strict output format compliance and have limited context window budget.

---

### Few-Shot

Two or more examples are included, allowing the model to recognize patterns and deliver more accurate responses. The more examples provided, the better the model typically performs, as it can generalize them to new, similar tasks.

```python
prompt = """
Classify the sentiment of this text as: positive, negative, or neutral.

Text: "This library is incredibly well-documented."
Sentiment: positive

Text: "Installation failed on my machine twice."
Sentiment: negative

Text: "The feature works as expected."
Sentiment: neutral

Text: "The API latency was terrible and the docs were confusing."
Sentiment:
"""
```

**Critical engineering note:** Careful selection of examples ensures that support examples are representative and diverse. This significantly affects the model's ability to generalize and correctly solve new queries.

---

## N-Way K-Shot: The Formal Framing

N-Way refers to the number of classes or categories in a task. K-Shot refers to the number of examples per class in the support set. In an N-Way K-Shot scenario, examples are divided into a Support Set — which contains examples the model uses for in-context learning — and a Query Set, which contains new examples the model must classify or process.

For a 3-way (positive/negative/neutral), 2-shot sentiment classifier, you'd provide 6 examples total (2 per class). This is important when designing your prompt templates programmatically:

```python
def build_few_shot_prompt(support_set: list[dict], query: str) -> str:
    """
    support_set: list of {"text": ..., "label": ...}
    """
    examples = "\n\n".join(
        f'Text: "{ex["text"]}"\nSentiment: {ex["label"]}'
        for ex in support_set
    )
    return f"Classify sentiment as positive, negative, or neutral.\n\n{examples}\n\nText: \"{query}\"\nSentiment:"
```

---

## Example Selection Strategies

This is where a lot of real-world ICL performance is won or lost. Not all examples are equal.

### 1. Random Selection

The baseline. Simple but high-variance — the wrong random draw can tank accuracy.

### 2. Similarity-Based (KNN) Retrieval

Retrieve examples most semantically similar to the query using embeddings.

Using a retrieval process to dynamically select the most relevant examples to include in the prompt can perform better than static few-shot learning. However, it requires a larger amount of labeled data, and the retrieval process can add latency.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_top_k_examples(query_embedding, example_embeddings, examples, k=3):
    sims = cosine_similarity([query_embedding], example_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return [examples[i] for i in top_k_idx]
```

Similarity-based few-shot demonstration selection strategies significantly improve performance of few-shot ICL compared to random selection.

### 3. Self-Generated ICL (SG-ICL)

When you don't have a labeled dataset at all, ask the model to generate its own examples first.

Self-Generated In-Context Learning (SG-ICL) generates demonstrations using the language model itself, reducing reliance on external datasets. It performs better than zero-shot prompting but not as well as techniques that operate on curated datasets. It should be used when there's no dataset available or when computational resources are limited.

```python
# Step 1: generate exemplars
generation_prompt = """
Generate 3 examples of customer reviews with their sentiment labels (positive/negative/neutral).
Format each as:
Text: "<review>"
Sentiment: <label>
"""

# Step 2: use those generated examples as your few-shot context for inference
```

---

## Chain-of-Thought (CoT) as an ICL Extension

CoT is a direct extension of ICL where examples include reasoning steps, not just input-output pairs.

Chain-of-thought prompting is a technique that allows large language models to solve a problem as a series of intermediate steps before giving a final answer. It was developed to help LLMs handle multi-step reasoning tasks, such as arithmetic or commonsense reasoning.

**Standard few-shot:**

```
Q: A server processes 120 requests/min. How long for 1,800 requests?
A: 15 minutes
```

**Few-shot CoT:**

```
Q: A server processes 120 requests/min. How long for 1,800 requests?
A: The rate is 120 requests per minute. Total requests = 1800.
   Time = 1800 / 120 = 15 minutes.
   Answer: 15 minutes.

Q: A pipeline runs at 450 records/sec. How long for 81,000 records?
A:
```

For coding tasks, CoT is particularly powerful. Prompting the model to "think step by step" or showing examples that include pseudocode before final code can substantially improve correctness on complex logic.

---

## Prompt Structure and Formatting

Organizing prompts into distinct sections (like `<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`, etc.) and using techniques like XML tagging or Markdown headers to delineate these sections reduces ambiguity and helps the model distinguish between instructions and input data.

A production-grade ICL prompt for a Python AI engineer might look like:

```python
SYSTEM = """
You are a code review assistant. Classify Python code snippets as:
- clean: well-structured, readable, follows PEP8
- needs_refactor: functional but has style or complexity issues
- broken: contains bugs or logic errors

Respond ONLY with one of: clean, needs_refactor, broken
"""

EXAMPLES = """
<example>
Code:
def add(a, b):
    return a + b
Label: clean
</example>

<example>
Code:
def f(x,y,z):
    r=x+y
    return r*z+r
Label: needs_refactor
</example>

<example>
Code:
def divide(a, b):
    return a / b  # no zero-division guard
Label: broken
</example>
"""

def classify_code(snippet: str) -> str:
    user_message = EXAMPLES + f"\nCode:\n{snippet}\nLabel:"
    # call your LLM API here
```

---

## Key Engineering Considerations

**Context window budget.** Given that LLMs are constrained by a finite attention budget, good context engineering means finding the smallest possible set of high-signal tokens that maximize the likelihood of a desired outcome. More examples cost tokens; find the sweet spot empirically via evals.

**Example order matters.** Research consistently shows that examples closer to the end of the prompt (recency bias) have stronger influence on generation. For classification tasks, shuffle labels across positions to avoid anchoring effects.

**Label distribution in examples.** Ensure your support set reflects the real distribution of your task. If 80% of real inputs are "neutral," don't show only extreme positive/negative examples — the model will be miscalibrated.

**ICL vs. fine-tuning tradeoff.** ICL allows enterprises to leverage pretrained LLMs without the heavy lift of retraining, enabling faster time-to-value and reduced costs. However, if you find yourself needing 20+ examples consistently, fine-tuning is likely more cost-effective long-term.

**Prompt sensitivity.** Prompt quality significantly impacts application performance, with variations in formatting and structure creating accuracy differences of up to 76 points. Always A/B test your prompt variants with a held-out eval set before deploying to production.

---

## Summary Decision Table

|Scenario|Recommended ICL Strategy|
|---|---|
|No labeled data, simple task|Zero-shot|
|Need strict output format|One-shot|
|Classification with labeled examples|Few-shot (3–8 examples)|
|No labeled dataset at all|SG-ICL (self-generated)|
|Large example pool, retrieval available|KNN/similarity-based retrieval ICL|
|Multi-step reasoning / complex logic|Few-shot Chain-of-Thought|
|High volume, consistent task|Consider fine-tuning over ICL|

The practical takeaway: treat ICL as your first tool before reaching for fine-tuning. It's fast to iterate, requires no training infrastructure, and with good example selection, can match fine-tuned model performance for many classification and generation tasks.