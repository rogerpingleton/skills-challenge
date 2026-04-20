# Evaluation metrics - perplexity, BLEU, ROUGE, semantic similarity, functional correctness, etc

## 1. Why Evaluation Matters

LLMs are deployed in high-stakes environments — medicine, law, education, customer service, and code generation. Without rigorous evaluation, model development becomes guesswork and unreliable outputs reach production. Evaluation metrics serve three fundamental purposes:

- **Development guidance** — objective signals to compare architectures, prompts, or fine-tuning strategies.
- **Regression prevention** — catching quality degradations before deployment.
- **Production monitoring** — tracking model drift and user-experience quality over time.

No single metric captures all dimensions of quality. A robust evaluation strategy combines reference-based metrics (BLEU, ROUGE), embedding-based metrics (BERTScore, semantic similarity), task-specific metrics (pass@k for code), and human evaluation.

---

## 2. Taxonomy of Evaluation Metrics

|Category|Approach|Examples|
|---|---|---|
|Language-model intrinsic|Probability of held-out text|Perplexity|
|N-gram overlap|Precision/recall of shared word sequences|BLEU, ROUGE, METEOR, CIDEr|
|Embedding-based|Cosine similarity in vector space|BERTScore, MoverScore, SAS|
|Learned/neural|Neural model trained on human judgments|BLEURT, COMET|
|Execution-based|Run code against unit tests|pass@k (HumanEval)|
|LLM-as-judge|A strong LLM rates generated output|G-Eval, GPTScore|
|Diversity|Uniqueness of generated outputs|Distinct-n, Self-BLEU|
|Human evaluation|Human raters score on rubrics|Likert scales, A/B preference|

---

## 3. Perplexity

### What It Measures

Perplexity (PPL) is an intrinsic metric that quantifies how well a language model predicts a held-out sequence. Intuitively it answers: _"On average, how many equally likely next tokens does the model consider at each step?"_ A perfectly certain model (probability = 1 for every true token) yields PPL = 1. A model that is genuinely confused averages over many candidate tokens, producing high PPL.

### Formula

```
PPL(W) = exp( -(1/N) * Σ log p(w_i | w_1 ... w_{i-1}) )
```

Where `N` is sequence length and `p(w_i | ...)` is the model's predicted probability for token `w_i`.

### When to Use

- Comparing architectures or training runs on the same tokeniser and test set.
- Monitoring linguistic fluency of generated text.
- Domain-adaptation checks (low PPL on in-domain text indicates good fit).
- **Not** suitable as a standalone metric for task quality — a model can achieve low PPL while producing repetitive or off-topic text.

### Python Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(text: str, model_name: str = "gpt2") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # mean negative log-likelihood

    return torch.exp(loss).item()

# Example
text = "The quick brown fox jumps over the lazy dog."
ppl = compute_perplexity(text)
print(f"Perplexity: {ppl:.2f}")
# Lower is better. Typical GPT-2 PPL on WikiText-103 ≈ 18–30
```

### Using the `evaluate` library (Hugging Face)

```python
import evaluate

perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(
    predictions=["The quick brown fox jumps over the lazy dog."],
    model_id="gpt2"
)
print(results["mean_perplexity"])
```

### Limitations

- Scores are **relative** to the model used for evaluation — different models yield different numbers.
- Sensitive to tokenisation; comparisons across tokenisers are not meaningful.
- Low PPL does not guarantee factual accuracy, coherence, or helpfulness.

---

## 4. BLEU (Bilingual Evaluation Understudy)

### What It Measures

BLEU compares a candidate (generated) text to one or more reference texts by counting overlapping n-grams (contiguous sequences of n words). It was originally designed for machine translation but is now a general text-generation baseline. BLEU scores range from 0 to 1 (or 0–100).

### Formula

```
BLEU = BP × exp( Σ wₙ · log pₙ )
```

Where:

- `pₙ` = clipped n-gram precision at order n (n = 1 to 4 is standard)
- `wₙ` = weight for each order (typically uniform: 1/n)
- `BP` = brevity penalty: `1` if candidate ≥ reference length, else `exp(1 - ref_len/cand_len)`

### When to Use

- Machine translation benchmarks.
- General text generation as a quick lexical-overlap baseline.
- Comparing system versions over the same reference set.

### Python Implementation

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Sentence-level BLEU
reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "sitting", "on", "the", "mat"]
score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
print(f"Sentence BLEU: {score:.4f}")

# Corpus-level BLEU (recommended for reliable scores)
references = [
    [["the", "cat", "sat", "on", "the", "mat"]],
    [["there", "is", "a", "cat", "on", "the", "mat"]],
]
hypotheses = [
    ["the", "cat", "is", "on", "the", "mat"],
    ["a", "cat", "is", "on", "the", "mat"],
]
corpus_score = corpus_bleu(references, hypotheses)
print(f"Corpus BLEU: {corpus_score:.4f}")
```

### Using `sacrebleu` (preferred for reproducibility)

```python
import sacrebleu

hypothesis = ["The cat sat on the mat."]
references = [["The cat is on the mat."]]

bleu = sacrebleu.corpus_bleu(hypothesis, references)
print(bleu.score)  # 0–100 scale
```

### Interpreting Scores

|BLEU Score|Interpretation|
|---|---|
|< 0.10|Almost no overlap|
|0.10 – 0.29|Understandable but poor quality|
|0.30 – 0.49|Moderate quality|
|0.50 – 0.69|Good to excellent quality|
|≥ 0.70|Often near human-level translation|

### Limitations

- **Insensitive to semantics** — synonyms and valid paraphrases are penalised.
- **Precision-only** — does not reward recall of important content.
- Unreliable on short texts without smoothing.
- Poor correlation with human judgment for open-ended or abstractive generation.

---

## 5. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

### What It Measures

ROUGE focuses on recall: how much of the reference content appears in the candidate. It is the standard metric for text summarisation. It comes in several variants:

|Variant|Description|
|---|---|
|ROUGE-N|Overlap of n-grams (ROUGE-1 = unigrams, ROUGE-2 = bigrams)|
|ROUGE-L|Longest Common Subsequence (LCS); respects word order without requiring contiguity|
|ROUGE-W|Weighted LCS; gives more credit to consecutive matches|
|ROUGE-S|Skip-bigram overlap; word pairs in order but with allowed gaps|
|ROUGE-SU|ROUGE-S plus unigram counts to avoid zero scores|

For each variant, ROUGE computes **Precision**, **Recall**, and **F1**. In summarisation, F1 is most commonly reported.

### Formula (ROUGE-N F1)

```
Precision = (matching n-grams) / (n-grams in candidate)
Recall    = (matching n-grams) / (n-grams in reference)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

### When to Use

- Text summarisation (primary use case).
- Document-level generation tasks.
- Any task where covering all key points from a reference matters.

### Python Implementation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

reference = "The cat sat on the mat near the window."
candidate = "A cat was sitting on the mat."

scores = scorer.score(reference, candidate)
for key, value in scores.items():
    print(f"{key}: P={value.precision:.3f} R={value.recall:.3f} F1={value.fmeasure:.3f}")
```

### Batch evaluation with `evaluate`

```python
import evaluate

rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=["A cat was sitting on the mat."],
    references=["The cat sat on the mat near the window."],
    use_stemmer=True,
)
print(results)
# {'rouge1': 0.727, 'rouge2': 0.444, 'rougeL': 0.727, 'rougeLsum': 0.727}
```

### Practical Benchmarks

- ROUGE-1 ≥ 0.45 is typical for strong abstractive summarisation on CNN/DailyMail.
- ROUGE-L often runs 3–5 points below ROUGE-1.

### Limitations

- Struggles with paraphrasing and abstractive summaries.
- Assigns equal weight to every token regardless of importance.
- Rewards extractive copies even when abstraction would be superior.
- Does not measure coherence, factual accuracy, or fluency.

---

## 6. METEOR

### What It Measures

METEOR was designed explicitly to improve on BLEU's weaknesses. It aligns candidate and reference text using exact matches, stemmed matches, and synonym matches (via WordNet), then computes an F1 score with a fragmentation penalty to capture word-order preservation.

### Formula

```
METEOR = Fmean × (1 - Penalty)
Fmean  = 10PR / (R + 9P)     # recall-weighted harmonic mean
Penalty = 0.5 × (chunks / unigram_matches)^3
```

### When to Use

- Machine translation evaluation where synonymy matters.
- Sentence-level evaluation (METEOR correlates better than BLEU at the sentence level).
- Any generation task where partial credit for near-matches is desired.

### Python Implementation

```python
# Install: pip install nltk
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.translate.meteor_score import meteor_score

reference = "the cat sat on the mat".split()
candidate = "the feline was sitting on the rug".split()

score = meteor_score([reference], candidate)
print(f"METEOR: {score:.4f}")
```

### Using `evaluate`

```python
import evaluate

meteor = evaluate.load("meteor")
results = meteor.compute(
    predictions=["the feline was sitting on the rug"],
    references=["the cat sat on the mat"]
)
print(results["meteor"])  # accounts for synonymy: higher than BLEU would be
```

### Limitations

- Relies on WordNet, which is English-centric and incomplete.
- More computationally expensive than BLEU/ROUGE.
- Fragmentation penalty can behave unexpectedly on very short texts.

---

## 7. BERTScore

### What It Measures

BERTScore uses contextual embeddings from a pre-trained transformer (e.g., `roberta-large`) to measure semantic similarity at the token level. Rather than checking for literal n-gram overlap, it computes cosine similarity between token embeddings and aggregates via:

- **Precision** — each token in the candidate is matched to its most similar token in the reference.
- **Recall** — each token in the reference is matched to its most similar token in the candidate.
- **F1** — harmonic mean of the above.

### When to Use

- Any text generation task where paraphrasing should not be penalised.
- Summarisation (outperforms ROUGE for abstractive models).
- Translation quality estimation.
- Code comment / documentation quality.

### Python Implementation

```python
# Install: pip install bert-score
from bert_score import score

candidates = ["The cat is sitting on the mat."]
references = ["A feline rested on the carpet."]

P, R, F1 = score(candidates, references, lang="en", verbose=True)
print(f"Precision: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1:        {F1.mean():.4f}")
```

### Choosing the Model

```python
# Use a specific backbone for reproducibility
P, R, F1 = score(
    candidates, references,
    model_type="roberta-large",   # recommended for English
    num_layers=17,                # layer to extract embeddings from
    lang="en"
)
```

### Using `evaluate`

```python
import evaluate

bertscore = evaluate.load("bertscore")
results = bertscore.compute(
    predictions=["The cat is sitting on the mat."],
    references=["A feline rested on the carpet."],
    lang="en"
)
print(results["f1"])
```

### Limitations

- Computationally heavier than n-gram metrics (requires a full forward pass per text pair).
- Scores are **not** directly comparable across different backbone models.
- Can be fooled by texts that are lexically similar but semantically divergent.

---

## 8. Semantic Similarity

### What It Measures

Semantic similarity quantifies how close two pieces of text are in meaning, independent of surface form. The most common approach uses **sentence embeddings** (dense vectors representing full sentences) and measures their **cosine similarity**:

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)    ∈ [-1, 1]
```

For NLP tasks, values cluster between 0 (unrelated) and 1 (nearly identical meaning).

### When to Use

- Question-answering: does the generated answer mean the same as the ground truth?
- Retrieval evaluation: are retrieved documents semantically relevant?
- RAG pipeline evaluation (answer relevancy, faithfulness).
- Duplicate detection and clustering.

### Python Implementation

```python
# Install: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good general purpose

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy hound.",  # paraphrase
    "Machine learning is a subfield of AI.",          # unrelated
]

embeddings = model.encode(texts)

# Pairwise similarity
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"Sim({i},{j}) = {sim:.4f}")
# Sim(0,1) ≈ 0.84  (paraphrase → high)
# Sim(0,2) ≈ 0.15  (unrelated → low)
```

### Cross-Encoder for Higher Accuracy

For tasks needing more precise similarity scores (at the cost of speed):

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/stsb-roberta-large")
pairs = [
    ("The cat sat on the mat.", "A feline rested on the carpet."),
    ("The cat sat on the mat.", "The stock market crashed today."),
]
scores = model.predict(pairs)
print(scores)  # [0.87, 0.01]
```

### Recommended Models by Use Case

|Model|Strengths|
|---|---|
|`all-MiniLM-L6-v2`|Fast, low-memory, great for bulk eval|
|`all-mpnet-base-v2`|Higher accuracy, general purpose|
|`multi-qa-mpnet-base-dot-v1`|Optimised for QA retrieval|
|`cross-encoder/stsb-roberta-large`|Highest accuracy pairwise, slow|

---

## 9. Functional Correctness (pass@k / HumanEval)

### What It Measures

For code generation tasks, traditional text metrics like BLEU are fundamentally inadequate. A generated function can be lexically distant from a reference implementation and still be perfectly correct — or lexically near-identical with a critical one-character bug. **Functional correctness** instead asks: _does the code actually run and pass the unit tests?_

The canonical benchmark is **HumanEval** (OpenAI, 2021): 164 hand-crafted Python programming problems, each with a function signature, docstring, and an average of 7.7 unit tests. The evaluation metric is **pass@k**.

### pass@k Formula

Given `n` generated samples per problem (e.g., n=200) and `c` correct samples:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `C(a, b)` is the combinatorial "a choose b". This gives the probability that **at least one** of k randomly drawn samples is correct, estimated without high variance. Common values: `pass@1`, `pass@10`, `pass@100`.

### Interpretation

- `pass@1` — probability the first attempt is correct; relevant for production autocomplete.
- `pass@10` — probability that at least one of 10 samples works; relevant for interactive code assistants.
- Higher k values simulate a developer who iterates on AI suggestions.

### Python Implementation (using the `evaluate` library)

```python
# Install: pip install evaluate datasets
import evaluate

code_eval = evaluate.load("code_eval")

# Simulated: 2 problems, 3 candidate solutions each
test_cases = ["assert add(2, 3) == 5", "assert multiply(2, 3) == 6"]
candidates = [
    ["def add(a, b): return a + b",       # correct
     "def add(a, b): return a - b",       # wrong
     "def add(a, b): return a * b"],      # wrong
    ["def multiply(a, b): return a * b",  # correct
     "def multiply(a, b): return a + b",  # wrong
     "def multiply(a, b): return a ** b"], # wrong
]

results, outputs = code_eval.compute(
    references=test_cases,
    predictions=candidates,
    k=[1, 2, 3],
)
print(results)
# {'pass@1': 0.5, 'pass@2': 0.75, 'pass@3': 1.0}
```

### Running the Full HumanEval Benchmark

```python
# Install: pip install human-eval
from human_eval.data import read_problems, write_jsonl

problems = read_problems()

def generate_one_completion(prompt: str) -> str:
    # Replace with your LLM call
    return "    pass"

samples = [
    {"task_id": task_id, "completion": generate_one_completion(problems[task_id]["prompt"])}
    for task_id in problems
]
write_jsonl("samples.jsonl", samples)
# Then evaluate:
# $ evaluate_functional_correctness samples.jsonl
```

### Using DeepEval for HumanEval

```python
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask

benchmark = HumanEval(tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS], n=10, k=3)
benchmark.evaluate(model=your_model)
print(benchmark.overall_score)
```

### Important Caveats

- Executing untrusted LLM-generated code requires **sandboxing** (timeouts, resource limits, network isolation).
- HumanEval's original test suites are small (~7 tests/problem); **HumanEval+** expands to ~616 tests/problem for more rigorous coverage.
- Data contamination is a real risk — models may have seen HumanEval problems in training. Supplement with internal or fresh benchmarks.

---

## 10. Modern & Specialized Metrics

### G-Eval (LLM-as-Judge)

G-Eval uses a powerful LLM (e.g., GPT-4) with a chain-of-thought prompt to score generated text against user-defined criteria. It is highly flexible and correlates well with human judgment for open-ended tasks like coherence, helpfulness, and factual accuracy.

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the actual output is factually correct relative to the expected output.",
        "Penalise missing critical information.",
        "Accept paraphrasing and style differences.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris is the capital of France.",
)
correctness_metric.measure(test_case)
print(correctness_metric.score)   # e.g., 0.95
print(correctness_metric.reason)  # explains the score
```

### BLEURT & COMET

Learned metrics trained on human quality judgments. They capture semantic and fluency aspects that n-gram metrics miss and are particularly strong for translation evaluation.

```python
import evaluate

bleurt = evaluate.load("bleurt", config_name="bleurt-20")
results = bleurt.compute(
    predictions=["The cat is on the mat."],
    references=["The feline rested on the carpet."]
)
print(results["scores"])  # higher = more similar to reference
```

### RAG-Specific Metrics (RAGAS)

For Retrieval-Augmented Generation systems, standard metrics don't capture retrieval quality or faithfulness to source documents.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

data = {
    "question": ["What year was Python created?"],
    "answer": ["Python was created in 1991 by Guido van Rossum."],
    "contexts": [["Python is a programming language created by Guido van Rossum, released in 1991."]],
    "ground_truth": ["Python was first released in 1991."]
}

dataset = Dataset.from_dict(data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
```

|RAGAS Metric|What it Measures|
|---|---|
|Faithfulness|Does the answer contradict the retrieved context?|
|Answer Relevancy|Is the answer relevant to the question?|
|Context Precision|Are retrieved chunks ranked with the most relevant first?|
|Context Recall|Does the retrieved context cover all facts needed to answer?|

### Distinct-n (Diversity)

Measures the ratio of unique n-grams to total n-grams across generated outputs. High Distinct-n indicates varied, non-repetitive generations.

```python
from collections import Counter

def distinct_n(texts: list[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

texts = [
    "The cat sat on the mat.",
    "A dog ran in the park.",
    "The cat sat on the mat.",   # duplicate
]
print(f"Distinct-1: {distinct_n(texts, 1):.4f}")
print(f"Distinct-2: {distinct_n(texts, 2):.4f}")
```

---

## 11. Evaluation Frameworks

### Hugging Face `evaluate`

The most accessible library for classic NLP metrics. It provides a consistent API for BLEU, ROUGE, METEOR, BERTScore, perplexity, code_eval, and many more, all with one-liner loading.

```bash
pip install evaluate datasets
```

```python
import evaluate

# Load any metric by name
bleu   = evaluate.load("bleu")
rouge  = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bs     = evaluate.load("bertscore")
ppl    = evaluate.load("perplexity", module_type="metric")
ce     = evaluate.load("code_eval")
```

**Best for:** individual metric computation, dataset evaluation pipelines, research reproducibility.

---

### DeepEval

An open-source framework designed as "pytest for LLMs". It provides 14+ built-in metrics for RAG pipelines, chatbots, and agents, with native pytest integration for CI/CD.

```bash
pip install deepeval
```

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_rag_response():
    test_case = LLMTestCase(
        input="What causes thunder?",
        actual_output="Thunder is caused by the rapid expansion of air heated by lightning.",
        retrieval_context=["Lightning heats surrounding air rapidly, causing it to expand and create a shockwave heard as thunder."]
    )
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8),
    ])
```

Run with: `deepeval test run test_my_llm.py`

**Best for:** LLM unit testing, CI/CD regression, RAG system validation.

---

### RAGAS

Purpose-built for RAG pipeline evaluation, with metrics for both retrieval and generation quality. Minimal setup, works with or without ground truth.

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": ["..."],
    "answer":   ["..."],
    "contexts": [["...", "..."]],
    "ground_truth": ["..."],
})
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results.to_pandas())
```

**Best for:** RAG-specific evaluation, quick prototyping, teams without ground truth labels.

---

### LangSmith

A managed observability and evaluation platform from the LangChain team. It logs all LLM calls, supports human annotation, A/B prompt testing, and custom evaluators.

```python
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

client = Client()

def my_evaluator(run, example):
    score = 1.0 if run.outputs["output"] == example.outputs["expected"] else 0.0
    return {"key": "exact_match", "score": score}

results = ls_evaluate(
    "my-dataset",          # dataset name in LangSmith
    llm_or_chain_factory=your_chain,
    evaluators=[my_evaluator],
)
```

**Best for:** LangChain/LangGraph applications, production monitoring, teams needing a hosted UI.

---

### MLflow LLM Evaluate

An extension of MLflow's experiment tracking for LLM evaluation. Ideal for teams already using MLflow who want to add LLM evaluation alongside traditional ML metrics.

```python
import mlflow
import pandas as pd

eval_data = pd.DataFrame({
    "inputs": ["What is ML?", "Explain neural networks."],
    "ground_truth": ["Machine learning is...", "Neural networks are..."],
})

with mlflow.start_run():
    results = mlflow.evaluate(
        model="endpoints:/my-model",
        data=eval_data,
        targets="ground_truth",
        model_type="question-answering",
        evaluators="default",
    )
    print(results.metrics)
```

**Best for:** MLOps teams, experiment comparison dashboards, mixed ML + LLM workflows.

---

### Framework Comparison

|Framework|Best For|Open Source|RAG Metrics|Code Eval|CI/CD|Hosted UI|
|---|---|---|---|---|---|---|
|`evaluate` (HF)|Classic NLP metrics|✅|❌|✅|Manual|❌|
|DeepEval|LLM unit testing|✅|✅|✅|✅ pytest|Optional|
|RAGAS|RAG pipelines|✅|✅|❌|Manual|❌|
|LangSmith|LangChain observability|Managed SaaS|✅|❌|✅|✅|
|MLflow Evaluate|MLOps integration|✅|Partial|❌|✅|✅|
|TruLens|Transparency, auditing|✅|✅|❌|Manual|✅|
|Arize Phoenix|Production monitoring|✅|✅|❌|❌|✅|

---

## 12. Choosing the Right Metric

### By Task

|Task|Primary Metrics|Secondary|
|---|---|---|
|Language model pre-training|Perplexity|—|
|Machine translation|BLEU (sacrebleu), COMET|METEOR, BERTScore|
|Text summarisation|ROUGE-1, ROUGE-2, ROUGE-L|BERTScore, BLEURT|
|Code generation|pass@k (HumanEval)|Distinct-n, execution time|
|Question answering|Exact Match, F1|Semantic similarity, BERTScore|
|RAG / document QA|Faithfulness, Context Precision|Answer relevancy, context recall|
|Chatbots / dialogue|G-Eval, Human eval|Distinct-n, coherence|
|Open-ended generation|G-Eval, Human eval|BERTScore, Distinct-n|

### Decision Heuristic

```
1. Is this a code generation task?
   → Yes: Use pass@k (HumanEval or custom test suite)
   → No: Continue ↓

2. Do I have reference outputs (ground truth)?
   → Yes: Use BLEU/ROUGE for quick baseline + BERTScore for semantic depth
   → No: Use perplexity or LLM-as-judge (G-Eval)

3. Is it a RAG / retrieval system?
   → Yes: Add RAGAS metrics (faithfulness, answer relevancy, context precision)

4. Is semantic meaning more important than exact wording?
   → Yes: Prioritise BERTScore and semantic similarity over BLEU/ROUGE

5. Is the task open-ended or creative?
   → Yes: Human evaluation + G-Eval; n-gram metrics are insufficient

6. Do I need to track production quality over time?
   → Yes: Add LangSmith / Arize Phoenix / Langfuse for monitoring
```

---

## 13. Quick-Reference Summary Table

|Metric|Reference Needed|Captures Semantics|Typical Task|Score Range|Lower/Higher = Better|
|---|---|---|---|---|---|
|Perplexity|No|No|LM training|1 – ∞|Lower|
|BLEU|Yes|No|Translation|0 – 1|Higher|
|ROUGE-N|Yes|No|Summarisation|0 – 1|Higher|
|ROUGE-L|Yes|No|Summarisation|0 – 1|Higher|
|METEOR|Yes|Partial (synonyms)|Translation, QA|0 – 1|Higher|
|BERTScore F1|Yes|Yes|Any generation|0 – 1|Higher|
|Cosine Similarity|Yes|Yes|QA, retrieval|-1 – 1|Higher|
|pass@k|Unit tests|N/A (execution)|Code generation|0 – 1|Higher|
|G-Eval|Optional|Yes (LLM judge)|Any|0 – 1|Higher|
|BLEURT|Yes|Yes (learned)|Translation|Unbounded|Higher|
|Faithfulness (RAG)|Context|Yes|RAG|0 – 1|Higher|

---

## Installation Summary

```bash
# Core NLP metrics
pip install evaluate datasets sacrebleu rouge-score nltk

# Embedding-based
pip install bert-score sentence-transformers scikit-learn

# Code evaluation
pip install human-eval code-eval

# Evaluation frameworks
pip install deepeval ragas langsmith mlflow

# Additional
pip install transformers torch
```

---

_Report compiled April 2026. Metric implementations verified against library versions current as of this date. Always pin library versions in production evaluation pipelines to ensure reproducibility._