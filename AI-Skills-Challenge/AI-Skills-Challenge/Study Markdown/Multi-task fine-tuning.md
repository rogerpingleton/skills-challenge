# Multi-task fine-tuning

## Multi-Task Fine-Tuning (MTFT)

Multi-task fine-tuning is the practice of training a model **simultaneously on multiple tasks** at once, rather than fine-tuning on a single task. Instead of optimizing for one objective, the model learns a shared representation that generalizes across all tasks in the training mix.

---

### The Core Problem It Solves

When you fine-tune a model on a single task, two problems emerge:

- **Catastrophic forgetting** — the model overwrites previously learned knowledge to specialize, losing general capabilities.
- **Poor generalization** — a model trained only on sentiment analysis, for example, becomes brittle outside that narrow distribution.

MTFT addresses both by maintaining diversity in the training signal.

---

### How It Works

You construct a **mixed dataset** from multiple tasks and train on all of them, typically with some form of sampling strategy to balance them.

```python
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Load multiple task datasets
summarization = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
translation    = load_dataset("opus_books",   "en-fr",  split="train[:5000]")
qa             = load_dataset("squad",                  split="train[:5000]")

# Format each with a task prefix so the model distinguishes tasks
def format_summarization(ex):
    return {"input": f"summarize: {ex['article']}", "target": ex["highlights"]}

def format_translation(ex):
    return {"input": f"translate English to French: {ex['en']}", "target": ex["fr"]}

def format_qa(ex):
    return {"input": f"question: {ex['question']} context: {ex['context']}", "target": ex["answers"]["text"][0]}

summarization = summarization.map(format_summarization)
translation   = translation.map(format_translation)
qa            = qa.map(format_qa)

# Mix them together
mixed_dataset = concatenate_datasets([summarization, translation, qa]).shuffle(seed=42)
```

The **task prefix** (`summarize:`, `translate:`, `question:`) is the key signal — it's how T5, FLAN, and similar models learn to route to the right behavior.

---

### Sampling Strategies

How you mix tasks matters enormously. Three common strategies:

|Strategy|Description|Best For|
|---|---|---|
|**Uniform**|Equal samples from each task|When tasks are similar in size/difficulty|
|**Temperature-scaled**|Proportional sampling with smoothing factor `T`|Most production cases|
|**Task-weighted**|Manual weights per task|When some tasks are higher priority|

```python
import numpy as np

def temperature_scaled_sampling(dataset_sizes: dict, temperature: float = 2.0) -> dict:
    """
    Higher temperature → more uniform. Lower → proportional to dataset size.
    T=1 is pure proportional; T→∞ is uniform.
    """
    sizes  = np.array(list(dataset_sizes.values()))
    scaled = sizes ** (1.0 / temperature)
    probs  = scaled / scaled.sum()
    return dict(zip(dataset_sizes.keys(), probs))

sizes = {"summarization": 50000, "translation": 200000, "qa": 10000}
print(temperature_scaled_sampling(sizes, temperature=2.0))
# {'summarization': 0.347, 'translation': 0.467, 'qa': 0.186}
```

---

### Instruction Fine-Tuning: MTFT at Scale

The most influential application of MTFT is **instruction fine-tuning** — the technique behind FLAN, InstructGPT, and most modern chat models.

The idea: frame _every_ NLP task as an instruction-following problem across hundreds of task types simultaneously.

```python
# Each task becomes an instruction/response pair
instruction_examples = [
    # Classification
    {"instruction": "Is this review positive or negative? 'The food was cold and tasteless.'",
     "response": "Negative"},

    # Summarization
    {"instruction": "Summarize this in one sentence: [long article...]",
     "response": "Researchers found that..."},

    # Code generation
    {"instruction": "Write a Python function that reverses a string.",
     "response": "def reverse_string(s):\n    return s[::-1]"},

    # Reasoning
    {"instruction": "If Alice has 3 apples and gives 1 to Bob, how many does Alice have?",
     "response": "Alice has 2 apples."},
]
```

**FLAN (Google, 2022)** demonstrated that fine-tuning on 60+ tasks dramatically improved zero-shot performance on _unseen_ tasks — the clearest proof that MTFT builds transferable reasoning.

---

### Key Challenges to Know

**1. Task conflict / negative transfer** Some tasks compete. Code generation and creative writing can pull the model in opposite directions. Mitigation: group tasks by similarity, use task-specific adapter layers (e.g. LoRA per task).

**2. Dataset imbalance** A 1M-sample translation dataset will dominate a 5K QA dataset without correction. Always apply temperature scaling or capping.

**3. Evaluation complexity** Each task needs its own metric (ROUGE for summarization, BLEU for translation, F1 for QA). You need a **per-task evaluation harness**, not a single loss number.

```python
from evaluate import load

metrics = {
    "summarization": load("rouge"),
    "translation":   load("sacrebleu"),
    "qa":            load("squad"),
}

def evaluate_all_tasks(model, task_datasets):
    results = {}
    for task, dataset in task_datasets.items():
        preds   = generate_predictions(model, dataset)
        results[task] = metrics[task].compute(predictions=preds, references=dataset["target"])
    return results
```

**4. Gradient conflict** Gradients from different tasks can cancel each other out. Advanced mitigation: **gradient surgery**(PCGrad) or **multi-task loss weighting** (e.g., GradNorm).

---

### MTFT vs. Related Techniques

|Technique|What's Trained|When to Use|
|---|---|---|
|**Single-task FT**|One task|Max performance on one task, data is plentiful|
|**Multi-task FT**|Many tasks jointly|Need generalization, limited per-task data|
|**Sequential FT**|Tasks one after another|Risk of catastrophic forgetting|
|**PEFT / LoRA**|Lightweight adapters|Low compute, many tasks needing isolation|
|**RLHF**|Human preference signal|Alignment on top of MTFT base|

In practice, modern LLM training pipelines layer these: **pretrain → MTFT (instruction tuning) → RLHF**.

---

### Practical Takeaways for an AI Engineer

1. **Always use task prefixes** when mixing tasks in a seq2seq or decoder-only setup.
2. **Start with temperature `T=2`** for sampling — it's a reliable default before tuning.
3. **Monitor per-task loss curves** separately; a single aggregate loss hides regressions.
4. **More diverse tasks = better zero-shot generalization**, even on tasks not in your training mix.
5. **FLAN-T5 and Mistral-Instruct** are strong starting checkpoints — they're already instruction-tuned via MTFT and respond well to further task-specific fine-tuning.
6. Use **LoRA per task** when tasks genuinely conflict — shared backbone, isolated task heads.