# Model distillation

## Model Distillation in AI Engineering

Model distillation is a training technique where a smaller "student" model is trained to mimic the behavior of a larger, more capable "teacher" model. Rather than training the student on hard labels (the final correct answer), you train it on the **soft outputs** — the full probability distribution the teacher produces over all possible tokens/classes.

### Why Soft Labels Matter

When a teacher model predicts the next token, it doesn't just say "the answer is Paris." It says something like:

- "Paris": 72%
- "London": 14%
- "Berlin": 8%
- "Rome": 4%
- ...

That distribution encodes **relational knowledge** — the model "knows" that London and Berlin are plausible in a way that "banana" is not. Training a student on this richer signal is far more data-efficient than training on one-hot hard labels.

The loss function typically combines:

```python
# Distillation loss (KL divergence from teacher's soft distribution)
distillation_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)

# Standard cross-entropy on ground truth labels
student_loss = F.cross_entropy(student_logits, true_labels)

# Combined loss
total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
```

The **temperature** `T` softens the distribution further — higher T reveals more of the model's uncertainty structure.

---

## Distillation vs. Other Finetuning Methods

|Method|Data Needed|Compute|Best For|
|---|---|---|---|
|**Full Finetuning**|Labeled examples|High|Domain shift with lots of data|
|**LoRA / PEFT**|Labeled examples|Low–Medium|Efficient task adaptation|
|**RLHF**|Human preference pairs|High|Alignment, instruction following|
|**Distillation**|Teacher model outputs|Medium|Compression, deployment efficiency|
|**Prompt Tuning**|Labeled examples|Very Low|Soft task conditioning|

---

## When to Use Distillation

### ✅ Use distillation when:

**1. You need to deploy at scale and cost matters** You've built a great pipeline using GPT-4 or Claude Opus, but inference cost is prohibitive. Distill it into a 7B model.

```
Teacher: Claude Opus / GPT-4 (expensive, slow)
Student: Llama 3 8B (cheap, fast, nearly as good on your task)
```

**2. You have a capable teacher but limited labeled data** The teacher's soft outputs act as a form of data augmentation — you get more signal per example.

**3. Latency is a hard constraint** Edge deployment, real-time applications, or mobile inference all require small models. Distillation helps you compress without retraining from scratch.

**4. You want to transfer a specific capability** Example: A large model is excellent at structured JSON extraction. You distill _just that behavior_ into a smaller model rather than general instruction-following.

---

## Concrete Examples

### Example 1: LLM API Cost Reduction

A startup uses GPT-4 for customer support classification (routing tickets). It costs $0.03/call at 50k calls/day = **$1,500/day**.

**Distillation approach:**

1. Run GPT-4 on 100k historical tickets → collect soft-label distributions
2. Finetune `distilbert-base` or a small Llama model on those outputs
3. Deploy the student: cost drops to ~$0.0001/call

### Example 2: Chain-of-Thought Distillation

A teacher model produces long reasoning traces (chain-of-thought). You distill both the reasoning _and_ the final answer into the student — this is how models like **Phi-3** and **Gemma** achieve outsized performance at small sizes.

```python
# Teacher generates reasoning + answer
teacher_output = "Let's think step by step: ... Therefore, the answer is 42."

# Student is trained to replicate this full output
# Not just the final label "42"
```

### Example 3: LoRA vs Distillation Decision Point

You want a model that writes SQL from natural language.

- **If you have 10k labeled (question → SQL) pairs**: Use **LoRA finetuning** on a base model. Fast, cheap, straightforward.
- **If you have only 500 pairs but access to GPT-4**: Use **distillation** — have GPT-4 generate SQL + confidence distributions for your 500 prompts + augmented variations, then train the student on those rich signals.
- **If quality must be maximized regardless of size**: **Full finetune** a large model with all available data.

---

## Key Takeaway

Distillation is fundamentally a **knowledge transfer + compression** strategy. It's the right tool when you have a capable teacher, need a lightweight student, and care about preserving the _quality of reasoning_ rather than just fitting labels. It shines in production environments where the gap between research-grade models and deployable models is a real engineering constraint.