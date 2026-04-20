# Post-training techniques - SFT, RLHF, DPO

Post-training is what transforms a raw pretrained model (which just predicts next tokens) into a useful, aligned assistant. Here's what you need to know as an AI Engineer.

---

## The Big Picture: Why Post-Training Matters

A pretrained foundation model is trained on internet-scale text with a single objective: predict the next token. It's incredibly capable, but also:

- Will complete prompts in undesirable ways ("How do I bake a cake?" → continues with a recipe _and_ unrelated text)
- Has no concept of instruction-following
- May produce harmful, biased, or confidently wrong outputs

Post-training shapes the model's _behavior_ without retraining it from scratch. The three dominant techniques are **SFT**, **RLHF**, and **DPO**.

---

## 1. Supervised Fine-Tuning (SFT)

### What it is

SFT trains the model on a curated dataset of `(prompt, ideal_response)` pairs using standard supervised learning (cross-entropy loss). It's the foundation of all post-training pipelines.

### How it works

```
Dataset:  [(prompt_1, response_1), (prompt_2, response_2), ...]
Loss:     Cross-entropy on response tokens only (not the prompt)
Goal:     Model learns to mimic the distribution of "good" responses
```

### Example

```python
# Conceptual training example (using HuggingFace TRL)
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

dataset = Dataset.from_list([
    {"text": "<user>Summarize this paper.</user><assistant>The paper proposes...</assistant>"},
    {"text": "<user>Write a SQL query for...</user><assistant>SELECT * FROM...</assistant>"},
])

trainer = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1"),
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
```

### What you need to know

- **Data quality >> data quantity.** 1,000 high-quality examples beat 100,000 noisy ones.
- SFT alone produces decent instruction-following, but the model can still be sycophantic or produce harmful outputs.
- It's the mandatory **first step** before RLHF or DPO — those techniques need a well-behaved starting point.
- Common data sources: human-written demonstrations, GPT-4 distillation (synthetic data), filtered web data.

---

## 2. Reinforcement Learning from Human Feedback (RLHF)

### What it is

RLHF optimizes the model using a **reward signal derived from human preferences**, rather than just imitating examples. It's the technique that made ChatGPT feel qualitatively different from raw GPT-3.

### The 3-Step Pipeline

```
Step 1: SFT          → Fine-tune on demonstrations (see above)
Step 2: Reward Model → Train a classifier on human preference pairs
Step 3: RL (PPO)     → Optimize the SFT model against the reward model
```

#### Step 2 — Train a Reward Model (RM)

Human annotators rank model outputs:

```
Prompt:     "Explain gravity to a 10-year-old."
Response A: "Gravity is a force that pulls objects toward each other..." ← preferred
Response B: "F = Gm₁m₂/r² is the gravitational formula..." ← not preferred

The RM learns: P(A is better than B) → scalar reward score
```

#### Step 3 — PPO Optimization

```python
# Conceptual PPO loop (using TRL's PPOTrainer)
from trl import PPOTrainer, PPOConfig

ppo_trainer = PPOTrainer(config=PPOConfig(), model=sft_model, ...)

for prompt_batch in dataloader:
    # 1. Generate responses from the current policy
    response_tensors = ppo_trainer.generate(prompt_batch)

    # 2. Score responses with the reward model
    rewards = [reward_model(p, r) for p, r in zip(prompt_batch, response_tensors)]

    # 3. PPO update — nudge the model toward higher-reward outputs
    #    KL penalty prevents the model from drifting too far from SFT model
    stats = ppo_trainer.step(prompt_batch, response_tensors, rewards)
```

### What you need to know

- The **KL divergence penalty** is critical — without it, the model "reward hacks" (finds degenerate outputs that fool the reward model).
- RLHF is **expensive and unstable**. PPO is notoriously finicky to tune.
- The reward model is a **bottleneck** — if human annotations are biased or inconsistent, the RM amplifies that.
- This is what OpenAI used for InstructGPT/ChatGPT. Anthropic used a variant called RLAIF (RL from AI Feedback) for Constitutional AI.

---

## 3. Direct Preference Optimization (DPO)

### What it is

DPO (Rafailov et al., 2023) is a cleaner alternative to RLHF that **skips the reward model entirely**. It reformulates the RL objective as a supervised classification problem directly on preference pairs.

### The Key Insight

RLHF implicitly defines an optimal policy. DPO shows you can rearrange the math to solve for that policy **directly** from preference data, without ever training a reward model or running PPO.

### The Loss Function (conceptually)

```
Given: (prompt, chosen_response, rejected_response)

DPO Loss = -log σ( β * [log π(chosen|prompt)   - log π_ref(chosen|prompt)]
                     - β * [log π(rejected|prompt) - log π_ref(rejected|prompt)] )

Where:
  π      = model being trained
  π_ref  = frozen SFT model (the reference)
  β      = temperature controlling deviation from reference
  σ      = sigmoid function
```

In plain English: **increase the probability of chosen responses _relative to the reference model_, decrease the probability of rejected ones.**

### Example

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Preference dataset format
dataset = Dataset.from_list([
    {
        "prompt": "Write a haiku about Python.",
        "chosen": "Indent with spaces\nLoops breathe through clean syntax\nGrace in simplicity",
        "rejected": "Python is a programming language that is used for many things"
    },
    # ... more pairs
])

trainer = DPOTrainer(
    model=sft_model,          # Policy to train
    ref_model=sft_model_ref,  # Frozen reference (same starting point)
    args=DPOConfig(beta=0.1),
    train_dataset=dataset,
)
trainer.train()
```

### What you need to know

- Much **simpler and more stable** than RLHF — no reward model, no PPO.
- Requires the same preference pair format as RLHF: `(prompt, chosen, rejected)`.
- **β (beta)** controls regularization strength — low β = more aggressive alignment, higher β = stays closer to the reference model.
- DPO can underperform RLHF on very complex tasks; RLHF's online nature (generating new samples during training) can be an advantage.
- Most open-source fine-tuned models today (Llama variants, Mistral instruct, etc.) use DPO or variants like **IPO**, **KTO**, or **ORPO**.

---

## Side-by-Side Comparison

||SFT|RLHF|DPO|
|---|---|---|---|
|**Data needed**|(prompt, response)|(prompt, chosen, rejected)|(prompt, chosen, rejected)|
|**Reward model**|No|Yes|No|
|**RL training**|No|Yes (PPO)|No|
|**Stability**|High|Low|High|
|**Cost**|Low|High|Medium|
|**Typical use**|First step always|SOTA alignment|Practical alternative to RLHF|

---

## Practical Decision Guide for AI Engineers

```
Are you adapting a base model for a specific task?
└── Start with SFT. Always.

Do you have human preference data (or can generate synthetic pairs)?
├── Short on compute / want simplicity?  →  Use DPO
└── Have resources & need max quality?   →  Consider RLHF (or look at RLAIF)

Are you using a model that's already instruction-tuned (e.g., Llama-3-Instruct)?
└── Skip SFT/RLHF, go straight to DPO for behavior steering.
```

### Key libraries to know

- **`trl`** (HuggingFace) — SFTTrainer, DPOTrainer, PPOTrainer. The go-to toolkit.
- **`axolotl`** — Config-driven fine-tuning, great for rapid iteration.
- **`LitGPT`** / **`unsloth`** — Efficient fine-tuning with LoRA/QLoRA support.

### On Parameter-Efficient Fine-Tuning (PEFT)

In practice, you almost never full-fine-tune. You combine these techniques with **LoRA/QLoRA**, which trains only a small set of adapter weights (~1% of parameters), making SFT and DPO feasible on consumer GPUs.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)
# Now pass this into SFTTrainer or DPOTrainer as normal
```

---

## Mental Model Summary

```
Pretraining      →  "I can predict text"
    ↓ SFT
Instruction FT   →  "I can follow instructions"
    ↓ RLHF / DPO
Aligned Model    →  "I follow instructions helpfully, honestly, and safely"
```

As an AI Engineer, your most common workflow will be: **start from an instruction-tuned base → apply DPO with domain-specific preference data → evaluate with a reward model or LLM-as-judge**. Full RLHF pipelines are typically only run by model labs with serious infrastructure budgets.