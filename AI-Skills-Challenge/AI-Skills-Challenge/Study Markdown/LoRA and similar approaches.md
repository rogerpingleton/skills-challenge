# LoRA and similar approaches

## 1. Background: Why Fine-Tuning Matters

Large language models (LLMs) are trained on massive, general-purpose corpora and perform impressively out of the box. But production AI systems frequently demand behavior that a base model cannot reliably deliver: strict output formatting, proprietary domain vocabulary, consistent brand voice, or highly specialized reasoning patterns. There are three primary customization strategies available to an engineer:

- **Prompt engineering** — Shape behavior entirely through carefully crafted system and user prompts, with zero model changes. Fast and cheap, but brittle at scale.
- **Retrieval-Augmented Generation (RAG)** — Inject external knowledge at inference time via a retrieval step. Excellent for dynamic or proprietary facts, but adds infrastructure and latency.
- **Fine-tuning** — Actually update the model's parameters using domain-specific examples so that the desired behavior is _baked into the weights_.

Fine-tuning is the right choice when you have a **behavioral problem, not a knowledge problem**. If prompt engineering produces inconsistent formatting across thousands of daily requests, if RAG cannot supply the right contextual reasoning patterns, or if your task requires deep fluency in specialized language, fine-tuning is warranted. The conventional barrier to fine-tuning was prohibitive compute cost — updating every parameter of a 70-billion-parameter model requires enormous GPU clusters. This is precisely the problem that **LoRA** was invented to solve.

---

## 2. The PEFT Landscape

**Parameter-Efficient Fine-Tuning (PEFT)** is an umbrella of techniques that adapt a large model to a new task by updating only a small fraction of its parameters while keeping the rest frozen. The payoff is dramatic: you capture most of the benefit of full fine-tuning at a tiny fraction of the memory and compute cost.

The dominant PEFT methods are:

|Method|Approach|Best For|
|---|---|---|
|**LoRA**|Inject trainable low-rank matrices into attention layers|General task and domain adaptation|
|**QLoRA**|LoRA on a 4-bit quantized base model|Large models on consumer/budget hardware|
|**Prefix Tuning**|Prepend trainable "virtual tokens" to each layer|Few-shot, low-data scenarios|
|**Adapter Layers**|Insert small trainable modules between transformer layers|Multi-task serving|
|**IA³**|Scale activations with learned vectors|Very low data regimes|

Among these, **LoRA and its derivatives dominate in practice** due to their simplicity, strong empirical performance, and deep ecosystem support via Hugging Face `peft`, `trl`, and popular frameworks like Unsloth and LLaMA-Factory.

---

## 3. LoRA: Low-Rank Adaptation — The Math and Mechanics

### The Core Idea

A transformer weight matrix `W` has dimensions `d × k`. Full fine-tuning would update every element of `W`. LoRA's key insight — backed by both theory and experiment — is that the _update_ to `W` during fine-tuning has **low intrinsic dimensionality**: you don't need to change every element to shift the model's behavior. The adaptation can be well-approximated by a low-rank matrix.

Instead of updating `W` directly, LoRA **freezes** `W` and adds a **residual low-rank branch**:

```
W_new = W + ΔW = W + B × A

Where:
  W  : frozen pretrained weights  (d × k)
  B  : trainable matrix           (d × r)
  A  : trainable matrix           (r × k)
  r  : rank (typically 8–64, much smaller than d or k)
```

During a forward pass, the output becomes:

```
h = Wx + (BA)x × (alpha / r)
```

The `alpha / r` scaling factor (`lora_alpha / lora_r`) controls how much the adapter contribution is weighted relative to the frozen base. Only `B` and `A` accumulate gradients; the original `W` is never touched.

### Why This Works

For a 70B LLaMA model with `r=16`, LoRA introduces only approximately **0.29% additional parameters**. Yet this tiny slice of trainable weights captures the directional changes needed to adapt the model's behavior. Empirically, LoRA consistently achieves 95–98% of full fine-tuning performance at a fraction of the cost.

### What Layers Are Targeted

LoRA can be applied to any linear layer. The most common targets are:

- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP layers:** `gate_proj`, `up_proj`, `down_proj`
- **Token embeddings and LM head** (needed when fine-tuning for new chat templates)

The Hugging Face `peft` library supports `target_modules="all-linear"` to apply LoRA to every linear layer in the model, which is generally the recommended default for instruction fine-tuning in 2025–2026.

---

## 4. QLoRA: Quantized LoRA

### The Memory Problem

Even LoRA training requires loading the full base model into GPU memory to run forward and backward passes through it. A 70B model in 16-bit precision occupies ~140 GB — far beyond a single GPU's capacity.

**QLoRA** (Dettmers et al., 2023) solves this by quantizing the frozen base model weights to **4-bit precision** using a special format called **NF4 (Normal Float 4)**, which is optimized for normally distributed neural network weights. The LoRA adapter matrices (`A` and `B`) remain in higher precision (bfloat16 or float16) during training.

### How QLoRA Works in Practice

```
Training step:
  1. Load base model W in 4-bit NF4 (stored in VRAM at 1/4 the memory)
  2. Dequantize W to float16/bfloat16 for each forward pass computation
  3. Compute output: h = Wx + BAx
  4. Run backward pass, accumulate gradients only on B and A
  5. W stays 4-bit; never receives gradient updates
```

QLoRA also introduces **Double Quantization** — quantizing the quantization constants themselves — saving additional memory.

### The QLoRA Trade-Off

QLoRA reduces base model memory by roughly **4×** (e.g., a 13B model drops from ~26 GB to ~6.5 GB), enabling fine-tuning of models that previously required multi-GPU setups on a single consumer GPU. The cost is:

- **~30% slower training** due to the extra dequantization step on every forward pass.
- A **slight performance degradation** versus standard LoRA on the same model (though this is often imperceptible in practice).
- Slightly **more LoRA adapters** are typically needed to compensate for quantization noise.

### QLoRA Python Setup (BitsAndBytes + PEFT)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization for extra memory savings
)

# 2. Load the quantized base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",                   # Spread across available GPUs/CPU
)

# 3. Configure LoRA adapters
lora_config = LoraConfig(
    r=16,                                # Rank
    lora_alpha=16,                       # Scaling factor
    target_modules="all-linear",         # Apply to all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 4. Wrap model with PEFT/LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,030,261,248 || trainable%: 0.52%
```

---

## 5. The LoRA Variant Ecosystem (2025–2026)

The success of LoRA sparked a family of derivatives, each addressing a specific limitation:

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes the weight matrix into **magnitude** (a scalar per output channel) and **direction** (the unit vector), and applies LoRA only to the directional component. The magnitude is updated separately. The result is more expressive updates that better mimic full fine-tuning's learning dynamics. Benchmarks show consistent improvements: +3.7% on LLaMA-7B and +1–4.4% on LLaMA-13B on commonsense reasoning tasks with no inference overhead (since the decomposition can be merged back into the base weights). In Hugging Face PEFT, it's a single flag: `use_dora=True`.

### rsLoRA (Rank-Stabilized LoRA)

Standard LoRA uses a fixed scaling factor of `alpha / r`. This causes the effective learning rate to change as you increase rank `r`, making hyperparameter tuning at different ranks inconsistent. rsLoRA replaces this with `alpha / sqrt(r)`, which stabilizes training across ranks and allows safely using higher ranks (r=64, r=128) without the instability that previously accompanied them.

### LoftQ

When you quantize the base model to 4-bit for QLoRA, you introduce quantization noise. LoftQ addresses this by **jointly optimizing the quantization and the initial LoRA matrices** using iterative SVD, so the adapters start from a position that already compensates for quantization error. This gives QLoRA a better starting point and can close the performance gap versus standard LoRA.

### PiSSA (Principal Singular Values and Singular Vectors Adaptation)

Standard LoRA initializes `A` with random Gaussian values and `B` with zeros, meaning the adapter starts from zero effect. PiSSA instead initializes `A` and `B` from the **principal components (SVD decomposition)** of the base weight matrix. The adapter therefore begins already encoding the most important structure of the weights, which can accelerate convergence.

### LongLoRA

Specifically designed for extending a model's context window. Standard self-attention scales quadratically with sequence length, making long-context fine-tuning extremely expensive. LongLoRA uses **Shift Short Attention (S²-Attn)**: it divides tokens into short chunks and computes attention within each chunk independently, massively reducing the compute required to fine-tune models for long contexts.

### QALoRA

An extension of QLoRA where the LoRA adapter weights themselves are also quantized, not just the base model weights. This enables more memory-efficient training as no precision conversion step is needed during backpropagation.

---

## 6. Where Do LoRA Adapters Live?

This is a crucial operational question with several answers depending on deployment strategy.

### As Separate Checkpoint Files

A trained LoRA adapter is saved as a small set of files, typically:

```
my-lora-adapter/
├── adapter_config.json      # LoRA hyperparameters: r, alpha, target_modules, etc.
├── adapter_model.safetensors  # The actual trained B and A matrices
└── tokenizer_config.json    # If the tokenizer was modified
```

The size of this artifact is tiny compared to the full model. For a 7B parameter base model with LoRA `r=16` on all linear layers, the adapter might be only **50–200 MB** while the base model is ~14 GB. This is one of LoRA's most valuable properties: you can maintain many task-specific adapters at low storage cost while sharing a single base model.

### Merged Into the Base Model

Adapters can be merged back into the base weights before deployment:

```python
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# Merge: W_merged = W + BA (no inference overhead)
merged_model = model.merge_and_unload()

# Save the merged model as a standalone checkpoint
merged_model.save_pretrained("./my-merged-model")
```

Once merged, the adapter matrices mathematically disappear into the base weights. There is no inference overhead and no dependency on the PEFT library at serving time. The tradeoff is that you lose the ability to swap adapters at runtime.

### Loaded Dynamically at Inference Time (Multi-LoRA Serving)

In production systems serving multiple specialized adapters, frameworks like **vLLM**, **TGI (Text Generation Inference)**, **LoRAX**, and **S-LoRA** support loading the base model once and dynamically swapping adapters per request:

```python
# Conceptual example using S-LoRA / vLLM adapter serving
server.register_adapter("medical",   "./lora-medical")
server.register_adapter("legal",     "./lora-legal")
server.register_adapter("code",      "./lora-code")
server.register_adapter("customer-support", "./lora-cs")

# Each incoming request specifies the adapter to use
response = server.generate(prompt, adapter_name="medical")
```

This architecture is transformative for organizations that need many specialized models: they maintain one large base model in GPU memory and hot-swap adapters for each request, rather than running dozens of separate model instances. Azure AI's serverless endpoints for fine-tuned Llama models use exactly this pattern — the LoRA adapters are kept in memory and swapped on demand, with a small per-hour surcharge for the adapter hosting.

---

## 7. Integration with Commercial LLMs

### Open-Weight Models (Llama, Mistral, Qwen, Phi, etc.)

Open-weight models are the most natural target for LoRA training. You have full access to weights, can use any training framework, and can deploy the result anywhere. The typical stack is:

- **Hugging Face `transformers`** — model loading, tokenization
- **Hugging Face `peft`** — LoRA/QLoRA adapter management
- **Hugging Face `trl`** — SFT, DPO, ORPO trainers
- **`bitsandbytes`** — 4-bit quantization for QLoRA
- **Unsloth** — heavily optimized training kernels, 2× faster than standard PEFT on consumer hardware
- **LLaMA-Factory** — full-featured GUI + config-file-driven training

### OpenAI (GPT-4o, GPT-3.5-turbo)

OpenAI's fine-tuning API is a managed service. You upload a JSONL dataset, configure hyperparameters, and OpenAI trains and hosts a fine-tuned variant of the model on your behalf. Internally, OpenAI uses LoRA-style PEFT, but this is abstracted from the user. You receive a model ID (e.g., `ft:gpt-4o-mini-2024-07-18:my-org::abc123`) and call it through the standard Chat Completions API. The adapter effectively lives within OpenAI's infrastructure. Pricing as of 2025: **$0.025 per 1K training tokens** for GPT-4o, with inference at $0.00375/1K input tokens.

### Azure AI Foundry (OpenAI and OSS Models)

Azure AI Foundry provides a managed fine-tuning workflow for both OpenAI models and open-source models (Llama, Phi, Mistral). It exposes LoRA hyperparameters (rank, alpha, learning rate, batch size, epochs) through a UI or SDK. For open-source models, Azure's serverless deployment infrastructure manages LoRA adapters in GPU memory and hot-swaps them per request. The Llama 3.1 8B hosted LoRA endpoint was priced at ~$0.74/hour for adapter hosting (down from $3.09/hour for Llama 2 7B), plus per-token inference costs.

### Completely Closed Models (Claude, Gemini)

Anthropic's Claude and Google's Gemini (via Vertex AI) do not expose weights for external LoRA fine-tuning. Customization is done through **API-side fine-tuning services** (where available) or through prompt engineering and RAG. As of April 2026, Anthropic does not offer direct LoRA fine-tuning access to Claude's weights; customization relies on prompt design, system prompts, and tool use.

---

## 8. Training a LoRA: A Practical Walkthrough

The following is a complete example fine-tuning `meta-llama/Meta-Llama-3.1-8B` for a medical question-answering task using QLoRA and Hugging Face `trl`.

### Step 1: Install Dependencies

```bash
pip install transformers peft trl bitsandbytes accelerate datasets
# Optional but recommended for speed:
pip install unsloth flash-attn
```

### Step 2: Prepare Your Dataset

Data should be formatted as prompt-completion pairs. For instruction tuning:

```python
# dataset format: JSONL with "messages" key (ChatML format)
[
  {
    "messages": [
      {"role": "system",  "content": "You are a clinical assistant."},
      {"role": "user",    "content": "What are the symptoms of Type 2 diabetes?"},
      {"role": "assistant","content": "Common symptoms include..."}
    ]
  },
  ...
]
```

Quality matters far more than quantity. 1,000 carefully curated examples often outperform 100,000 noisy ones.

### Step 3: Configure and Train

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# --- QLoRA Configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Load Base Model ---
model_id = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # faster attention if supported
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Uncomment for DoRA variant:
    # use_dora=True,
)

# --- Load Dataset ---
dataset = load_dataset("json", data_files="medical_qa.jsonl", split="train")

# --- Training Arguments ---
training_config = SFTConfig(
    output_dir="./llama-3.1-8b-medical-qlora",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch size = 16
    gradient_checkpointing=True,        # trade compute for memory
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=25,
    save_strategy="epoch",
    max_seq_length=2048,
    packing=True,                       # pack short sequences for efficiency
)

# --- Train ---
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()

# --- Save Adapter Only ---
trainer.model.save_pretrained("./llama-3.1-8b-medical-adapter")
tokenizer.save_pretrained("./llama-3.1-8b-medical-adapter")
```

### Step 4: Inference with the Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./llama-3.1-8b-medical-adapter")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipe("What are the first-line treatments for hypertension?", max_new_tokens=256)
print(response[0]["generated_text"])
```

### Step 5: Merge and Export (Optional)

```python
merged = model.merge_and_unload()
merged.save_pretrained("./llama-3.1-8b-medical-merged")
# Now deploy as a standalone model — no PEFT dependency required
```

---

## 9. Key Hyperparameters and Tuning Guidance

### LoRA Rank (`r`)

Controls the capacity of the adapter. Higher rank = more expressive updates at the cost of more parameters. 2025 research on rank selection found that intermediate ranks (32–64) offer the best balance for significant domain shifts. In practice:

|Use Case|Recommended `r`|
|---|---|
|Light instruction following / style|8–16|
|General domain adaptation|16–32|
|Significant domain shift (new specialty)|32–64|
|Very complex behavioral changes|64–128 (use rsLoRA)|

### `lora_alpha`

The scaling factor is applied as `alpha / r`. When `alpha == r`, the scaling is 1.0. A common convention is to set `alpha = r`(resulting in scale 1.0) or `alpha = 2*r` (scale 2.0, a larger adapter contribution). Experiment-driven guidance from Lightning AI: changing alpha relative to r has a significant effect, and `alpha = r` is a solid starting point.

### Target Modules

For instruction-following and domain adaptation: `target_modules="all-linear"` is the current best-practice recommendation (2025–2026). For lighter adaptation, `["q_proj", "v_proj"]` is common. Including the `lm_head` and `embed_tokens` is necessary when fine-tuning for new chat templates (e.g., Llama 3.1 with a custom format).

### Learning Rate

LoRA uses a higher learning rate than full fine-tuning because only a small parameter subset is updated. A rate of `2e-4`with a cosine schedule and 5% warmup is a widely validated starting point. Full fine-tuning typically uses `1e-5` to `5e-5`.

### Dropout (`lora_dropout`)

Regularizes the adapter to prevent overfitting. Values of `0.05` to `0.1` are common. With small datasets (<1,000 examples), `0.1` is safer. With large, clean datasets, `0.0` can work.

### Hardware Reference

|Scenario|GPU|VRAM|Model|Method|
|---|---|---|---|---|
|Consumer fine-tuning|RTX 3090/4090|24 GB|7–8B|QLoRA|
|Single professional GPU|A100 40 GB|40 GB|13B|QLoRA or LoRA|
|Comfortable single GPU|A100 80 GB|80 GB|70B|QLoRA|
|Multi-GPU cluster|8×H100|640 GB|70B|LoRA (FSDP/DeepSpeed)|

---

## 10. When Should You Train a LoRA?

Fine-tuning with LoRA earns its place in specific, well-defined scenarios. The general rule of thumb is to exhaust prompt engineering first, then RAG, before reaching for LoRA fine-tuning.

### ✅ Strong Signals to Fine-Tune

**Behavioral consistency at scale.** If prompt engineering produces variable results and you are handling thousands of requests per day, even a small inconsistency rate becomes significant. Fine-tuning bakes the desired behavior into weights, producing reliable outputs without depending on long, expensive system prompts.

**Strict output formatting.** When outputs must conform to rigid schemas (JSON with specific fields, clinical report templates, legal document structures, code in a specific dialect), fine-tuning reliably enforces format in a way that prompting often cannot.

**Domain-specific language fluency.** Models that regularly encounter jargon, abbreviations, or reasoning patterns absent from their training data will hallucinate or underperform. A LoRA fine-tuned on domain corpora learns the vocabulary and reasoning patterns natively.

**Inference cost reduction at scale.** Long system prompts and few-shot examples are paid for on every single API call. Fine-tuning bakes those instructions into the model, removing the need for large prompts and reducing per-token inference costs at high throughput.

**Smaller model reaching larger model quality.** A LoRA-fine-tuned 8B model on a specific narrow task can match or exceed a general-purpose 70B model on that task at a fraction of the inference cost.

**Behavior you cannot inject via context.** Some behaviors — tone, ethical guardrails, systematic reasoning style — are very difficult to enforce consistently through prompting alone. Fine-tuning embeds them as a model tendency.

### ❌ When NOT to Fine-Tune

**You need dynamic or frequently updated facts.** Fine-tuned weights are static. If you are trying to give the model access to current information, product catalogs, or real-time data, RAG is the right tool.

**You have not tried prompt engineering first.** Most teams over-apply fine-tuning. A well-crafted system prompt with chain-of-thought instructions and a few examples can solve 80% of behavioral problems in an afternoon. Fine-tuning takes days and thousands of dollars.

**You have less than ~100 high-quality training examples.** With too little data, the model will overfit to your examples and lose its general reasoning ability. RAG or prompt engineering are safer.

**Your task is a knowledge gap, not a behavior gap.** If the model doesn't know about your company's Q4 revenue figures or your product SKUs, that's a facts problem — solve it with RAG.

---

## 11. Real-World Use Case Examples

### Example 1: Medical Question Answering (Healthcare)

**Problem:** A hospital's AI assistant hallucinated clinical protocols and used lay language instead of standard medical terminology. Prompting alone couldn't solve it at the required consistency across 50,000 daily interactions.

**Solution:** Fine-tuned LLaMA-3.1-8B with QLoRA on ~150,000 de-identified patient interactions and clinical guidelines. Dataset formatted as `(symptom_query, clinical_response)` pairs reviewed by physicians.

**Result:** Accuracy rose significantly; high-risk clinical misinterpretations were reduced by over half. The fine-tuned 8B model matched the performance of a general-purpose 70B model on the specific clinical Q&A task at one-tenth the inference cost.

**Key LoRA Config:** `r=32`, `target_modules="all-linear"`, DoRA enabled, 2 epochs, QLoRA (NF4).

### Example 2: SQL Generation for Enterprise Analytics

**Problem:** A data analytics platform needed to translate natural language questions into SQL queries for a proprietary database schema with hundreds of tables, custom function names, and idiosyncratic naming conventions.

**Solution:** Fine-tuned Mistral-7B-Instruct using LoRA (no QLoRA needed, fits in 24 GB VRAM) on 5,000 `(question, correct_SQL)` pairs generated from the internal schema + some synthetic augmentation.

**Result:** Query accuracy on the company's schema jumped from ~45% (prompted GPT-4o) to ~91% (fine-tuned Mistral-7B). Inference costs dropped 10× by switching from GPT-4o to the hosted fine-tuned model.

**Key LoRA Config:** `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`, 3 epochs.

### Example 3: Customer Support Style Adaptation

**Problem:** A SaaS company's customer support bot used a formal, robotic tone inconsistent with their brand voice. Long system prompts enforcing tone added significant token cost at 10,000 daily interactions.

**Solution:** Fine-tuned GPT-4o-mini via the OpenAI Fine-Tuning API on 800 curated `(customer_question, ideal_brand_voice_response)` pairs written by the company's support team.

**Result:** Consistent brand voice with no system prompt overhead, saving ~$300/day in token costs. The fine-tuned model was served through the standard Chat Completions API.

### Example 4: Code Generation for a Domain-Specific Language (DSL)

**Problem:** A robotics company needed an LLM to write programs in their proprietary robot scripting language, which was absent from any training corpus.

**Solution:** Created 2,000 `(task_description, robot_script)` training pairs from their internal documentation and existing scripts. Fine-tuned CodeLlama-13B using QLoRA on 2× A100 GPUs.

**Result:** The model learned the DSL's syntax and idioms from scratch. Zero-shot performance on new tasks was ~70% (vs. ~5% for prompted GPT-4o, which had never seen the DSL).

---

## 12. Serving LoRA Adapters in Production

### Merged Model Serving (Simplest)

Merge the adapter into the base model and serve with any standard inference framework (vLLM, TGI, Ollama). No PEFT dependency, maximum portability, no switching overhead. Best when you have one adapter per deployment.

```bash
# vLLM serving after merging
docker run --runtime nvidia --gpus all -p 8000:8000 \
  vllm/vllm-openai --model ./my-merged-model
```

### Multi-Adapter Dynamic Serving (Advanced)

For serving multiple specialized adapters on shared infrastructure, use **vLLM** (with LoRA support), **LoRAX**, or **TGI**. All three load the base model once and swap adapters per request.

```python
# vLLM with LoRA adapter serving
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64,
)

# Per-request adapter specification
from vllm.lora.request import LoRARequest

outputs = llm.generate(
    prompts=["Explain the treatment for sepsis."],
    sampling_params=SamplingParams(max_tokens=256),
    lora_request=LoRARequest("medical", 1, "./lora-medical")
)
```

### Cloud Deployment (Managed)

Major cloud platforms abstract LoRA adapter management:

- **Azure AI Foundry Serverless** — Upload adapter checkpoint, get a REST endpoint. Adapters are kept hot in memory and swapped on demand. Charged per token + hourly adapter hosting fee (~$0.74/hr for Llama 3.1 8B).
- **Together.AI, Fireworks.ai, RunPod** — All support hosting custom LoRA adapters on shared base model infrastructure.
- **OpenAI Fine-Tuning API** — Fully managed; you get a model ID, call it like any other ChatCompletion.

---

## 13. Cost Considerations

|Activity|Approximate Cost|
|---|---|
|LoRA fine-tune 7B model, ~10K examples, 3 epochs (cloud A100)|$50–$150|
|LoRA fine-tune 13B model (cloud)|$200–$500|
|Full fine-tune 7B model (for comparison)|$1,000–$3,000|
|Full fine-tune 13B model (for comparison)|$10,000–$20,000|
|OpenAI GPT-4o fine-tuning (per 1K training tokens)|$0.025|
|Azure Llama 3.1 8B LoRA endpoint hosting|~$0.74/hr|
|Single A100 GPU cloud rental (per hour)|~$2–$4|

Data preparation is often 20–40% of total fine-tuning cost in terms of engineering time. Poor quality data is the single most common reason fine-tunes underperform expectations.

---

## 14. Common Pitfalls and How to Avoid Them

**Misconfigured QLoRA (silent failures).** The most common beginner trap: incorrect `bitsandbytes` configuration causes NaN losses or silent breakage. Always test a tiny batch (2–4 examples) before launching a full run. Verify `bnb_4bit_quant_type="nf4"` and that `bnb_4bit_compute_dtype` matches your `torch_dtype`.

**Training on noisy data.** The model will learn your mistakes as faithfully as your correct examples. Curate aggressively; deduplicate; have domain experts review a sample of examples before training.

**Overfitting on small datasets.** Iterating over a small dataset more than 2–3 times can hurt performance rather than help it. Use a held-out evaluation set and monitor loss on it throughout training. Higher `lora_dropout` (0.1) provides regularization.

**Wrong chat template.** For instruction fine-tuning, rendering training examples with a different chat template than the one used at inference will cause silent degradation. Always apply the tokenizer's `apply_chat_template()` method both during training and inference.

**Evaluation data leakage.** Mining web Q&A wholesale without verifying that evaluation examples are disjoint from training examples inflates reported scores. Keep test data strictly separated.

**Choosing fine-tuning before prompt engineering.** The single most expensive mistake. Always validate that prompt engineering cannot achieve 80% of your goal in far less time before committing to a fine-tuning project.

**Catastrophic forgetting.** Aggressive fine-tuning on a narrow task can degrade the model's general capabilities. Mitigate by mixing in a small fraction (~5–10%) of general-domain data in the training set, and monitoring general benchmarks (MMLU, etc.) alongside task-specific metrics.

---

## 15. Summary and Decision Framework

### The LoRA/QLoRA Decision Tree

```
Is your problem a KNOWLEDGE gap (the model lacks facts)?
  └─ YES → Use RAG (+ prompt engineering)
  └─ NO  → Is your problem a BEHAVIOR gap?
              └─ YES → Can a well-crafted prompt + few-shot examples solve it?
                          └─ YES → Use prompt engineering (fastest, cheapest)
                          └─ NO  → Do you have GPU memory constraints?
                                      └─ YES (>8B model on <40GB VRAM) → Use QLoRA
                                      └─ NO (smaller model or large GPU)  → Use LoRA
                                         └─ Need max quality? → Consider DoRA or PiSSA
                                         └─ Very large model? → Consider QLoRA + rsLoRA
```

### Quick Reference: LoRA vs QLoRA

||**LoRA**|**QLoRA**|
|---|---|---|
|Base model precision|float16 / bfloat16|4-bit NF4|
|Memory savings|~50% vs full fine-tune|~75% vs full fine-tune|
|Training speed|Faster|~30% slower (dequantization)|
|Performance|Slightly better|Near-identical to LoRA|
|GPU requirement (8B model)|~16 GB|~6–8 GB|
|When to use|Model fits in VRAM comfortably|Squeezing large models onto limited hardware|

### The Recommended Starting Configuration (2026)

For the majority of instruction-following and domain adaptation tasks, start here and adjust from results:

```
r              = 16
lora_alpha     = 16
target_modules = "all-linear"
lora_dropout   = 0.05
use_dora       = True          # small quality improvement at no inference cost
learning_rate  = 2e-4
lr_scheduler   = cosine + 5% warmup
epochs         = 2–3
quantization   = QLoRA NF4 (unless model fits comfortably in VRAM)
framework      = Hugging Face TRL + PEFT, or Unsloth for speed
```

---

_Report compiled April 2026. Sources include Hu et al. (2022) LoRA paper, Dettmers et al. (2023) QLoRA paper, Lightning AI experiments, Hugging Face fine-tuning guides (2025), and survey data from letsdatascience.com, mercity.ai, and analyticsvidhya.com._