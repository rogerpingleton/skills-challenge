# Model compression - quantization, pruning, distillation

## 1. Why Model Compression Matters

Modern LLMs are staggeringly large. Models like Meta's Llama 4 Maverick exceed 400 billion parameters. Even a "modest" 70B-parameter model in standard FP16 precision requires roughly **140 GB of GPU VRAM** — equivalent to four NVIDIA A100s just to run inference.

Model compression addresses this by shrinking models along two dimensions:

- **Size reduction** — less storage and GPU memory
- **Latency optimization** — fewer compute operations → faster inference

The business case is stark. Industry data shows that organizations deploying systematic compression strategies report up to a **70% reduction in inference costs** and **10x improvement in deployment speed**, while advanced techniques can achieve **80–95% model size reductions** while retaining 95%+ of original accuracy. For a large-scale service serving millions of users, even small improvements in memory and throughput translate to enormous infrastructure savings.

Beyond the cloud, the rise of edge AI — from medical wearables to autonomous vehicles — makes compression not a preference but a hard requirement.

---

## 2. Quantization

### Concept

Quantization reduces the numerical precision used to represent model weights (and optionally activations). Deep learning models are typically trained in 32-bit floating point (FP32). Quantization converts these to smaller formats:

|Format|Bits|Memory vs FP32|
|---|---|---|
|FP16 / BF16|16|−50%|
|INT8|8|−75%|
|INT4 / FP4|4|−87.5%|
|1-bit (binary)|1|−96.9%|

The mathematical operation maps a continuous floating-point range to a discrete fixed-point set. A scaling factor and zero-point are used to minimize the rounding error introduced.

**Concrete example:** A LLaMA 70B model in FP16 requires ~140 GB of VRAM. With INT4 quantization, this drops to ~35–45 GB — fitting on a single high-end consumer GPU such as an RTX 4090.

### Types of Quantization

**Post-Training Quantization (PTQ)** — The model is already trained; quantization is applied afterward with a small calibration dataset. This is the lowest-effort approach and the recommended starting point for most engineers.

**Dynamic Quantization** — Weights are quantized offline, but activations are quantized on-the-fly during inference. The scaling factor is determined from values seen at runtime. Well-supported in PyTorch and effective for CPU deployment.

**Static Quantization** — Both weights and activations are quantized offline. Requires a representative calibration dataset to determine activation ranges. Faster than dynamic quantization at runtime.

**Quantization-Aware Training (QAT)** — Simulates quantization noise during the training loop itself using "fake quantize" operations. The model learns weights that are robust to low-precision arithmetic. Produces the best accuracy at aggressive bit widths but requires access to the training pipeline.

### Key Formats and Libraries (2025/2026)

- **bitsandbytes** — The most widely used quantization backend in the HuggingFace ecosystem. Supports INT8 and 4-bit (NF4/FP4). Works via a single flag: `load_in_4bit=True`.
- **GPTQ** — Uses second-order (Hessian) information for layer-wise quantization. Can compress GPT-scale models to 3–4 bits with minimal accuracy loss. Achieves 3.25–4.5× speed improvements on NVIDIA GPUs.
- **AWQ (Activation-aware Weight Quantization)** — Identifies the ~1% of salient weight channels using activation statistics and protects them during quantization. Typically maintains higher accuracy than GPTQ at the same bit width. Memory: ~3.5 GB for a 7B model vs 14 GB in FP16.
- **GGUF** — The de facto format for CPU inference via `llama.cpp`. Supports true on-disk file size reduction (unlike bitsandbytes which only affects GPU memory).
- **FP8** — A newer format supported by NVIDIA Hopper GPUs (H100). Meta's Llama 4 Maverick was quantized to FP8 using Red Hat's LLM Compressor.

### Python Example: PTQ with bitsandbytes (4-bit)

```python
# pip install transformers accelerate bitsandbytes -q

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# 4-bit NF4 quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # QLoRA-style nested quantization for extra savings
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
```

### Python Example: PyTorch Dynamic Quantization (CPU)

```python
import torch
import torch.quantization

# Load your trained model
model = MyTransformerModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Dynamic INT8 quantization — weights quantized offline, activations at runtime
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # target only Linear layers
    dtype=torch.qint8,
)

# Compare sizes
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"Original: {original_size / 1e6:.1f} MB  →  Quantized: {quantized_size / 1e6:.1f} MB")
```

---

## 3. Pruning

### Concept

Pruning removes parameters that contribute minimally to model output. The intuition: neural networks are highly over-parameterized. By zeroing out or deleting redundant weights, neurons, or even entire layers, we obtain a smaller, sparser model.

### Types of Pruning

**Unstructured (weight-level) pruning** — Individual weights are set to zero based on a criterion such as magnitude. Creates sparse weight matrices. Does not immediately reduce compute unless paired with sparse hardware kernels, but reduces memory.

**Structured pruning** — Removes entire neurons, attention heads, or layers. This produces dense (not sparse) smaller models that naturally benefit from standard matrix multiplication. More hardware-friendly. Examples:

- Head pruning in transformers (remove attention heads with low activation)
- Layer pruning (drop entire transformer blocks)
- Channel pruning in CNNs

**Magnitude-based pruning** — The simplest approach: remove weights with the smallest absolute values. The assumption is that low-magnitude weights contribute least to the output.

**Gradient/saliency-based pruning** — Uses gradient information (e.g., Optimal Brain Surgeon, oBERT) to estimate the importance of each weight. More accurate but more expensive to compute.

**Iterative vs one-shot pruning** — Iterative pruning gradually removes weights in rounds, fine-tuning between rounds. One-shot pruning does it in a single pass (faster, but typically lower quality).

**Concrete example:** A robotics company using a hybrid pruning + quantization pipeline reduced their model size by 75% and power consumption by 50% while maintaining 97% accuracy for smart warehouse robot navigation.

### Python Example: Unstructured Magnitude Pruning with PyTorch

```python
import torch
import torch.nn.utils.prune as prune

model = MyModel()

# Prune 20% of weights from all Linear layers (global unstructured)
parameters_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, torch.nn.Linear)
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.20,   # remove 20% of weights globally
)

# Check sparsity
total, zero = 0, 0
for module, _ in parameters_to_prune:
    total += module.weight.nelement()
    zero += (module.weight == 0).sum().item()

print(f"Global sparsity: {100.0 * zero / total:.1f}%")

# Make pruning permanent (remove mask, bake zeros in)
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)
```

### Python Example: Structured Head Pruning (Transformers)

```python
# Using the nn_pruning library or manual approach
# Below: zero out the 25% least-used attention heads by activation norm

from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("bert-base-uncased")

def prune_attention_heads(model, prune_fraction=0.25):
    for layer in model.encoder.layer:
        attn = layer.attention.self
        num_heads = attn.num_attention_heads
        head_size = attn.attention_head_size

        # Score heads by L2 norm of their query weight rows
        query_weight = attn.query.weight.view(num_heads, head_size, -1)
        head_scores = query_weight.norm(dim=(1, 2))

        n_to_prune = int(num_heads * prune_fraction)
        heads_to_prune = head_scores.argsort()[:n_to_prune].tolist()

        # HuggingFace built-in head pruning
        layer.attention.prune_heads(set(heads_to_prune))

    return model

pruned_model = prune_attention_heads(model, prune_fraction=0.25)
```

---

## 4. Knowledge Distillation

### Concept

Knowledge distillation takes a fundamentally different approach. Rather than modifying an existing model, it **transfers knowledge** from a large, high-performing **teacher model** to a smaller, more efficient **student model**.

The student is trained not only on ground-truth labels but also to mimic the teacher's output distribution — the "soft labels" (logits). Because soft labels encode the teacher's uncertainty and inter-class relationships, the student learns richer representations than it would from hard labels alone.

**Loss function:**

```
L_total = α * L_task(student_output, true_labels)
        + (1 - α) * L_distill(student_logits / T, teacher_logits / T)
```

Where `T` is the **temperature** parameter that softens the probability distribution, making low-probability classes more visible to the student. A temperature of 1 is standard; higher values (e.g., 4–8) expose more information from the teacher.

### Variants

**Response-based distillation** — Student mimics the teacher's final output logits. Simplest form; widely used for classification and language modeling.

**Feature-based distillation** — Student mimics intermediate layer activations (hidden states, attention maps). Captures richer structural knowledge at the cost of requiring matched architectures.

**Relation-based distillation** — Student learns relationships between data samples as encoded by the teacher (e.g., pairwise similarity matrices).

**Real-world example:** DistilBERT (distilled from BERT-base) retains 97% of BERT's language understanding performance with 40% fewer parameters and runs 60% faster.

### Python Example: Response-Based Distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Teacher: large pre-trained model
teacher = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
teacher.eval()  # Teacher is frozen

# Student: smaller model
student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)
ce_loss = nn.CrossEntropyLoss()

TEMPERATURE = 4.0
ALPHA = 0.5  # balance between task loss and distillation loss

def distillation_step(batch, labels):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Teacher inference (no gradients)
    with torch.no_grad():
        teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits

    # Student forward pass
    student_logits = student(input_ids, attention_mask=attention_mask).logits

    # Task loss (hard labels)
    task_loss = ce_loss(student_logits, labels)

    # Distillation loss (soft labels, scaled by T^2 as per Hinton et al.)
    soft_teacher = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (TEMPERATURE ** 2)

    total_loss = ALPHA * task_loss + (1 - ALPHA) * distill_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

---

## 5. Other Compression Techniques

### 5.1 Low-Rank Factorization / Decomposition

Large weight matrices can often be approximated by the product of two smaller matrices. For a weight matrix **W** of shape (m × n), if its effective rank is much smaller than min(m, n), we can write:

```
W ≈ A × B   where A is (m × r) and B is (r × n), with r << min(m, n)
```

This is the mathematical basis of **LoRA (Low-Rank Adaptation)** and the popular **QLoRA** variant (LoRA + quantization), which has become the dominant approach for fine-tuning LLMs on consumer hardware.

```python
# LoRA fine-tuning with PEFT
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

lora_config = LoraConfig(
    r=16,                         # rank — smaller = more compression
    lora_alpha=32,                # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# Trainable params: ~4M (0.13% of 3B total) — massive reduction in fine-tuning cost
```

SVD (Singular Value Decomposition) can be applied post-hoc to decompose weight matrices without retraining, though results are generally weaker than LoRA.

### 5.2 Speculative Decoding

Not a weight-compression technique, but an inference-time optimization that dramatically reduces token generation latency. A small, fast **draft model** proposes multiple tokens ahead; the large **target model** verifies them in parallel. Accepted tokens collapse what would have been sequential forward passes into single steps.

This is hardware-friendly (no retraining required), stacks cleanly with quantization, and can yield significant throughput improvements at long sequence lengths. NVIDIA's EAGLE-3 is a state-of-the-art draft model architecture for this purpose.

### 5.3 Neural Architecture Search (NAS)

Instead of compressing an existing model, NAS automates the search for an efficient architecture from the ground up. The search explores combinations of layer types, widths, depths, and skip connections to find Pareto-optimal trade-offs between accuracy and compute.

NAS has found applications in healthcare AI (lightweight medical imaging models), recommendation systems (low-latency delivery), and robotics. The main challenge is scalability: evaluating thousands of candidate architectures for a model with billions of parameters is extremely expensive.

### 5.4 KV Cache Compression

Specific to transformer-based LLMs: the key-value (KV) cache stores attention states for previous tokens and grows linearly with sequence length. For long-context models, this becomes a significant memory bottleneck.

Techniques include:

- **Selective KV eviction** — discard attention states for less important tokens (e.g., H2O, StreamingLLM)
- **KV quantization** — store cache entries at lower precision (INT8/FP8)
- **Cross-modal relevance compression** — for vision-language models, discard KV entries irrelevant to the current modality (e.g., AirCache)

### 5.5 Mixture-of-Experts (MoE) Sparse Activation

In MoE architectures, only a fraction of the model's parameters are activated for any given token. Models like Mixtral 8×7B have 47B total parameters but only activate ~13B per forward pass, giving performance comparable to a 70B dense model at much lower inference cost. This is architectural compression by design, not post-hoc.

---

## 6. Hybrid Pipelines

In practice, the largest efficiency gains come from **combining** techniques. The general recommended ordering, supported by recent literature, is:

```
Original Model
     │
     ▼
[1] Pruning ──────── Remove redundant parameters first.
     │                A pruned model quantizes more effectively
     │                due to reduced parameter redundancy.
     ▼
[2] Distillation ─── If shrinking to a student architecture,
     │                use the (pruned) teacher here.
     ▼
[3] Quantization ─── Apply last. Works on the already-efficient
                      model, maximizing precision-per-parameter value.
```

**Example: NVIDIA Minitron pipeline** NVIDIA prunes and distills their large foundation models first, then quantizes the result. This compound approach delivers the best accuracy-efficiency trade-off and is now a standard production workflow.

**Example: QLoRA** Combines 4-bit NF4 quantization with LoRA fine-tuning. The frozen base model runs in 4-bit; only the small LoRA adapter matrices are trained in BF16. This enables fine-tuning 70B+ models on a single consumer GPU.

```python
# QLoRA: fine-tune a quantized model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)  # enable gradient checkpointing etc.

lora_config = LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

---

## 7. What You Need to Know When Working with Compressed Models

### 7.1 Accuracy Degradation is Real — Measure It on Your Task

Generic benchmarks (perplexity, MMLU) may not reflect your specific use case. Always evaluate compressed models on **your** evaluation set. For complex reasoning tasks especially, INT4 quantization can subtly reduce reliability. Implement **canary deployments** to compare compressed model outputs against a full-precision FP16 baseline before promoting to production.

### 7.2 Format Fragmentation

GPTQ, AWQ, GGUF, and bitsandbytes each have different weight layouts, toolchains, and inference engine compatibility. Choosing the wrong format can cause inference engine incompatibility:

|Format|Best For|Inference Engine|
|---|---|---|
|bitsandbytes|HuggingFace inference, fine-tuning|Transformers / PEFT|
|GPTQ|GPU serving, vLLM|vLLM, TGI, AutoGPTQ|
|AWQ|High-accuracy GPU serving|vLLM, TGI|
|GGUF|CPU inference, local deployment|llama.cpp, Ollama|
|FP8|Datacenter H100 GPUs|TensorRT-LLM, vLLM|

### 7.3 Memory ≠ File Size

INT4 quantization reduces **GPU VRAM footprint**. Actual on-disk file size reduction depends on the storage format — only GGUF supports true file-size compression. A model stored in bitsandbytes INT4 may still have a large `.safetensors` file.

### 7.4 Hardware Matters

A compressed model's real-world speedup is hardware-dependent. GPTQ INT4 achieves 3–4.5× throughput improvement on NVIDIA GPUs; on older GPUs without INT4 tensor cores, gains may differ. **Always benchmark on your actual target hardware.** Theoretical compression ratios and wall-clock inference speed are different things.

### 7.5 Task-Specific Calibration Datasets

PTQ methods (especially GPTQ and AWQ) use a calibration dataset to determine quantization parameters. For domain-specific models, using a **representative calibration set from your domain** (not generic datasets like C4 or WikiText) materially improves post-quantization accuracy.

```python
# Using a custom calibration dataset with AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
)

model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)

# Provide domain-specific calibration examples
examples = [tokenizer(text, return_tensors="pt") for text in your_domain_texts[:128]]
model.quantize(examples)
model.save_quantized("./my-model-gptq-4bit")
```

### 7.6 Compression-Aware Fine-Tuning

For best results, avoid treating compression as a purely post-hoc step. If you control the training pipeline, use QAT or train with LoRA adapters from the start. Models fine-tuned with compression awareness learn more robust representations and tolerate lower bit widths with less accuracy loss.

### 7.7 Right-Size Your Model First

Before applying any compression, ask: _do I actually need this many parameters for my task?_ A 2.7B model like Phi-3 or Qwen2.5-3B fine-tuned on your domain may outperform a poorly compressed 70B model at a fraction of the cost. The most efficient compression is architectural right-sizing before quantization is ever applied.

### 7.8 Monitor for Behavioral Drift, Not Just Accuracy

Quantization noise can subtly alter a model's "personality" — sometimes reducing creativity, shifting calibration, or introducing systematic biases in edge cases. Beyond aggregate accuracy metrics, monitor:

- Output calibration (confidence vs. accuracy)
- Performance on rare/low-frequency classes or tokens
- Latency distributions (p50, p95, p99) — not just mean

---

## 8. Python Tooling Cheatsheet

|Task|Library|Install|
|---|---|---|
|4-bit / 8-bit quantization|`bitsandbytes`|`pip install bitsandbytes`|
|GPTQ quantization|`auto-gptq`|`pip install auto-gptq`|
|AWQ quantization|`autoawq`|`pip install autoawq`|
|LoRA / QLoRA fine-tuning|`peft`|`pip install peft`|
|Pruning (PyTorch native)|`torch.nn.utils.prune`|built-in|
|Distillation scaffolding|`transformers` + custom|`pip install transformers`|
|Compression experiments|`llmcompressor`|`pip install llmcompressor`|
|CPU inference (GGUF)|`llama-cpp-python`|`pip install llama-cpp-python`|
|GPU serving (GPTQ/AWQ)|`vllm`|`pip install vllm`|
|Neural architecture search|`optuna` + custom|`pip install optuna`|

---

## 9. Technique Comparison Summary

|Technique|Compression Ratio|Accuracy Impact|Requires Retraining|Best For|
|---|---|---|---|---|
|PTQ INT8|~4×|Low|No|Quick wins, general use|
|PTQ INT4 (GPTQ/AWQ)|~8×|Low–Medium|No (calibration only)|LLM deployment|
|QAT|~4–8×|Very low|Yes|Mobile / edge production|
|Unstructured pruning|Variable|Medium|Often yes|Research, sparse hardware|
|Structured pruning|2–4×|Medium|Yes (fine-tune)|Attention head reduction|
|Knowledge distillation|2–10×|Low (if well-tuned)|Yes (train student)|Building smaller model families|
|LoRA / QLoRA|Not for size (for fine-tuning efficiency)|N/A|Yes (the point is fine-tuning)|Efficient task adaptation|
|Low-rank factorization|2–5×|Medium|Sometimes|Weight matrix compression|
|Speculative decoding|No weight change|Minimal|No|Latency reduction|
|NAS|2–10×|Low|Yes (from scratch)|Custom efficient architectures|
|MoE sparse activation|~4–8× effective|Low|Yes (by design)|Foundation model architecture|

---

## 10. References & Further Reading

- A Survey of Model Compression Techniques: Past, Present, and Future — Frontiers in Robotics & AI, March 2025
- A Survey on Model Compression for Large Language Models — TACL / MIT Press
- A Review of State-of-the-Art Techniques for LLM Compression — Complex & Intelligent Systems, Springer, 2025
- Top 5 AI Model Optimization Techniques — NVIDIA Technical Blog
- LLM Compression and Optimization — Red Hat Blog
- Model Compression: Make Your ML Models Lighter and Faster — Towards Data Science
- GPTQ Paper (Frantar et al., 2023)
- AWQ Paper (Lin et al., 2023)
- QLoRA Paper (Dettmers et al., 2023)
- HuggingFace LLM Compressor
- Awesome LLM Compression (GitHub)

---

_Report generated April 2026. Techniques and library versions evolve rapidly — verify compatibility with your target framework versions before production deployment._