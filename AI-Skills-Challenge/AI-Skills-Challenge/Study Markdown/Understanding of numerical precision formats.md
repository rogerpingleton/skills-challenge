# Understanding of numerical precision formats

This is one of the most practically important topics in AI engineering — it directly affects model performance, memory usage, training stability, and deployment cost. Here's everything you need to know.

---

## Why precision formats matter

Neural networks are just enormous arrays of numbers. At any given moment you have weights, activations, gradients, and optimizer states all living in memory simultaneously. The format you use to represent each of those numbers determines:

- how much RAM/VRAM you need
- how fast matrix multiplications run
- whether training is stable or diverges
- what hardware accelerators you can use

The core tradeoff is always **range × precision vs. memory × speed**.

---

## The formats you'll encounter

Every floating point number is stored as three parts: a **sign bit**, an **exponent** (controls range), and a **mantissa/fraction**(controls precision).

Here's a visual map of how the major formats compare:

!numerical_precision_formats_overview.svg|697

---

## Deep dive: each format

### FP64 (double precision)

64 bits: 1 sign, 11 exponent, 52 mantissa. This is standard scientific computing precision, and you essentially never use it in deep learning. GPUs don't accelerate it well, and the extra precision doesn't matter for gradient-based optimization. The only time you'd see it is in certain scientific ML domains (physics simulations, high-precision numerical solvers).

### FP32 (single precision)

The historical baseline for deep learning. 32 bits: 1 sign, 8 exponent, 23 mantissa. In PyTorch, `torch.float32` is the default. When you do:

```python
import torch
t = torch.tensor([1.0, 2.0, 3.0])
print(t.dtype)  # torch.float32
```

This is what you're getting. It gives you ~7 decimal digits of precision and a range up to ~3.4 × 10³⁸, which is more than enough headroom for most operations. Training a large model in pure FP32 is stable but memory-hungry and slow on modern hardware.

### BF16 (Brain Float 16)

Developed at Google Brain. This is the key insight: **keep the exponent from FP32 (8 bits), but trim the mantissa from 23 bits down to 7**. The result is a 16-bit format with the same numerical range as FP32, but only about 2 decimal digits of precision.

Why does this work? Gradients in neural networks span many orders of magnitude, so range matters more than fine-grained precision. BF16 preserves that range exactly while halving memory.

```python
import torch
t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
# Each value uses 2 bytes instead of 4
# Can represent the same extreme values as FP32
```

BF16 is now the default training format on TPUs and A100/H100 GPUs. It's strongly preferred over FP16 for training because overflow (hitting the max value of 65504) is a persistent problem with FP16 that just doesn't exist with BF16.

### FP16 (half precision)

16 bits: 1 sign, 5 exponent, 10 mantissa. The narrower exponent range (max ~65504) is a real problem during training — activation values or gradients that exceed this overflow to `inf`, which immediately kills your training run.

The workaround is **loss scaling**: multiply the loss by a large scalar before backprop, which shifts the gradient magnitudes into the representable range, then divide the gradients back before the optimizer step. PyTorch's `torch.cuda.amp` does this automatically:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    with autocast():           # casts ops to FP16 automatically
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()   # scales loss before backward
    scaler.step(optimizer)          # unscales gradients, then steps
    scaler.update()                 # adjusts scale factor dynamically
```

FP16 is still widely used for inference (not training) and is the native format on older NVIDIA hardware (V100 and earlier).

### Mixed Precision Training

In practice, you almost never train entirely in one format. The standard recipe for large model training looks like this:

```
Model weights stored in:   FP32 (master copy) or BF16
Forward/backward pass:     BF16 or FP16
Optimizer states:          FP32 (Adam's momentum/variance need the precision)
Gradients accumulated in:  FP32
```

The reason optimizer states stay in FP32 is that small gradient updates — often on the order of 1e-7 — get rounded to zero in BF16, silently preventing learning. The 32-bit master copy ensures weight updates accumulate correctly.

With HuggingFace Transformers this is just:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    bf16=True,        # use BF16 for forward/backward
    # FP32 master weights and optimizer states handled automatically
)
```

### FP8

The frontier format, introduced with NVIDIA's H100. Two variants exist because the exponent/mantissa tradeoff pulls in opposite directions depending on where in the network you are:

**E4M3** (4 exponent, 3 mantissa) — tighter range, better precision. Used for **weights and activations** in the forward pass, where values are relatively well-behaved.

**E5M2** (5 exponent, 2 mantissa) — wider range, less precision. Used for **gradients** in the backward pass, where values can span a much wider range.

In Python with Transformer Engine (NVIDIA's library for H100):

```python
import transformer_engine.pytorch as te

# Automatically uses FP8 for supported ops on H100
model = te.Linear(1024, 1024)
with te.fp8_autocast(enabled=True):
    out = model(x)
```

FP8 roughly doubles throughput vs BF16 on H100s, which is why it's increasingly common for large-scale training runs.

### INT8

Integer quantization is conceptually different from floating point. Instead of a sign/exponent/mantissa, you just have a signed integer from -128 to 127. There's no concept of representing different scales — so you need to **calibrate** a scaling factor that maps your float range into this integer range.

The key formula is: `x_quantized = round(x / scale)`, where `scale = max(|x|) / 127`.

```python
import torch

def quantize_to_int8(tensor):
    scale = tensor.abs().max() / 127.0
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale

def dequantize_from_int8(quantized, scale):
    return quantized.to(torch.float32) * scale

weights = torch.randn(256, 256)
q, scale = quantize_to_int8(weights)
reconstructed = dequantize_from_int8(q, scale)
print(f"Max error: {(weights - reconstructed).abs().max():.4f}")
```

INT8 is primarily used for **inference**, not training. The precision loss is acceptable for forward passes (you're just multiplying fixed weights by inputs), but gradients need more precision than INT8 can offer.

`bitsandbytes` and HuggingFace's `load_in_8bit` make this practical:

```python
from transformers import AutoModelForCausalLM

# Load a 7B model that would normally need ~28GB in FP32
# INT8 quantization reduces this to ~7GB
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### INT4 (and beyond)

4-bit quantization is the current practical limit for weight-only quantization in inference. GPTQ and GGUF (used by llama.cpp) use 4-bit formats. At 4 bits you're representing only 16 distinct values per weight — precise calibration and groupwise scaling (applying different scale factors to small groups of weights, e.g., 128 at a time) is essential to preserve accuracy.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4: better distribution for weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16, store in 4-bit
    bnb_4bit_use_double_quant=True  # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=config)
```

---

## Decision map: which format to use when

Here's the practical decision logic for AI engineering work:

**Training a large model from scratch:** BF16 mixed precision with FP32 optimizer states. Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` or the HuggingFace `bf16=True` flag. Don't use FP16 unless you're on hardware that doesn't support BF16 (older V100s).

**Fine-tuning a pre-trained model:** Same as above. QLoRA fine-tuning pairs 4-bit quantized frozen weights with BF16 trainable adapters — the most memory-efficient recipe currently available.

**Serving/inference on a GPU server:** INT8 or FP8 for maximum throughput. If quality is critical, BF16. INT8 is the sweet spot for most production deployments.

**Running locally (CPU or limited VRAM):** GGUF Q4_K_M or Q5_K_M quantization via llama.cpp. These use 4-5 bit formats with careful calibration and run surprisingly well on consumer hardware.

**Debugging a training instability:** Switch everything to FP32 first to rule out numerical issues. If stability returns, the culprit is precision. Then add formats back one at a time.

---

## Common failure modes to know

**FP16 overflow:** Your loss suddenly becomes `nan`. This is almost always FP16 activations or gradients hitting the 65504 ceiling. Fix: switch to BF16, or add loss scaling. To check:

```python
# Quick diagnostic
x = torch.tensor(70000.0, dtype=torch.float16)
print(x)  # tensor(inf) — you've overflowed
```

**Gradient underflow:** The opposite — gradients are so small they round to zero. Most dangerous in FP16 for very deep networks. The GradScaler handles this automatically.

**Quantization outliers:** Some transformer weight matrices have a small number of extreme outliers that collapse the INT8 scale, making most values map to 0 or 1. LLM.int8() (bitsandbytes) solves this by keeping outlier dimensions in FP16 and quantizing the rest.

**Accumulation errors:** When summing many small FP16 values, errors compound. PyTorch's autocast keeps reduction operations (layer norm, softmax) in FP32 by default for this reason.

---

## Memory footprint quick reference

|Format|Bytes/parameter|7B model|70B model|
|---|---|---|---|
|FP32|4|28 GB|280 GB|
|BF16/FP16|2|14 GB|140 GB|
|INT8|1|7 GB|70 GB|
|INT4|0.5|3.5 GB|35 GB|

This is just weights. During training you also need gradients (same size as weights), optimizer states (2× weights for Adam), and activations — so multiply the training requirement roughly by 4–6× vs. inference-only.

The single most impactful thing you can do for model accessibility is moving from FP32 → BF16 (2× compression) and then BF16 → INT4 (another 4× compression). A 7B model that requires a $10,000 GPU in FP32 fits on a consumer RTX 4090 in INT4.