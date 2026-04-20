# Parallel inference strategies

## Parallel Inference Strategies in AI Engineering

Inference optimization is critical when deploying LLMs at scale. Parallel inference strategies aim to maximize throughput, minimize latency, and make efficient use of hardware. Here's a breakdown of the major approaches:

---

### 1. Data Parallelism

The simplest form: run **multiple independent inference requests simultaneously** across replicated model instances.

- Each GPU (or node) holds a full copy of the model
- Different requests are routed to different replicas
- A load balancer distributes traffic across replicas

**Best for:** High-throughput serving where each request fits in a single GPU's memory.

```python
# Conceptual: launching multiple model replicas with Ray Serve
from ray import serve
from transformers import pipeline

@serve.deployment(num_replicas=4)  # 4 parallel replicas
class LLMReplica:
    def __init__(self):
        self.model = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

    async def __call__(self, request):
        prompt = await request.json()
        return self.model(prompt["text"], max_new_tokens=200)
```

---

### 2. Tensor Parallelism

The model's weight **tensors are sharded across multiple GPUs**. Each GPU holds a slice of every layer, and they communicate during each forward pass.

- Splits large matrix multiplications across devices
- All-reduce operations synchronize partial results between GPUs at each layer
- Reduces the memory footprint per GPU, enabling larger models

**Best for:** Models too large to fit on a single GPU (e.g., 70B+ parameter models).

```
Layer N weights split across 4 GPUs:
GPU 0: W[:, 0:1024]
GPU 1: W[:, 1024:2048]
GPU 2: W[:, 2048:3072]
GPU 3: W[:, 3072:4096]
→ All-Reduce → combined output
```

Libraries like **vLLM**, **Megatron-LM**, and **TensorRT-LLM** implement this natively.

---

### 3. Pipeline Parallelism

The model's **layers are distributed across GPUs in sequence** — GPU 0 handles layers 1–8, GPU 1 handles layers 9–16, etc. Requests flow through the pipeline stage by stage.

**Micro-batching** is essential here: while GPU 1 processes batch _n_, GPU 0 can already start on batch _n+1_, keeping all stages busy.

```
GPU 0 (Layers 1–8):   [Batch A] [Batch B] [Batch C]
GPU 1 (Layers 9–16):       [Batch A] [Batch B] [Batch C]
GPU 2 (Layers 17–24):           [Batch A] [Batch B]
```

**Best for:** Very deep models where tensor parallelism communication overhead becomes a bottleneck.

---

### 4. Continuous Batching (Iteration-Level Scheduling)

Traditional static batching waits for a full batch before processing. **Continuous batching** (pioneered by Orca, used heavily in vLLM) processes requests at the **token iteration level** — new requests join mid-flight as others finish.

```
Step 1: [Req A (tok 1), Req B (tok 1), Req C (tok 1)]
Step 2: [Req A (tok 2), Req B (tok 2), Req C (tok 2)]
Step 3: [Req A (tok 3), Req D (tok 1) ← joined!, Req C (tok 3)]  # B finished
```

This dramatically improves GPU utilization since GPUs aren't idle waiting for slow requests to finish before accepting new ones.

---

### 5. Speculative Decoding

A **small draft model** generates _k_ candidate tokens in parallel, and the **large target model** verifies all of them in a single forward pass (which is parallelizable across token positions).

```
Draft model → [tok1, tok2, tok3, tok4, tok5]  (fast, cheap)
Target model → verifies all 5 in one pass     (parallel verification)
Result: accept first 3, reject 4 & 5, resample from position 4
```

This exploits the fact that the target model's verification pass is much cheaper than generating each token autoregressively. Net effect: **2–4x latency reduction** with no quality loss.

```python
# Hugging Face supports this natively
outputs = target_model.generate(
    inputs,
    assistant_model=draft_model,  # speculative decoding
    max_new_tokens=200,
)
```

---

### 6. Disaggregated Prefill / Decode (Chunked Prefill)

Prefill (processing the prompt) and decode (generating tokens) have very different compute profiles:

- **Prefill** is compute-bound and parallelizable across all input tokens
- **Decode** is memory-bandwidth-bound and sequential

Modern systems (e.g., **Distserve**, **Splitwise**) route these to **separate hardware** or handle them in **separate phases**, preventing long prompts from blocking decode throughput.

---

### 7. Sequence Parallelism

For extremely long contexts, the **sequence dimension itself** is sharded across GPUs. Each GPU handles a portion of the input tokens for attention computation, with communication to handle cross-sequence dependencies (e.g., via **Ring Attention**).

```
8K token sequence split across 4 GPUs:
GPU 0: tokens [0:2048]
GPU 1: tokens [2048:4096]
GPU 2: tokens [4096:6144]
GPU 3: tokens [6144:8192]
→ Ring all-reduce for cross-chunk attention
```

---

### Strategy Selection Guide

|Strategy|Memory Pressure|Latency Target|Throughput Target|Hardware|
|---|---|---|---|---|
|Data Parallelism|Low (model fits 1 GPU)|Moderate|✅ High|Multi-GPU / Multi-node|
|Tensor Parallelism|High (large model)|✅ Low|Moderate|NVLink GPUs|
|Pipeline Parallelism|High (deep model)|Moderate|✅ High|Any multi-GPU|
|Continuous Batching|Any|✅ Low|✅ High|Any|
|Speculative Decoding|Moderate|✅ Very Low|Moderate|Any|
|Disaggregated Prefill|Any|✅ Low|✅ High|Heterogeneous|
|Sequence Parallelism|Long context|Moderate|Moderate|NVLink GPUs|

In practice, production systems like **vLLM**, **TensorRT-LLM**, and **SGLang** **combine multiple strategies** — for example, tensor parallelism + continuous batching + chunked prefill — to squeeze maximum performance from the available hardware.