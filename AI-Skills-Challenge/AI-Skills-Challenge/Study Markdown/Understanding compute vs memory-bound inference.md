# Understanding compute vs memory-bound inference

## 1. Executive Summary

Inference optimization for Large Language Models (LLMs) is not a single problem — it is two fundamentally different problems that happen to share a model. The **prefill phase** (processing the input prompt) is typically **compute-bound**: raw GPU throughput is the limiting factor. The **decode phase** (token-by-token generation) is typically **memory-bound**: memory bandwidth dominates, and GPU compute cores sit idle waiting for data.

Misdiagnosing the bottleneck leads to wasted effort. Applying compute optimizations to a memory-bound workload yields minimal gains, and vice versa. This report explains how to correctly identify which regime you are in, the criteria for selecting an optimization approach, and concrete Python implementations for each.

---

## 2. Foundational Concepts

### 2.1 Arithmetic Intensity

**Arithmetic Intensity (AI)** is the central diagnostic metric. It measures how much computation is performed per byte of memory traffic:

```
Arithmetic Intensity = FLOPs / Bytes Accessed
```

- A **high AI** means many operations per memory byte — the operation is efficient and likely **compute-bound**.
- A **low AI** means few operations per memory byte — the operation stalls waiting for data — **memory-bound**.

### 2.2 The Hardware Ridge Point

Every GPU has a theoretical **ridge point** — the arithmetic intensity at which the workload transitions from memory-bound to compute-bound:

```
Ridge Point (ops/byte) = Peak FLOPS / Peak Memory Bandwidth
```

**Example: NVIDIA A100-80GB**

- Peak FP16 FLOPS: 312 TFLOPS
- Peak Memory Bandwidth: 2 TB/s
- Ridge Point: **~156 ops/byte**

If your workload's arithmetic intensity is below 156 ops/byte on this GPU, you are memory-bound. Above it, you are compute-bound.

### 2.3 Memory Hierarchy in Modern GPUs

Understanding where data lives is critical:

|Level|Size (A100)|Bandwidth|Latency|
|---|---|---|---|
|Registers|~256 KB/SM|Extremely high|~1 cycle|
|L1/Shared|192 KB/SM|~19 TB/s|~20 cycles|
|L2 Cache|40 MB|~12 TB/s|~200 cycles|
|HBM (VRAM)|80 GB|2 TB/s|~1000 cycles|

For LLM inference, model weights (tens to hundreds of GB) live in HBM. Every forward pass requires loading those weights from HBM to compute units — this is the source of the memory wall.

---

## 3. The Two Inference Phases: Prefill and Decode

LLM inference has two structurally different phases with entirely different bottlenecks.

### 3.1 Prefill Phase — Compute-Bound

The prefill phase processes the entire input prompt in a single parallel forward pass (akin to training).

**Why it's compute-bound:**

- All input tokens are processed simultaneously as large matrix-matrix multiplications (GEMM operations).
- High arithmetic intensity: extensive data reuse across the full sequence.
- GPU Tensor Cores are heavily utilized.

**Key metric:** Time to First Token (TTFT)

**Typical Arithmetic Intensity:** In LLaMA-2-7B at sequence length 2048, projection layers (q_proj, k_proj, v_proj, etc.) exhibit ~1024 ops/byte — well above most GPUs' ridge points, making them firmly compute-bound.

### 3.2 Decode Phase — Memory-Bound

After prefill, the model generates output tokens one at a time in an autoregressive loop. Each token requires a full forward pass through every transformer layer.

**Why it's memory-bound:**

- Each step is a matrix-_vector_ operation (not matrix-matrix), severely underutilizing GPU parallelism.
- The **KV Cache** (key-value tensors for every prior token, every layer) must be read from HBM on each step.
- Arithmetic intensity drops to single digits or tens of ops/byte.

**Key metric:** Inter-Token Latency (ITL)

**KV Cache Memory Footprint:**

```
KV Cache = num_layers × 2 × num_kv_heads × head_dim × seq_len × dtype_bytes
```

For Llama-3-70B (80 layers, 8 KV heads, head_dim=128) at BF16 with 4096 context:

```
= 80 × 2 × 8 × 128 × 4096 × 2 ≈ 1.3 GB per request
```

This scales linearly with context length — long-context workloads saturate VRAM bandwidth before compute becomes a bottleneck.

### 3.3 Phase Summary

|Property|Prefill|Decode|
|---|---|---|
|Bottleneck|Compute (FLOPS)|Memory Bandwidth|
|Operation type|Matrix × Matrix|Matrix × Vector|
|Parallelism|Full (across tokens)|Sequential (per token)|
|GPU utilization|High|Low (waiting for data)|
|Primary metric|TTFT|ITL (tokens/sec)|
|KV cache role|Written once|Read every step|

---

## 4. The Roofline Model: Diagnosing Your Bottleneck

The Roofline Model is the standard visual tool for identifying whether a workload is compute-bound or memory-bound.

### 4.1 How to Read a Roofline Plot

The model plots achievable performance (FLOPS/sec) against arithmetic intensity (FLOPS/byte):

- **Diagonal line** (sloping up): the memory bandwidth ceiling. Points here are memory-bound — performance scales linearly with arithmetic intensity.
- **Horizontal line** (flat): the compute ceiling. Points here are compute-bound — no more performance is achievable regardless of data layout.
- **Ridge point**: where the two lines meet — the optimal operating point.

### 4.2 Programmatically Computing Arithmetic Intensity

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def estimate_arithmetic_intensity(
    model_name: str,
    batch_size: int,
    seq_len: int,
    mode: str = "decode",  # "prefill" or "decode"
) -> dict:
    """
    Estimate arithmetic intensity for LLM inference phases.
    Returns ops, bytes accessed, and intensity for each linear layer.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    hidden = config.hidden_size
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    layers = config.num_hidden_layers
    intermediate = getattr(config, "intermediate_size", hidden * 4)
    dtype_bytes = 2  # BF16/FP16

    results = {}

    # For decode mode: seq_len per step = 1 (generating one token at a time)
    # For prefill: seq_len = prompt length
    effective_seq = 1 if mode == "decode" else seq_len

    # --- Attention projections (Q, K, V, O) ---
    # Q: [batch, seq, hidden] × [hidden, hidden] → FLOPs = 2 * batch * seq * hidden^2
    q_ops = 2 * batch_size * effective_seq * hidden * hidden
    # Bytes: weight matrix + input + output
    q_bytes = (hidden * hidden + batch_size * effective_seq * hidden * 2) * dtype_bytes
    results["q_proj"] = {"ops": q_ops, "bytes": q_bytes, "intensity": q_ops / q_bytes}

    # Same for K, V (may differ with GQA)
    kv_dim = (kv_heads * hidden) // heads
    kv_ops = 2 * batch_size * effective_seq * hidden * kv_dim
    kv_bytes = (hidden * kv_dim + batch_size * effective_seq * (hidden + kv_dim) * 2) * dtype_bytes
    results["k_proj"] = {"ops": kv_ops, "bytes": kv_bytes, "intensity": kv_ops / kv_bytes}
    results["v_proj"] = {"ops": kv_ops, "bytes": kv_bytes, "intensity": kv_ops / kv_bytes}

    # --- KV Cache read (decode only) ---
    if mode == "decode":
        kv_cache_bytes_per_layer = 2 * batch_size * seq_len * kv_dim * dtype_bytes
        kv_cache_total = layers * kv_cache_bytes_per_layer
        # Attention score computation (QK^T) per decode step
        attn_ops = 2 * batch_size * heads * 1 * seq_len  # q_len=1
        results["kv_cache_read"] = {
            "ops": attn_ops * layers,
            "bytes": kv_cache_total,
            "intensity": (attn_ops * layers) / kv_cache_total,
        }

    # --- FFN layers ---
    ffn_ops = 2 * batch_size * effective_seq * hidden * intermediate
    ffn_bytes = (hidden * intermediate + batch_size * effective_seq * (hidden + intermediate)) * dtype_bytes
    results["ffn_gate"] = {"ops": ffn_ops, "bytes": ffn_bytes, "intensity": ffn_ops / ffn_bytes}

    # Ridge point for common GPUs
    gpu_specs = {
        "A100_40GB": {"tflops": 312e12, "bandwidth_bytes_s": 1.6e12},
        "A100_80GB": {"tflops": 312e12, "bandwidth_bytes_s": 2.0e12},
        "H100_SXM5": {"tflops": 989e12, "bandwidth_bytes_s": 3.35e12},
        "A10G":      {"tflops": 125e12, "bandwidth_bytes_s": 600e9},
    }

    print(f"\n=== Arithmetic Intensity Analysis: {model_name} ({mode} mode) ===")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}\n")
    print(f"{'Layer':<20} {'OPs':>14} {'Bytes':>14} {'AI (ops/byte)':>16}")
    print("-" * 68)
    for name, v in results.items():
        bound = "compute-bound" if v["intensity"] > 150 else "memory-bound"
        print(f"{name:<20} {v['ops']:>14.2e} {v['bytes']:>14.2e} {v['intensity']:>14.1f}  [{bound}]")

    print("\n=== GPU Ridge Points ===")
    for gpu, specs in gpu_specs.items():
        ridge = specs["tflops"] / specs["bandwidth_bytes_s"]
        print(f"  {gpu:<15}: {ridge:.0f} ops/byte")

    return results


# Example usage:
# results = estimate_arithmetic_intensity("meta-llama/Llama-2-7b-hf", batch_size=1, seq_len=512, mode="decode")
# results = estimate_arithmetic_intensity("meta-llama/Llama-2-7b-hf", batch_size=32, seq_len=2048, mode="prefill")
```

---

## 5. Decision Criteria: Choosing the Right Optimization Strategy

Before applying any optimization, you need to answer four questions:

### 5.1 What Is Your Primary Inference Phase?

|If your workload is...|Primary bottleneck|
|---|---|
|Processing long prompts (RAG, summarization)|Compute (prefill)|
|Generating long outputs (code gen, creative)|Memory (decode)|
|Short prompt + short answer (chat)|Memory (decode)|
|Large batch offline processing|Compute or both|

### 5.2 What Is Your Latency vs. Throughput Goal?

|Priority|Implication|
|---|---|
|Minimize TTFT|Optimize prefill compute parallelism|
|Minimize ITL (streaming feel)|Optimize decode memory bandwidth|
|Maximize throughput (batch)|Increase arithmetic intensity via batching|
|Minimize cost|Quantization + batching|

### 5.3 What Is Your Batch Size?

Batch size directly shifts the regime:

- **Batch size = 1**: Almost always memory-bound, even for large models. GPU cores idle 90%+ of the time.
- **Batch size = 8–32**: Transitional. Matrix-vector operations become matrix-matrix.
- **Batch size > 64**: Shifts toward compute-bound for prefill; decode remains memory-bound due to KV cache growth.

The rule of thumb: a batch size of ~`ridge_point / (2 * hidden_size / num_params_per_byte)` is needed to escape memory-bound decode.

### 5.4 Decision Flowchart

```
Is your workload latency-sensitive (real-time serving)?
├─ YES → Are you bottlenecked on TTFT or ITL?
│         ├─ TTFT (first token slow) → Compute-bound prefill optimization
│         │    → Tensor parallelism, Flash Attention, model compilation
│         └─ ITL (streaming feels slow) → Memory-bound decode optimization
│              → Quantization, KV cache compression, speculative decoding
└─ NO (batch/offline) → Maximize throughput
      → Continuous batching, larger batch sizes, quantization
      → Profile with MBU/MFU metrics to find headroom
```

---

## 6. Optimizing Memory-Bound Inference

The decode phase is almost universally memory-bound. These techniques directly address memory bandwidth pressure.

### 6.1 Quantization

Quantization reduces weight precision, shrinking the bytes that must be loaded per forward pass. This directly increases arithmetic intensity.

**INT4 Weight-Only Quantization with bitsandbytes:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_quantized_model(model_name: str, bits: int = 4):
    """
    Load an LLM with weight-only quantization to reduce memory footprint
    and improve memory-bound decode throughput.
    
    bits=4 → ~4x smaller weights → ~4x reduction in HBM bandwidth needed
    bits=8 → ~2x smaller weights → ~2x reduction in HBM bandwidth needed
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16, store in INT4
        bnb_4bit_use_double_quant=True,           # nested quantization (saves ~0.4 bits/param)
        bnb_4bit_quant_type="nf4",                # NormalFloat4 for better accuracy
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Report memory savings
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model memory (quantized): {param_bytes / 1e9:.2f} GB")

    return model, tokenizer


# GGUF quantization via llama.cpp (for CPU/edge inference):
# pip install llama-cpp-python
def load_gguf_model(gguf_path: str, n_gpu_layers: int = -1):
    """
    GGUF format enables fine-grained quantization (Q4_K_M, Q5_K_S, Q8_0, etc.)
    and offloading to CPU when VRAM is insufficient.
    """
    from llama_cpp import Llama

    model = Llama(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers,  # -1 = all layers on GPU
        n_ctx=4096,
        n_threads=8,
        verbose=False,
    )
    return model
```

**Effect on arithmetic intensity (from LLaMA-2-7B decode analysis):**

|Precision|Bytes/weight|Approx. AI (ops/byte)|Regime|
|---|---|---|---|
|FP16|2 bytes|~8|Memory-bound|
|INT8|1 byte|~16|Memory-bound (improved)|
|INT4|0.5 bytes|~32|Memory-bound (significantly improved)|

### 6.2 KV Cache Optimization

The KV cache is often larger than the model itself for long-context workloads. These techniques target it directly.

```python
from vllm import LLM, SamplingParams

def setup_vllm_with_paged_attention(
    model_name: str,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
):
    """
    vLLM's PagedAttention manages KV cache memory in non-contiguous pages,
    eliminating memory fragmentation and enabling:
    - Higher concurrent request throughput
    - Prefix caching (reuse KV for shared prompt prefixes)
    - Efficient beam search without KV duplication
    """
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,    # Cache KV for common prefixes (e.g., system prompts)
        kv_cache_dtype="fp8",          # Quantize KV cache itself (FP8 = 2x less bandwidth)
    )
    return llm


def benchmark_kv_pressure(model_name: str, prompts: list[str]):
    """
    Profile KV cache usage under load.
    Prometheus metrics from vLLM:
      vllm:gpu_cache_usage_perc  → % KV cache occupied
      vllm:num_requests_waiting  → backpressure indicator
    """
    import time
    llm = setup_vllm_with_paged_attention(model_name)
    params = SamplingParams(max_tokens=256, temperature=0.0)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Throughput: {total_tokens / elapsed:.1f} tokens/sec")
    print(f"Latency: {elapsed / len(prompts) * 1000:.1f} ms/request")
```

### 6.3 Speculative Decoding

Speculative decoding converts memory-bound decode steps into compute-bound verification passes. A small draft model proposes multiple tokens; the main model verifies them all in a single (compute-efficient) forward pass.

```python
from vllm import LLM, SamplingParams

def setup_speculative_decoding(
    target_model: str,
    draft_model: str,
    num_speculative_tokens: int = 5,
):
    """
    Speculative decoding reduces decode latency by 2–3x on memory-bound workloads.
    
    How it works:
      1. Draft model generates K candidate tokens fast (cheap memory reads)
      2. Target model verifies all K tokens in ONE parallel forward pass
      3. Accepted tokens advance the sequence; rejected tokens are discarded
      4. Net effect: ~K tokens generated per target model call instead of 1
    
    This increases arithmetic intensity of the target model from matrix-vector
    to near matrix-matrix operations — shifting the regime toward compute-bound.
    
    Best draft models: 10–100x smaller than the target, same vocabulary.
    E.g., target=Llama-3-70B, draft=Llama-3-8B
    """
    llm = LLM(
        model=target_model,
        speculative_config={
            "model": draft_model,
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    return llm


def ngram_speculative_decoding(model_name: str):
    """
    N-gram speculation: no draft model needed.
    Reuses repeated phrases in the prompt as speculative tokens.
    Effective for: RAG (reusing retrieved passages), code completion, templates.
    """
    llm = LLM(
        model=model_name,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_num_tokens": 4,  # n-gram window size
        },
    )
    return llm
```

### 6.4 Continuous (In-Flight) Batching

Static batching ties GPU resources to the longest sequence in a batch. Continuous batching allows new requests to join mid-generation, dramatically improving GPU utilization.

```python
# With vLLM's AsyncLLMEngine (production serving):
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import AsyncEngineArgs, SamplingParams
import asyncio

async def serve_with_continuous_batching(model_name: str):
    """
    Continuous batching (in-flight batching) keeps GPUs saturated by:
    1. Filling the batch with new requests as others complete
    2. Evicting finished sequences immediately (no padding waste)
    3. Allowing variable-length sequences in the same batch
    
    Result: 10–20x throughput improvement over static batching at scale.
    """
    engine_args = AsyncEngineArgs(
        model=model_name,
        max_num_seqs=256,              # max concurrent sequences in flight
        max_num_batched_tokens=8192,   # max tokens across all sequences per step
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(prompt: str, request_id: str):
        params = SamplingParams(max_tokens=512, temperature=0.7)
        async for output in engine.generate(prompt, params, request_id=request_id):
            if output.finished:
                return output.outputs[0].text
        return ""

    # Simulate concurrent requests
    tasks = [
        generate(f"Explain concept {i} in detail.", f"req_{i}")
        for i in range(100)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 7. Optimizing Compute-Bound Inference

The prefill phase is compute-bound, especially for long prompts and large batches.

### 7.1 FlashAttention

The standard attention implementation reads/writes the full attention matrix (O(n²) in sequence length) to HBM. FlashAttention fuses the computation into a single kernel that stays in SRAM, reducing HBM traffic by up to 10x.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_with_flash_attention(model_name: str):
    """
    Flash Attention 2 reduces HBM bandwidth for attention computation from O(n²) to O(n).
    Critical for long-context prefill (compute-bound).
    
    Speed improvement:
    - 2–4x faster attention forward pass
    - Scales to longer contexts without OOM
    - Reduced memory footprint enables larger batch sizes
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # requires flash-attn package
        device_map="auto",
    )
    return model


# Manual Flash Attention with torch.nn.functional (PyTorch 2.0+):
def scaled_dot_product_attention_optimized(q, k, v, mask=None):
    """
    torch.nn.functional.scaled_dot_product_attention automatically uses
    FlashAttention, memory-efficient attention, or math attention
    depending on available hardware and input shape.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=True,   # enables causal masking without materializing the mask
    )
```

### 7.2 Tensor Parallelism

For compute-bound prefill with very long sequences or large models, tensor parallelism shards weight matrices across multiple GPUs, reducing per-GPU compute and memory pressure simultaneously.

```python
# With vLLM tensor parallelism:
from vllm import LLM

def load_tensor_parallel(model_name: str, tensor_parallel_size: int = 4):
    """
    Tensor parallelism splits weight matrices (e.g., [hidden, 4*hidden])
    column-wise across GPUs. Each GPU handles 1/N of the matrix multiply.
    
    Trade-off:
    - Reduces per-GPU memory and compute load (good for large models)
    - Adds all-reduce communication overhead (inter-GPU bandwidth cost)
    - Best for: models that don't fit on one GPU, or long-context prefill
    
    Rule: use TP=2 or TP=4 for models > 13B; TP=8 for 70B+
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
    )
    return llm
```

### 7.3 Model Compilation (torch.compile)

Python and eager-mode overhead become significant for compute-bound workloads. `torch.compile` traces the model, fuses operations, and generates optimized CUDA kernels.

```python
import torch
from transformers import AutoModelForCausalLM

def compile_for_inference(model_name: str):
    """
    torch.compile reduces compute-bound prefill latency by:
    1. Operator fusion: merging elementwise ops into a single kernel
    2. Memory layout optimization: choosing contiguous layouts for GEMM
    3. Kernel selection: picking the fastest CUBLAS/CUTLASS kernel variant
    
    Best for: repeated inference with fixed shapes (static compilation)
    Use 'reduce-overhead' for dynamic shapes (most LLM cases)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Compile the forward pass
    model.forward = torch.compile(
        model.forward,
        mode="reduce-overhead",      # good for LLM inference (dynamic shapes)
        fullgraph=False,             # allow graph breaks for unsupported ops
    )

    return model


# For maximum compute throughput with fixed batch size:
def compile_static_batch(model, batch_size: int, seq_len: int):
    """
    Static shape compilation generates the most aggressive optimizations.
    Trade the flexibility of dynamic shapes for maximum kernel efficiency.
    """
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True)
    return compiled
```

### 7.4 Disaggregated Prefill-Decode

Modern production systems run prefill and decode on separate hardware pools, allowing each to be independently scaled and optimized.

```python
# Conceptual architecture (implemented in systems like Splitwise, DistServe):

class DisaggregatedInferenceConfig:
    """
    Disaggregated inference separates prefill and decode into distinct server pools.
    
    Prefill servers:
    - Optimized for compute throughput (high FLOPS GPUs: H100, A100)
    - Large batch sizes of prompts
    - Can use fewer, more powerful GPUs
    
    Decode servers:
    - Optimized for memory bandwidth (high HBM BW: H100 SXM > PCIe)
    - Many concurrent sequences
    - KV cache transferred from prefill server after prompt processing
    
    Benefits:
    - Eliminates prefill-decode interference (prefill stalls decode in co-located setups)
    - Independent scaling: more decode capacity for high-QPS, more prefill for long-context
    - Better hardware utilization: each pool uses hardware suited to its bottleneck
    
    Implementation: vLLM supports this via disaggregated prefill in v0.6+
    """
    prefill_gpus: list = None          # H100 SXM5 cluster
    decode_gpus: list = None           # H100 PCIe or A100 cluster
    kv_transfer_bandwidth: float = 0   # NVLink / RDMA bandwidth in GB/s
```

---

## 8. Python Implementation Examples

### 8.1 Full Diagnostic Pipeline

```python
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class InferenceBenchmark:
    """
    Comprehensive inference profiler that measures TTFT, ITL,
    and estimates whether workload is compute or memory bound.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.eval()

    def measure_ttft(self, prompt: str) -> float:
        """Time to First Token: measures prefill latency (compute-bound)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Only one forward pass (no generation loop)
            _ = self.model(**inputs)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0

        return ttft * 1000  # ms

    def measure_itl(self, prompt: str, num_tokens: int = 50) -> tuple[float, list]:
        """
        Inter-Token Latency: measures decode phase (memory-bound).
        Returns mean ITL and per-token latencies.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_times = []

        with torch.no_grad():
            past_key_values = None
            input_ids = inputs["input_ids"]

            # Prefill
            torch.cuda.synchronize()
            outputs = self.model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Decode loop — each iteration is one token
            for _ in range(num_tokens):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                token_times.append(time.perf_counter() - t0)

                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        itl_ms = [t * 1000 for t in token_times]
        return np.mean(itl_ms), itl_ms

    def compute_mbu(self, model_params_bytes: int, itl_seconds: float) -> float:
        """
        Model Bandwidth Utilization (MBU) = fraction of peak memory bandwidth used.
        
        MBU = (model_params_bytes / itl_seconds) / peak_bandwidth
        
        MBU close to 1.0 → memory-saturated, minimal headroom
        MBU << 1.0 → memory underutilized (overhead, batching opportunity)
        """
        # NVIDIA A100 80GB peak bandwidth
        peak_bandwidth = 2.0e12  # bytes/sec
        achieved_bandwidth = model_params_bytes / itl_seconds
        return achieved_bandwidth / peak_bandwidth

    def run_full_diagnostic(self, prompt: str):
        param_count = sum(p.numel() for p in self.model.parameters())
        param_bytes = param_count * 2  # BF16

        ttft = self.measure_ttft(prompt)
        mean_itl, itl_list = self.measure_itl(prompt)
        mbu = self.compute_mbu(param_bytes, mean_itl / 1000)

        prompt_tokens = len(self.tokenizer.encode(prompt))

        print("=" * 55)
        print("  INFERENCE DIAGNOSTIC REPORT")
        print("=" * 55)
        print(f"  Prompt tokens:     {prompt_tokens}")
        print(f"  Model parameters:  {param_count / 1e9:.1f}B")
        print(f"  TTFT:              {ttft:.1f} ms  (prefill)")
        print(f"  Mean ITL:          {mean_itl:.2f} ms/token  (decode)")
        print(f"  Throughput:        {1000 / mean_itl:.1f} tokens/sec")
        print(f"  MBU:               {mbu:.2%}")
        print("-" * 55)
        if mbu > 0.7:
            print("  DIAGNOSIS: Memory-bound decode")
            print("  RECOMMEND: Quantization, speculative decoding,")
            print("             KV cache compression, larger batches")
        elif ttft > mean_itl * prompt_tokens * 0.5:
            print("  DIAGNOSIS: Compute-bound prefill")
            print("  RECOMMEND: Flash Attention, tensor parallelism,")
            print("             prefill chunking, model compilation")
        else:
            print("  DIAGNOSIS: Mixed / overhead-bound")
            print("  RECOMMEND: Profile with nsys/ncu for kernel-level")
            print("             analysis")
        print("=" * 55)
```

### 8.2 Quick Optimization Selector

```python
def select_optimization_strategy(
    model_size_b: float,           # model size in billions of parameters
    batch_size: int,
    avg_prompt_tokens: int,
    avg_output_tokens: int,
    latency_sla_ms: float | None,  # None = throughput-optimized
    gpu_vram_gb: float,
) -> dict:
    """
    Rule-based strategy selector based on workload characteristics.
    
    Returns a dict of recommended optimizations and their priority.
    """
    recommendations = {}

    model_fp16_gb = model_size_b * 2  # rough FP16 estimate
    kv_cache_pressure = avg_output_tokens > 512 or avg_prompt_tokens > 2048

    # 1. Memory headroom
    if model_fp16_gb > gpu_vram_gb * 0.7:
        recommendations["quantization"] = {
            "priority": "CRITICAL",
            "method": "INT4 (bitsandbytes NF4 or GGUF Q4_K_M)",
            "rationale": "Model doesn't fit comfortably in VRAM at FP16",
        }

    # 2. Decode optimization (almost always needed)
    if avg_output_tokens > 20:
        recommendations["speculative_decoding"] = {
            "priority": "HIGH" if latency_sla_ms else "MEDIUM",
            "method": "Draft model speculation or n-gram (vLLM)",
            "rationale": f"Generating {avg_output_tokens} tokens — decode is memory-bound",
        }

    # 3. KV cache
    if kv_cache_pressure:
        recommendations["kv_cache_optimization"] = {
            "priority": "HIGH",
            "method": "PagedAttention (vLLM) + FP8 KV cache + prefix caching",
            "rationale": "Long sequences create heavy KV cache bandwidth pressure",
        }

    # 4. Batching
    if batch_size > 1 or latency_sla_ms is None:
        recommendations["continuous_batching"] = {
            "priority": "HIGH",
            "method": "vLLM continuous batching or TensorRT-LLM in-flight batching",
            "rationale": "Batching improves GPU utilization for memory-bound decode",
        }

    # 5. Prefill optimization
    if avg_prompt_tokens > 1000:
        recommendations["flash_attention"] = {
            "priority": "HIGH",
            "method": "FlashAttention-2 via transformers attn_implementation",
            "rationale": f"Long prompts ({avg_prompt_tokens} tokens) — prefill is compute-bound",
        }

    # 6. Multi-GPU
    if model_fp16_gb > gpu_vram_gb:
        recommendations["tensor_parallelism"] = {
            "priority": "CRITICAL",
            "method": f"TP={int(model_fp16_gb / gpu_vram_gb) + 1} (vLLM tensor_parallel_size)",
            "rationale": "Model exceeds single GPU VRAM",
        }

    # 7. Compilation
    if batch_size > 8 and avg_prompt_tokens > 500:
        recommendations["compilation"] = {
            "priority": "MEDIUM",
            "method": "torch.compile mode='reduce-overhead' or TensorRT-LLM",
            "rationale": "Large compute-bound batches benefit from kernel fusion",
        }

    return recommendations


# Example: Llama-3-70B, real-time chatbot
strategy = select_optimization_strategy(
    model_size_b=70,
    batch_size=1,
    avg_prompt_tokens=512,
    avg_output_tokens=256,
    latency_sla_ms=200,
    gpu_vram_gb=80,  # A100 80GB
)
for name, rec in strategy.items():
    print(f"[{rec['priority']}] {name}: {rec['method']}")
    print(f"         → {rec['rationale']}\n")
```

---

## 9. Metrics and Monitoring

### 9.1 Key Metrics

|Metric|Formula|What it measures|
|---|---|---|
|**TTFT**|Time from request to first output token|Prefill (compute-bound) efficiency|
|**ITL**|Time between consecutive output tokens|Decode (memory-bound) efficiency|
|**Throughput**|Output tokens / second|Overall system efficiency|
|**MBU**|Achieved BW / Peak BW|Memory utilization (decode)|
|**MFU**|Achieved FLOPS / Peak FLOPS|Compute utilization (prefill)|

### 9.2 Profiling with PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_inference(model, inputs, num_warmup=3, num_active=5):
    """
    Use PyTorch profiler to identify compute vs memory bound kernels.
    Look for:
    - High 'Self CUDA Time' on memory ops → memory-bound
    - High 'Self CUDA Time' on matrix multiply kernels → compute-bound
    """
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**inputs)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        with_modules=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_active),
    ) as prof:
        with torch.no_grad():
            for _ in range(num_active + 2):
                with record_function("model_inference"):
                    _ = model(**inputs)
                prof.step()

    # Print top CUDA ops sorted by time
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    # Export for Tensorboard or Chrome trace
    prof.export_chrome_trace("inference_trace.json")
    return prof
```

### 9.3 vLLM Prometheus Metrics

```python
# Key vLLM metrics for production monitoring
VLLM_METRICS = {
    # Memory pressure (decode bottleneck)
    "vllm:gpu_cache_usage_perc": "% KV cache occupied — >85% triggers eviction",
    "vllm:num_requests_waiting": "Backpressure indicator — high = decode bottleneck",
    
    # Throughput
    "vllm:avg_generation_throughput_toks_per_s": "Decode tokens/sec — track vs MBU",
    "vllm:avg_prompt_throughput_toks_per_s": "Prefill tokens/sec — track vs MFU",
    
    # Latency components
    "vllm:time_to_first_token_seconds": "TTFT histogram — compute-bound signal",
    "vllm:time_per_output_token_seconds": "ITL histogram — memory-bound signal",
    
    # Batch efficiency
    "vllm:num_running_seqs": "Concurrent sequences in batch",
}
```

---

## 10. Summary Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│              INFERENCE OPTIMIZATION DECISION MAP            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Profile first: measure TTFT vs ITL, compute MBU/MFU       │
│                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │    MEMORY-BOUND          │  │    COMPUTE-BOUND         │  │
│  │  (decode phase)          │  │  (prefill phase)         │  │
│  │                          │  │                          │  │
│  │  Signals:                │  │  Signals:                │  │
│  │  • High ITL              │  │  • High TTFT             │  │
│  │  • MBU close to 1.0      │  │  • MFU close to 1.0      │  │
│  │  • Batch size = 1–4      │  │  • Large batch or long   │  │
│  │  • Long output gen       │  │    prompt workloads      │  │
│  │                          │  │                          │  │
│  │  Solutions:              │  │  Solutions:              │  │
│  │  1. Quantization (INT4)  │  │  1. FlashAttention-2     │  │
│  │  2. Speculative decoding │  │  2. Tensor parallelism   │  │
│  │  3. KV cache compression │  │  3. torch.compile        │  │
│  │  4. Continuous batching  │  │  4. Prefill chunking     │  │
│  │  5. PagedAttention       │  │  5. Disaggregated prefill│  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                             │
│  Both phases co-exist in every inference request.           │
│  Optimize for the dominant bottleneck of your workload.     │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Impact Summary

|Technique|Target|Typical Speedup|Tradeoff|
|---|---|---|---|
|INT4 Quantization|Memory-bound decode|1.5–3x ITL|Slight accuracy loss|
|Speculative Decoding|Memory-bound decode|2–3x ITL|Requires draft model|
|Continuous Batching|Memory-bound, throughput|10–20x throughput|None significant|
|KV Cache Quantization (FP8)|Memory-bound|1.3–1.8x|Minor accuracy loss|
|FlashAttention-2|Compute-bound prefill|2–4x TTFT|Requires CUDA ≥ 8.0|
|Tensor Parallelism|Compute-bound, large models|Near-linear with GPUs|Communication overhead|
|torch.compile|Compute-bound|1.2–2x|Compilation overhead|
|Disaggregated Prefill|Both|Phase-specific|Infrastructure complexity|

---

## 11. References

- Yuan, Z. et al. (2024). _LLM Inference Unveiled: Survey and Roofline Model Insights_. arXiv:2402.16363
- Wang, H. et al. (2025). _A Systematic Characterization of LLM Inference on GPUs_. arXiv:2512.01644
- NVIDIA Developer Blog. _Mastering LLM Techniques: Inference Optimization_. (December 2025)
- Databricks Engineering. _LLM Inference Performance Engineering: Best Practices_.
- Baseten. _A Guide to LLM Inference and Performance_. (November 2023)
- Clarifai. _LLM Inference Optimization Techniques_. (January 2026)
- Williams, S., Waterman, A., Patterson, D. (2009). _Roofline: An Architectural Model for Multicore Architectures_. ACM/IEEE SC '09.
- Pope, R. et al. (2023). _Efficiently Scaling Transformer Inference_. MLSys 2023.

---

_Report generated: April 2026 | Covers model families through Llama-3, Qwen-2.5, and Mistral generations_