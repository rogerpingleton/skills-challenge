# Hardware (GPU, TPU, memory specs)

## 1. The Fundamental Nature of Inference: Memory-Bound, Not Compute-Bound

The single most important mental model for an AI engineer optimizing inference is this:

> **LLM inference is not compute-bound — it is memory-bound.**

GPUs have gotten dramatically faster at raw computation (FLOPS) every generation. But memory bandwidth and capacity have not kept pace. During the decode (token generation) phase, the GPU spends most of its time _loading data from memory_, not performing matrix multiplications. Each token generation requires:

- Loading all model weights (or relevant layers) from HBM
- Loading accumulated KV cache entries for the full context
- Performing a relatively small matrix-vector multiplication

The result: the hardware compute units sit idle most of the time, waiting on memory. This is why your H100 at 99% "GPU utilization" in `nvidia-smi` can still be significantly underperforming — utilization metrics measure compute scheduling, not efficiency.

**Arithmetic Intensity** is the key metric here: the ratio of FLOPs performed to bytes moved from memory. Decode-phase inference has very low arithmetic intensity (~1 FLOP/byte), which is why memory bandwidth dominates. This has profound implications for every optimization decision you make.

---

## 2. The Two Phases of LLM Inference

Understanding hardware bottlenecks requires separating the two distinct phases:

### Prefill Phase

- Processes the entire input prompt **in parallel** (like training — batch matrix multiplications)
- **Compute-bound**: GPU cores are heavily utilized
- Latency metric: **TTFT** (Time to First Token)
- The prompt tokens' KV values are computed here and cached for the decode phase

### Decode Phase (Autoregressive Generation)

- Generates output **one token at a time** — no parallelism across tokens due to causal dependency
- **Memory-bound**: Each step loads all model weights + growing KV cache from HBM
- Latency metric: **TBT** (Time Between Tokens) → **TPOT** (Time Per Output Token)
- This is where most hardware optimization effort pays off

The distinction matters for hardware selection: a chip that is fantastic at the prefill phase (high FLOP/s) may not be optimal for decode (high memory bandwidth). This is also the rationale behind **prefill-decode disaggregation** — splitting these phases across different hardware.

---

## 3. GPU Architecture & Inference

### NVIDIA GPU Hierarchy (2025–2026)

|GPU|HBM Capacity|Memory Bandwidth|Key Precision|Notes|
|---|---|---|---|---|
|A100 (80GB)|80 GB HBM2e|~2 TB/s|FP16/BF16/INT8|Workhorse, still common|
|H100 (SXM5)|80 GB HBM3|~3.35 TB/s|FP8 + Transformer Engine|Enterprise standard|
|H200|141 GB HBM3e|~4.8 TB/s|FP8|1.4x H100 for memory-bound tasks|
|B200 (Blackwell)|192 GB HBM3e|~8 TB/s|FP4, FP6, FP8|Up to ~30x inference speedup vs prior gen|
|GB200 NVL72|13.5 TB (72 GPUs)|~130 TB/s collective|FP4|Rack-scale inference unit|

### What Matters on a GPU for Inference

**Memory Bandwidth (most important for decode)** The raw GB/s available to move weights and KV cache entries in/out of compute cores. This directly throttles token generation throughput. The B200's ~8 TB/s HBM3e nearly doubles the H100.

**HBM Capacity (determines what fits)** How much model weight + KV cache you can hold on-device. A 70B model in FP16 requires ~140 GB just for weights — meaning you need multiple GPUs or quantization. Memory capacity dictates your model size, batch size, and maximum context length before you spill to CPU or disk.

**Tensor Cores and Supported Precisions** NVIDIA's Tensor Cores are specialized matrix multiply units that dramatically outperform CUDA cores for AI workloads. Each generation adds lower-precision formats:

- **FP8** (H100/H200): ~2x throughput vs FP16 with minimal quality loss
- **FP4/FP6** (Blackwell): Further 2x gains, critical for inference economics
- Lower precision = more weights fit in cache, less bandwidth consumed = faster inference

**NVLink & NVSwitch (inter-GPU bandwidth)** When splitting models across GPUs (tensor parallelism), all-reduce communication over NVLink is 7–10x faster than PCIe. The GB200 NVL72's NVLink 5 provides ~130 TB/s of collective bandwidth across 72 GPUs — this effectively creates a unified 13.5 TB memory pool.

### The CUDA Ecosystem Moat

NVIDIA's 15-year lead with CUDA means nearly every library, framework, and optimization tool works natively on NVIDIA GPUs. PyTorch, HuggingFace Transformers, vLLM, TensorRT-LLM — all are CUDA-first. This ecosystem advantage is distinct from raw hardware performance and is a real factor in infrastructure decisions.

---

## 4. TPU Architecture & Why It Matters Now

### The Systolic Array Advantage

TPUs use **systolic arrays** — a grid of multiply-accumulate units where data flows rhythmically through processing elements without being repeatedly fetched from memory. This contrasts with GPU CUDA cores that require individual memory fetches per operation. For inference:

- **Deterministic data flow** → lower latency variance
- **No instruction decode overhead** → GPUs burn cycles fetching and decoding instructions; TPUs execute fixed operations in silicon
- **60–65% better energy efficiency** vs comparable NVIDIA GPUs for inference workloads

### TPU Generation Snapshot (2025)

|Generation|Notes|
|---|---|
|TPU v5p|Enterprise training + inference; widely deployed|
|TPU v6 (Trillium)|4.7x perf/chip vs v5; ~60-65% better efficiency vs H100|
|TPU v7 (Ironwood)|4,614 TFLOPS (BF16) vs 459 for v5p; inference-only; 100% better perf/watt vs v6|

### The Real-World Cost Picture

The economics have shifted dramatically. One validated case: a Series C computer vision startup switched 128 H100s to TPU v6e pods and their monthly inference bill dropped from $340K to $89K. Google Cloud's AI revenue is reportedly growing 2.1x faster than Azure ML, which remains heavily NVIDIA-dependent.

### The Ecosystem Friction

TPUs require **JAX** or **TensorFlow/XLA** — the industry runs on PyTorch. This is the primary adoption barrier. However:

- PyTorch TPU support has improved significantly
- For inference, CUDA dependency is much weaker than for training
- The JAX ecosystem (Flax, Optax) is maturing rapidly
- Learning JAX + TPU optimization now is genuinely career-differentiating

### When to Consider TPUs

- High-volume, stable inference workloads (not research/experimentation)
- Models using TensorFlow or JAX natively
- Workloads where cost-per-query is the primary SLO
- Google Cloud environments already

---

## 5. Other Hardware Accelerators

### AWS Inferentia2 / Trainium2

- Up to **70% lower cost** vs GPU-based alternatives for inference
- Inferentia2: 4x higher throughput, 10x lower latency vs Inferentia1
- Tightly integrated with AWS SageMaker and Bedrock
- Best for: stable, high-volume AWS-native inference at scale

### AMD MI300X / MI350X / MI355X

- **192 GB HCM** (combined HBM + CDNA architecture) on MI300X — largest memory footprint in the GPU market
- Critical for large model inference without multi-GPU parallelism
- ROCm + HIP ecosystem is improving; vLLM has solid MI300X support
- OpenAI committed to a 6-gigawatt AMD MI450 deployment in 2026 — a strong signal
- Best for: inference of very large models (70B+) where model fits on single chip

### Groq LPU (Language Processing Unit)

- Architecture designed for deterministic, ultra-low latency inference
- 10x lower latency than H100 benchmarks for single-request scenarios
- Limited batch flexibility; less suited for high-concurrency serving
- Best for: latency-critical applications (real-time voice, interactive agents)

### Intel Gaudi3

- Positioned as cost-effective alternative to H100
- Competitive on longer-output LLM inference in some benchmarks
- Best for: cost-sensitive enterprise deployments not requiring maximum throughput

### Edge/NPU Hardware

- **Apple M-series Neural Engine**: Exceptional performance-per-watt for on-device inference
- **Qualcomm Snapdragon AI**: Mobile/edge deployment
- **NVIDIA Jetson Orin**: Edge inference with CUDA compatibility
- Relevant for: on-device models, privacy-sensitive deployments, offline inference

---

## 6. Memory: The True Bottleneck

### Memory Hierarchy and Latency Gradient

```
Register File      (on-core)    ~1 cycle,    ~50 TB/s
L1 Cache           (on-SM)      ~5 cycles,   ~20 TB/s
L2 Cache           (on-chip)    ~20 cycles,  ~5 TB/s
HBM (VRAM)         (on-package) ~200 cycles, ~3-8 TB/s
CPU DRAM           (off-chip)   ~1000 cycles, ~100-400 GB/s
NVMe SSD           (off-device) ~100k cycles, ~5-12 GB/s
```

**The engineering challenge**: LLM weights and KV caches live in HBM. Each decode step moves massive amounts of data up this hierarchy, hits HBM, and returns. Every optimization technique in inference is essentially an attempt to reduce the amount of data that must traverse the HBM bottleneck, or to do more useful work per byte moved.

### What Lives in GPU Memory During Inference

For a **70B parameter model**, rough memory breakdown:

- **Model weights in FP16**: ~140 GB
- **Model weights in FP8**: ~70 GB
- **KV cache (batch=32, 8K context, FP16)**: ~40–50 GB (often **exceeds model weights**)
- **Activation buffers**: ~1–5 GB

The KV cache growing to exceed model size is not an edge case — it's the norm at production batch sizes and long contexts. This is why memory capacity and bandwidth are your primary hardware constraints.

---

## 7. KV Cache: The Central Challenge

### What It Is

During attention computation, each token produces **Key (K)** and **Value (V)** tensors that all future tokens must attend to. Without caching, generating token N requires recomputing K and V for all N-1 previous tokens — O(N²) cost. KV caching stores these tensors so each new token only needs one forward pass.

**Memory formula:**

```
KV cache size = 2 × num_layers × num_heads × head_dim × precision_bytes × batch_size × seq_length
```

For Llama-3-70B at FP16, batch=1, 128K context: **~128 GB** — exceeding a single H100's VRAM entirely.

### The Key Problems

**Fragmentation**: Traditional static pre-allocation wastes 60–80% of KV cache memory because sequences finish at different lengths but hold reserved blocks until completion.

**Linear scaling with context**: As context windows grow from 8K → 128K → 1M tokens, the KV cache size grows linearly and becomes the dominant memory consumer.

**Bandwidth pressure**: Each decode step must load the entire KV cache from HBM — at long sequences, this dominates decode latency.

### KV Cache Optimization Techniques

**PagedAttention (vLLM's contribution)** Manages KV cache like OS virtual memory — allocates small fixed-size pages on demand, recycles them when sequences finish. Reduces waste from 60–80% down to under 4%. Enables 2–4x throughput improvements on the same hardware. Now essentially the standard in production inference engines.

```python
# vLLM uses PagedAttention automatically
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    gpu_memory_utilization=0.90,  # % of VRAM for KV cache pool
    max_model_len=32768,
)
```

**KV Cache Quantization** Compress stored K/V tensors to lower precision:

- **FP8 KV cache**: Halves cache memory with <1% accuracy loss; native H100/H200 hardware support
- **INT4/INT8 KV cache**: 2–4x compression; requires calibration

```python
# vLLM with FP8 KV cache
llm = LLM(
    model="...",
    kv_cache_dtype="fp8",  # Native H100 support
)
```

**Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)** Architectural choices that reduce KV cache size at model design time:

- **MQA**: All attention heads share a single K/V head → drastic reduction in cache size
- **GQA**: Groups of heads share K/V → balance between MQA efficiency and full MHA quality
- Llama-3, Mistral, Gemma all use GQA — a major reason they're inference-efficient

**KV Cache Eviction / Sparse Attention** Drop low-importance tokens from the cache based on attention scores:

- **StreamingLLM**: Retains "attention sink" tokens (first ~4) + sliding window of recent tokens → theoretically infinite context at fixed memory cost
- **Entropy-guided caching**: Allocate more budget to layers with broader attention patterns

**Multi-tier KV Cache (Offloading)** When GPU memory is insufficient, tier the cache:

- **GPU HBM** → **CPU DRAM** → **NVMe SSD**
- LMCache enables up to 15x higher throughput vs baseline by caching and reusing KV across requests
- CPU-GPU transfer adds ~10–50ms latency per retrieval — acceptable for some workloads
- Critical for RAG, multi-turn chat, and long-document tasks

```python
# LMCache integration with vLLM
from lmcache.integration.vllm import LMCacheConfig

lmcache_config = LMCacheConfig(
    chunk_size=256,
    local_cpu="cpu",    # CPU memory tier
    max_local_cpu_size=50,  # GB
)
```

---

## 8. Parallelism Strategies

When a model exceeds single-GPU memory, or when you need to reduce per-request latency through parallelism, you have three primary strategies:

### Tensor Parallelism (TP)

Split individual weight matrices **horizontally** across GPUs. Each GPU holds a shard of every layer.

- **Pro**: Reduces per-request latency (all GPUs work on every token)
- **Con**: Requires high-bandwidth NVLink interconnect; all-reduce communication every layer
- **Use when**: Minimizing TTFT/latency is the priority; have NVLink-connected GPUs
- `vllm serve --tensor-parallel-size 4`

### Pipeline Parallelism (PP)

Split the model **vertically** by layer groups — GPU 1 handles layers 1–16, GPU 2 handles 17–32, etc.

- **Pro**: Lower inter-GPU communication overhead; scales to more GPUs
- **Con**: Pipeline bubbles reduce efficiency; higher per-request latency
- **Use when**: Maximizing throughput for large batch workloads; GPUs connected via PCIe
- `vllm serve --pipeline-parallel-size 2`

### Data Parallelism / Replica Scaling

Run **full model replicas** on separate GPU sets, routing different requests to different replicas.

- **Pro**: Linear throughput scaling; no cross-GPU synchronization
- **Con**: Must fit model on each replica's GPU(s); doesn't reduce per-request latency
- **Use when**: Model fits on a single GPU; scaling throughput for high QPS

### Expert Parallelism (for MoE models)

Route different expert networks (in Mixture-of-Experts architectures like Mixtral, DeepSeek) to different GPUs.

- DeepSeek-V3 and similar MoE models are particularly amenable to this
- Relevant for trillion-parameter sparse models

### Prefill-Decode Disaggregation

Split the **prefill phase** and **decode phase** across different hardware:

- Prefill nodes: High compute density (good for long prompts, batch prefill)
- Decode nodes: High memory bandwidth (good for continuous single-token generation)
- Emerging pattern in 2025–2026 for large-scale serving infrastructure

---

## 9. Key Optimization Techniques

### Quantization

Reducing the numerical precision of weights and/or activations is the single highest-ROI optimization for most deployments.

|Method|Precision|Memory Savings|Accuracy Impact|Notes|
|---|---|---|---|---|
|FP16/BF16|16-bit|Baseline|None|Default serving precision|
|FP8|8-bit|~2x|<0.5%|Native H100/B200 hardware; use first|
|GPTQ|INT4|~4x|1–3%|Post-training; good for smaller models|
|AWQ|INT4|~4x|<1%|Activation-aware; better than GPTQ|
|GGUF/llama.cpp|INT4/INT8|2–4x|1–2%|CPU + consumer GPU deployment|
|NVFP4|4-bit|~4x|<1%|Blackwell-native; newest format|

**Practical guidance**: For production GPU serving, start with FP8 weights + FP8 KV cache (essentially free on H100+). If memory is still tight, move to AWQ INT4 for weights.

```python
# Quantization with vLLM (FP8 - recommended starting point)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    quantization="fp8",  # or "awq", "gptq"
    kv_cache_dtype="fp8",
)
```

### Speculative Decoding

Uses a **small draft model** to speculatively generate N candidate tokens, which the large target model then **verifies in one parallel forward pass**. Since verification is highly parallelizable (unlike generation), correct speculations yield multiple tokens for the cost of one.

- Draft model: typically 1–7B parameters
- Acceptance rate: 70–90% on domain-specific tasks
- Speedup: **2–3x** for latency-sensitive (low-concurrency) scenarios
- Caveat: At high batch sizes, the verify step becomes expensive; speedup diminishes

```python
# Speculative decoding in vLLM
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
)
```

### FlashAttention / FlashAttention-3

Rewrites the attention kernel to avoid materializing the full attention matrix in HBM. Instead, it tiles the computation so attention scores are computed entirely in on-chip SRAM.

- Reduces attention memory from O(N²) to O(N) HBM usage
- 2–4x faster attention on H100 GPUs
- **FlashAttention-3** adds FP8 support and async pipeline stages for Hopper architecture
- **FlashMLA** (from DeepSeek) extends this to Multi-head Latent Attention architectures
- This is essentially mandatory in any serious inference stack — it's now the default in vLLM, TensorRT-LLM, SGLang

### Continuous Batching (In-flight Batching)

Traditional static batching waits for all requests in a batch to complete before starting new ones. This is catastrophically inefficient when requests have variable output lengths.

Continuous batching evicts completed sequences mid-batch and immediately fills their slots with new requests — GPU utilization stays high even with heterogeneous workloads.

- vLLM, TensorRT-LLM, and SGLang all implement this
- Critical for production serving; don't run static batching in production

### Prefix Caching

When many requests share a common prefix (a system prompt, a document, a RAG context), the prefill computation for that prefix can be cached and reused. This is extremely high-value for:

- Chatbots with large system prompts
- RAG applications with shared context chunks
- Batch evaluation with shared few-shot examples

SGLang's RadixAttention implements prefix caching with smart LRU eviction and is particularly effective for shared-prefix workloads.

### Kernel Fusion & Graph Compilation

Modern inference engines fuse multiple operations (e.g., LayerNorm + attention projection + activation) into single CUDA/Triton kernels, reducing kernel launch overhead and intermediate memory traffic.

- **torch.compile** with `mode="reduce-overhead"` or `"max-autotune"` automates much of this
- TensorRT-LLM applies aggressive layer fusion during its compilation phase (~28 min cold start)
- Triton kernels (used extensively in vLLM/SGLang) are hand-tuned for specific hardware

---

## 10. Inference Serving Frameworks

### Framework Comparison (2025–2026)

|Framework|Best For|Throughput|Cold Start|Key Features|
|---|---|---|---|---|
|**vLLM**|Fastest to production; model flexibility|High|~60s|PagedAttention, continuous batching, broad model support|
|**TensorRT-LLM**|Peak throughput, single-model prod|Highest|~28 min|Kernel fusion, FP4/FP8/INT8, Blackwell-optimized|
|**SGLang**|RAG, chatbots, shared-prefix workloads|High|~60s|RadixAttention prefix caching, structured output|
|**llama.cpp / Ollama**|Local/edge, CPU+GPU hybrid|Moderate|Fast|GGUF, runs on CPU, consumer hardware|
|**HuggingFace TGI**|HF ecosystem integration|Moderate|Fast|Broad model compatibility, easy deployment|

### Choosing a Framework

```
Is this a research/dev environment or local testing?
  → llama.cpp / Ollama / HF TGI

Is time-to-production more important than peak throughput?
  → vLLM (gold standard for most teams)

Is this a high-QPS chatbot or RAG application with shared prefixes?
  → SGLang

Is this a stable, single-model production deployment on NVIDIA hardware 
  where you can invest 30 minutes per startup?
  → TensorRT-LLM (13% faster than vLLM at high concurrency)
```

### Key vLLM Configuration Knobs

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    
    # Hardware utilization
    gpu_memory_utilization=0.90,   # Leave 10% headroom
    tensor_parallel_size=4,         # Across 4 GPUs
    
    # Precision
    dtype="bfloat16",               # or "float16", "float8"
    quantization="fp8",             # Weight quantization
    kv_cache_dtype="fp8",           # KV cache quantization
    
    # Context
    max_model_len=32768,            # Max sequence length
    
    # Speculative decoding (optional)
    speculative_model="...",
    num_speculative_tokens=5,
    
    # Prefix caching
    enable_prefix_caching=True,     # Enable for RAG/chatbot
)
```

---

## 11. Hardware Selection Decision Guide

### Decision Tree

```
Start with: What is your primary SLO?

LATENCY (minimize TTFT + TPOT for single requests)
  ├── Budget available?
  │     ├── Yes → NVIDIA H200 or B200; tensor parallelism across NVLink GPUs
  │     │         Consider Groq LPU for extreme single-request latency
  │     └── No  → Quantize to FP8/AWQ; speculative decoding
  └── On-device/edge? → Apple M-series / Jetson Orin / Qualcomm

THROUGHPUT (maximize tokens/sec across many concurrent requests)
  ├── Stable, high-volume workload?
  │     ├── Google Cloud → TPU v6/v7 (60-65% better efficiency)
  │     ├── AWS        → Inferentia2 (up to 70% cost reduction)
  │     └── Multi-cloud/on-prem → NVIDIA H100/H200 + TensorRT-LLM
  └── Variable/experimental workload?
        → NVIDIA GPUs + vLLM (ecosystem flexibility)

COST (minimize $/1M tokens)
  ├── Locked into cloud provider → Use their custom silicon (TPU/Inferentia)
  ├── Need CUDA ecosystem → H100 + aggressive quantization (FP8/AWQ)
  └── Large model (70B+) needing large VRAM → AMD MI300X (192GB HBM)
```

### Memory Sizing Rules of Thumb

```python
# Approximate VRAM required for inference (FP16 weights)
# Model weights only — add KV cache budget on top

model_sizes = {
    "7B":   "~14 GB  → 1x A100-40GB (tight), 1x RTX 4090",
    "13B":  "~26 GB  → 1x A100-80GB",
    "34B":  "~68 GB  → 1x H100-80GB (with quantization), or 2x A100",
    "70B":  "~140 GB → 2x H100-80GB (FP16) or 1x MI300X (192GB HBM)",
    "405B": "~810 GB → 8x H100 (FP8 = ~405GB, need 6x H100 minimum)",
}

# KV cache additional memory per request (FP16, 32-layer 70B model)
# kv_per_token_bytes = 2 * 80_layers * 8_heads * 128_dim * 2_bytes ≈ 330KB/token
# For 4K context, 1 request: ~1.3 GB
# For 128K context, 1 request: ~42 GB (!!!)
```

---

## 12. Metrics That Matter

Understanding which metrics to instrument and optimize against is critical for hardware-aware inference work.

|Metric|Definition|Affected By|Optimization Lever|
|---|---|---|---|
|**TTFT**|Time to first token|Prefill compute, batch size|Tensor parallelism, prefix caching|
|**TPOT**|Time per output token|Memory bandwidth, KV cache size|Quantization, GQA, smaller models|
|**Throughput**|Tokens/sec across all requests|Batch size, GPU utilization|Continuous batching, quantization|
|**MFU**|Model FLOP Utilization|Arithmetic intensity, memory bounds|All of the above|
|**Memory Utilization**|% HBM used|Model size + KV cache|PagedAttention, quantization|
|**Cache Hit Rate**|% requests served from KV cache|Prefix reuse, routing|Prefix caching, sticky routing|
|**P95/P99 Latency**|Tail latency under load|Head-of-line blocking, scheduling|Continuous batching, request routing|

### Profiling Tools

```bash
# NVIDIA profiler
nsys profile --trace=cuda,nvtx python serve.py
ncu --set full python inference.py

# vLLM built-in benchmarking
python -m vllm.entrypoints.benchmark_throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 1000

# PyTorch profiler
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = model.generate(...)
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
```

---

## 13. The Evolving Landscape (2025–2026)

### Structural Shifts Underway

**The training-to-inference shift**: Training is a one-time capital expense. Inference runs forever at scale. Over 90% of total LLM operational cost is inference. Hardware vendors have responded — Google's Ironwood is inference-only, NVIDIA's Blackwell FP4 is inference-focused, AWS doubled down on Inferentia. The economics of inference now drive chip architecture decisions.

**MoE (Mixture of Experts) models** are becoming dominant. Models like DeepSeek-V3 (671B sparse, ~37B active) and Mixtral deliver large-model quality at much lower inference compute cost because only a fraction of weights are activated per token. Hardware and serving frameworks are increasingly optimizing for sparse activation patterns.

**Context windows expanding**: 128K is common today; million-token contexts are achievable. This makes KV cache management even more critical — the KV cache problem grows faster than any other bottleneck as context windows extend.

**Disaggregated inference**: Separating prefill and decode across specialized hardware is moving from research into production. llm-d (from Red Hat/IBM) and similar frameworks implement this at scale.

**Custom silicon everywhere**: AWS, Google, Microsoft (Maia), Meta (MTIA), Apple — every hyperscaler is designing inference chips to reduce NVIDIA dependency and optimize for their specific workloads. As an AI engineer, expect to encounter non-NVIDIA hardware increasingly in production environments.

### What to Learn Now

1. **JAX + TPU optimization** — 340% YoY growth in job postings for TPU engineers; early expertise commands premium compensation
2. **FP4/FP8 quantization workflows** — These are table stakes for production Blackwell deployments
3. **vLLM internals** — PagedAttention, continuous batching, prefix caching; the framework is used everywhere
4. **FlashAttention-3 / FlashMLA** — Understanding IO-aware kernel design pays dividends in inference optimization
5. **Prefill-decode disaggregation** — Emerging architectural pattern for large-scale serving

---

## Quick Reference: Optimization Impact

|Technique|Latency Reduction|Throughput Gain|Memory Savings|Difficulty|
|---|:-:|:-:|:-:|:-:|
|FP8 quantization (weights)|~20–30%|~2x|~50%|Low|
|AWQ/GPTQ INT4|~30–40%|~2–3x|~75%|Low-Med|
|PagedAttention|—|2–4x|~60–80% waste → <4%|Low (vLLM default)|
|Continuous batching|—|3–10x|—|Low (vLLM default)|
|FlashAttention-3|~30–50% on attn|2–4x on attn|~O(N²) → O(N)|Low (auto)|
|Speculative decoding|2–3x|Neutral/mild|—|Medium|
|GQA (model arch)|~20–40%|—|~4–8x KV cache|N/A (model choice)|
|KV cache FP8|~10–20%|~20–40%|~50%|Low|
|Tensor parallelism (4 GPU)|~3–4x|~3–4x|Distributes load|Medium|
|Prefix caching|Up to 10x for hits|3–10x|—|Low-Med|

**Stack them**: FP8 + FlashAttention + continuous batching + PagedAttention on an H100 delivers **5–8x better cost-efficiency** than naive FP16 inference with static batching — more impact than upgrading from H100 to H200.

---

_Last updated: April 2026. Hardware landscape evolves rapidly; verify specific benchmark numbers against current sources._