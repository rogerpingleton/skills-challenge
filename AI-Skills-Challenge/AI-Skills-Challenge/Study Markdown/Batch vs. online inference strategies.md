# Batch vs. online inference strategies

## 1. Overview

Inference is the act of running a trained model against new inputs to produce predictions or generated outputs. For most production AI systems — especially those powered by large language models (LLMs) — **inference is the dominant cost driver**, not training. IBM Research (2025) estimated that over 70% of enterprise AI costs now come from inference workloads.

As an AI Engineer, your core challenge in inference optimization is navigating three competing objectives simultaneously:

- **Latency** — how quickly a single request gets a response
- **Throughput** — how many requests the system can handle per second
- **Cost** — the hardware, memory, and operational expense per token or prediction

The choice between batch and online inference strategies is the highest-leverage architectural decision you will make. Everything else — quantization, caching, parallelism — layers on top of it.

---

## 2. Core Concepts: What Makes Inference Hard

Before diving into strategies, you need to understand _why_ LLM inference is inherently expensive and different from classical ML inference.

### 2.1 Memory-Bandwidth Bound, Not Compute Bound

LLM inference is **memory-IO bound**, not compute bound. The bottleneck is loading model weights from GPU memory (HBM) into the compute units (CUDA cores / tensor cores), not the matrix multiplications themselves. For every forward pass, the full set of model weights must be streamed from HBM.

This is a counterintuitive but critical insight: a single request running through a 70B parameter model underutilizes the GPU's compute capacity massively. The GPU spends most of its time waiting for memory loads.

Batching is the primary antidote — by sharing a single weight-loading pass across many simultaneous requests, you amortize the memory bandwidth cost, increasing effective compute utilization.

### 2.2 Autoregressive Decoding: Prefill vs. Decode

LLMs generate tokens sequentially via two distinct phases:

- **Prefill phase**: The entire input prompt is processed in a single (potentially chunked) forward pass, producing the first output token and populating the KV cache. This phase is **compute-bound** and parallelizes well.
- **Decode phase**: Each subsequent token is generated one at a time, attending to all prior tokens. This phase is **memory-bandwidth-bound** and inherently sequential per sequence.

These two phases have different computational profiles and different optimal batch sizes. This asymmetry is a major source of complexity in inference optimization and motivates advanced techniques like _disaggregated serving_ (covered in Section 10).

### 2.3 Variable Output Length

Unlike classical ML models (e.g., a classifier that always returns a probability vector), LLMs produce outputs of variable length. A user might ask a simple yes/no question or request a 2,000-word essay. This variability is the root cause of inefficiency in naive batching approaches.

---

## 3. The Fundamental Axis: Batch vs. Online Inference

At the highest level, inference workloads fall into two categories:

### Batch (Offline) Inference

Requests are collected and processed together in groups, not immediately upon arrival. The system prioritizes **throughput**over latency. Latency is either irrelevant (e.g., a nightly job) or acceptable at a higher level (e.g., minutes rather than milliseconds).

**Characteristics:**

- Inputs are known in advance or can be queued
- Optimized to maximize tokens/second (TPS) or requests/second (RPS) at minimum cost
- GPU utilization can be pushed very high
- Suitable for non-interactive workloads

**Typical use cases:**

- Document summarization pipelines run overnight
- Embedding generation for a vector database refresh
- Bulk content moderation or classification
- Fine-tuning dataset pre-processing
- Generating product descriptions for an entire catalog

### Online (Real-Time) Inference

Each request is served as it arrives, with the system optimizing for **low latency**. The user (human or machine) is waiting for a response in real time.

**Characteristics:**

- Requests arrive unpredictably (Poisson-distributed in practice)
- Time-to-first-token (TTFT) and inter-token latency (ITL) matter directly for user experience
- Must handle traffic spikes without degrading response time
- Harder to achieve high GPU utilization

**Typical use cases:**

- Chatbots and conversational AI
- Copilot/autocomplete features (code, email, writing)
- Real-time fraud detection
- Autonomous agents acting in response to live events

> **Key insight:** These are not mutually exclusive. Production systems often serve both online and batch workloads simultaneously, using priority queuing to ensure interactive requests are never blocked by batch jobs.

---

## 4. Batching Strategies in Depth

The word "batching" is overloaded. There are four distinct batching strategies, each with different characteristics.

### 4.1 No Batching

Each request is processed one at a time: receive request → run model → return result → receive next request.

This is almost never acceptable in production. The GPU sits idle between requests and the memory-bandwidth inefficiency described in Section 2 is at its worst. The only valid use case is local development or single-user prototype environments.

### 4.2 Static Batching

The system collects requests until a **fixed batch size** is reached, then processes the entire batch together. All requests in the batch must complete before results are returned.

**How it works:**

1. Requests queue up in a waiting pool
2. Once `N` requests are available (or a timeout fires), all `N` are packed into a tensor
3. The full batch runs through the model in one forward pass
4. All `N` results are returned simultaneously

**The padding problem:** Because transformer models require inputs of uniform length within a batch, shorter prompts are padded with dummy tokens to match the longest sequence in the batch. If one request has 50 tokens and another has 2,000, the 50-token request carries 1,950 padding tokens through every layer. This wastes compute proportionally.

**The head-of-line blocking problem:** Every request in the batch must wait for the _slowest_ request to finish generation. A simple "yes or no" question gets held up by a request asking for a 500-word essay.

```
Batch (size=4):
  Req A: "Hello" → "Hi there!"              (2 tokens out)   ✓ DONE
  Req B: "Summarize War and Peace" → ...    (800 tokens out)  ← everyone waits for this
  Req C: "What is 2+2?" → "4"              (1 token out)     ✓ DONE
  Req D: "Translate this paragraph" → ...  (60 tokens out)   ✓ DONE

GPU sits idle once A, C, D finish — waiting for B.
```

**When to use static batching:**

- Offline/batch inference jobs where latency is irrelevant
- All inputs are known ahead of time and have similar lengths
- Maximum throughput per GPU-hour is the sole objective
- Image generation workloads (Stable Diffusion), where generation time is roughly uniform

### 4.3 Dynamic Batching

A refinement of static batching: the system starts processing a batch either when it reaches a configured maximum size _or_when a **timeout window expires**, whichever comes first.

```
Config: max_batch_size=16, timeout=100ms

t=0ms:  Request A arrives
t=30ms: Request B arrives
t=55ms: Request C arrives
t=100ms: Timeout fires → batch {A, B, C} is dispatched (partial batch)
```

**Advantages over static batching:**

- Reduces worst-case latency (requests aren't held indefinitely)
- Handles variable traffic gracefully
- Better for live traffic where a full batch may never form during low-traffic periods

**Remaining limitations:**

- Head-of-line blocking still exists within a batch
- Padding waste is still present
- Optimal for workloads with roughly uniform output lengths (e.g., image generation with a fixed number of diffusion steps)

For LLM inference specifically, dynamic batching is a significant improvement over static batching but still leaves substantial performance on the table, which brings us to the most important strategy.

### 4.4 Continuous Batching (In-Flight Batching / Iteration-Level Scheduling)

Continuous batching is the state-of-the-art strategy for LLM serving and is the engine behind vLLM, SGLang, TensorRT-LLM, and Hugging Face TGI. It was introduced by Orca (2022) and has since become the industry standard.

**The core idea:** Instead of processing requests at the _batch level_, scheduling happens at the _token level_ (iteration level). After every single decoding step, the scheduler can:

- Remove completed sequences from the batch
- Insert new waiting requests into the freed slots

This eliminates head-of-line blocking entirely. The batch composition changes dynamically at every forward pass.

```
Token step 1:  Batch = [Req A (decoding), Req B (decoding), Req C (decoding), Req D (decoding)]
Token step 2:  Req C finishes → Batch = [Req A, Req B, Req D, Req E (new!)]
Token step 3:  Req A finishes → Batch = [Req B, Req D, Req E, Req F (new!)]
...GPU never sits idle...
```

**Three underlying techniques that make this work:**

1. **KV Caching**: The key-value attention states for each token are stored so they don't need to be recomputed. Each new decoding step only processes the latest token, reading prior states from cache.
    
2. **Chunked Prefill**: Long prompts that can't fit in GPU memory in one pass are split into chunks. Each chunk runs a partial prefill, storing intermediate KV states as it goes. This enables mixing prefill and decode operations in the same batch.
    
3. **Ragged Batching**: Instead of padding all sequences to a uniform length (creating a rectangular tensor), ragged batching operates on variable-length sequences simultaneously using attention masks to control which tokens attend to which. This eliminates padding waste entirely.
    

**Performance impact:** Anyscale benchmarks demonstrated up to **23x throughput improvement** over naive static batching, with reduced p50 latency, when combining continuous batching with memory optimizations like PagedAttention.

**When to use continuous batching:**

- Any interactive / online inference workload
- LLM serving with highly variable output lengths
- Conversational AI, code completion, agentic systems
- Production serving at scale where cost-per-token matters

---

## 5. Online Inference Strategies

Beyond _how_ requests are batched, the serving _architecture_ matters significantly for online inference.

### 5.1 Request Routing and Priority Queues

In mixed workloads, a priority queue separates interactive (high priority) from batch (low priority) requests. Schedulers allocate GPU slots to high-priority requests first. This allows you to co-locate batch jobs on the same hardware without degrading interactive latency.

```python
# Conceptual priority queue pattern
import heapq

HIGH_PRIORITY = 0  # interactive
LOW_PRIORITY  = 1  # batch jobs

queue = []
heapq.heappush(queue, (HIGH_PRIORITY, request_id, request))
heapq.heappush(queue, (LOW_PRIORITY,  batch_id,   batch_request))

# Scheduler always serves HIGH_PRIORITY first
priority, rid, req = heapq.heappop(queue)
```

### 5.2 Adaptive Batch Sizing

Rather than fixed batch sizes, production systems should dynamically adjust batch size based on current system state: GPU memory utilization, current queue depth, and real-time latency measurements. A 2025 paper ("Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching") demonstrated throughput gains of 8–28% and capacity improvements of 22% over static batch sizes by continuously monitoring memory utilization and adjusting batch size to comply with SLA constraints.

### 5.3 Speculative Decoding

A technique where a small, fast "draft" model generates multiple candidate tokens, which a larger "verifier" model then validates in a single parallel forward pass. When the draft tokens are correct (which happens frequently for predictable continuations), the system produces multiple tokens per decoding step, dramatically reducing latency.

Note that speculative decoding and large batch sizes are _incompatible_ — larger batches consume the compute budget that speculation needs to run the verifier in parallel. This is a subtle but important nuance in system design.

### 5.4 Disaggregated Serving (Prefill-Decode Disaggregation)

Since prefill and decode have fundamentally different computational profiles, advanced systems separate them onto different GPUs or nodes:

- **Prefill workers**: Optimized for compute-heavy, parallelizable work. Process incoming prompts.
- **Decode workers**: Optimized for memory-bandwidth-heavy, sequential work. Generate tokens.

This prevents the two phases from competing for the same GPU resources, which becomes a serious problem under heavy traffic. Disaggregation is a frontier technique becoming more common in large-scale deployments (2025+).

---

## 6. Streaming Inference

Streaming inference occupies a middle ground — it processes a continuous flow of incoming data in real time, typically from event streams (Kafka, Kinesis) or sensor feeds. This is distinct from both traditional batch (large, bounded job) and online (single user request) patterns.

**Use cases:**

- IoT monitoring and anomaly detection
- Live translation of audio/video streams
- Real-time recommendation on user activity streams
- Fraud detection on financial transaction feeds

**Infrastructure requirements:** Streaming inference demands robust, fault-tolerant infrastructure (e.g., Kafka + a serving engine) to handle high, continuous throughput without downtime. The inference engine must process each event within a time window (micro-batch) to avoid unbounded queue growth.

---

## 7. Key Performance Metrics

As an AI Engineer, you need a precise vocabulary for measuring inference performance. These are the metrics you must instrument:

|Metric|Abbreviation|Definition|Primary relevance|
|---|---|---|---|
|Time to First Token|TTFT|Latency from request submission to first output token|Online / interactive|
|Inter-Token Latency|ITL|Time between consecutive output tokens|Online / streaming UX|
|Time Per Output Token|TPOT|Average decode time per token|System efficiency|
|Tokens Per Second|TPS|Total tokens generated per second across all requests|Throughput / capacity|
|Requests Per Second|RPS|Total requests completed per second|Throughput|
|P50 / P95 / P99 Latency|—|Percentile latencies; P99 captures tail behavior|SLO compliance|
|GPU Utilization|—|% of GPU compute or memory bandwidth in use|Hardware efficiency|
|Saturation Throughput|—|Max TPS while still meeting SLOs|Capacity planning|

**Setting SLOs (Service Level Objectives):** For interactive applications, define explicit targets such as P95 TTFT < 500ms and P99 ITL < 100ms. Use P95/P99 rather than averages — averages mask tail latency that users experience disproportionately.

---

## 8. The Latency–Throughput Tradeoff

The most important nuance in inference optimization is that **latency and throughput are fundamentally in tension**, and the relationship is non-linear.

As batch size increases:

- **Throughput increases** — more requests share each weight-load operation
- **Latency increases** — each request waits longer (for the batch to fill, and for co-located requests to finish)
- **Throughput eventually saturates** — once the GPU is fully utilized, adding more requests to the batch doesn't increase tokens/sec but does increase latency

This creates a characteristic curve: throughput rises steeply at first, then flattens as batch size grows, while latency increases monotonically.

```
Throughput
    |         ___________
    |        /
    |       /
    |      /
    |_____/_____________________ Batch Size
    
Latency
    |                    /
    |                   /
    |                  /
    |_________________/_________ Batch Size
```

**Practical implications:**

- For latency-sensitive workloads: use smaller effective batch sizes, accept lower GPU utilization
- For throughput-maximizing workloads: push batch size until throughput saturates (not until OOM)
- The "knee of the curve" — where throughput gain per unit of latency cost is highest — is your production operating point

---

## 9. GPU Memory & the KV Cache

The KV (key-value) cache is the most important memory structure in LLM inference and directly controls how many requests you can batch simultaneously.

### What is the KV Cache?

During the attention mechanism, each token produces key and value vectors that subsequent tokens must attend to. Rather than recomputing these for all prior tokens on every decoding step, they are stored in GPU memory (VRAM). For a sequence of length `L` using a model with `H` heads and `D` head-dimension across `N` layers:

```
KV cache size per token = 2 × N × H × D × dtype_bytes
```

For a 70B parameter model (e.g., Llama-3 70B), this is roughly 1-2 MB _per token per request_. With a batch of 100 requests each generating 2,000 tokens, KV cache alone can consume tens of gigabytes.

### PagedAttention

vLLM's key innovation: rather than pre-allocating a contiguous block of VRAM for each request's maximum possible KV cache, PagedAttention manages KV cache in fixed-size "pages" (like virtual memory in an OS). Pages are allocated on demand and can be shared across requests with identical prefixes. This dramatically increases the number of concurrent requests that can be served on a given GPU.

```python
# Conceptual: with standard allocation
kv_cache = torch.zeros(max_seq_len, num_layers, 2, num_heads, head_dim)
# ^ Reserved upfront regardless of actual generation length — very wasteful

# With PagedAttention
# Memory is allocated page-by-page as tokens are generated
# Short sequences don't waste memory allocated for long ones
```

### KV Cache and Batching

The KV cache is the primary constraint on batch size for online inference. More concurrent requests = more KV cache = less room for model weights and activations. This is why adaptive batch sizing (Section 5.2) that monitors GPU memory utilization is so important.

---

## 10. Complementary Techniques

These techniques interact with batching strategy and must be understood holistically.

### Quantization

Reducing model weight precision (FP32 → FP16 → INT8 → FP8 → INT4) shrinks model size and accelerates memory loads. This directly enables larger batch sizes (more requests fit in VRAM) and reduces TTFT. The tradeoff is potential output quality degradation that must be measured carefully per model and task.

Quantizing the KV cache itself (KV quantization) is particularly powerful for batching: it allows more concurrent sequences in memory simultaneously.

### Prefix Caching

If many requests share a common prefix (e.g., a long system prompt), the KV states for that prefix can be computed once and cached. Subsequent requests with the same prefix skip the prefill for those tokens entirely. This is especially impactful for RAG (retrieval-augmented generation) patterns and multi-turn conversations.

### Tensor Parallelism and Pipeline Parallelism

For models too large for a single GPU, weights are sharded across multiple GPUs. This reduces per-GPU memory pressure and enables larger effective batch sizes, but introduces inter-GPU communication overhead that must be managed carefully.

### Model Distillation and Pruning

Replacing a large model with a smaller distilled version reduces compute and memory requirements uniformly across all inference patterns, directly improving cost/token at any batch size.

---

## 11. Tooling & Frameworks

### LLM Inference Engines

|Framework|Key features|Best for|
|---|---|---|
|**vLLM**|Continuous batching, PagedAttention, prefix caching; de facto standard|General LLM serving, highest throughput|
|**SGLang**|Continuous batching + structured generation, RadixAttention for prefix sharing|Complex agentic pipelines, structured outputs|
|**TensorRT-LLM**|NVIDIA-optimized, in-flight batching, kernel-level tuning|Maximum performance on NVIDIA hardware|
|**LMDeploy**|Quantization-first, hardware-flexible, persistent batching|Edge/embedded deployment|
|**Hugging Face TGI**|Easy to deploy, continuous batching, broad model support|Rapid prototyping, open-source models|

### General Inference Serving

|Framework|Notes|
|---|---|
|**NVIDIA Triton**|Multi-framework, ensemble models, concurrent model execution, dynamic batching|
|**TorchServe**|Native PyTorch, built-in batch processing, custom handlers|
|**ONNX Runtime**|Cross-platform, hardware-agnostic, supports CUDA/TensorRT/DirectML acceleration|
|**TensorFlow Serving**|REST/gRPC, versioning/hot-swap, production-grade for TF models|

### Cloud Managed Inference

|Provider|Offering|
|---|---|
|AWS SageMaker|Serverless + real-time endpoints, auto-scaling, batch transform|
|Google Vertex AI|Sub-100ms global latency, BigQuery integration, AutoML pipelines|
|Azure ML|Managed online/batch endpoints, AKS integration|
|RunPod / Hyperstack|GPU-first, vLLM-native, cost-effective for open model hosting|

---

## 12. Decision Framework: Choosing the Right Strategy

Use the following decision tree when designing your inference system:

```
1. Is the workload interactive (user waiting)?
   ├── YES → Online inference
   │         ├── Does output length vary significantly?
   │         │   ├── YES → Continuous batching (vLLM/SGLang/TGI)
   │         │   └── NO  → Dynamic batching (Triton, TorchServe)
   │         └── Is latency the primary SLO (< 200ms TTFT)?
   │               ├── YES → Consider speculative decoding + smaller batch size
   │               └── NO  → Optimize for throughput within SLO
   │
   └── NO  → Batch (offline) inference
             ├── All inputs known ahead of time?
             │   ├── YES → Static batching at max batch size for GPU
             │   └── NO  → Dynamic batching with large timeout window
             └── What is the primary metric?
                   ├── Min cost/token → Max batch size, high-throughput GPU
                   └── Min wall-clock time → Horizontal scale + static batching

2. Mixed workloads?
   → Priority queue separating interactive (HIGH) from batch (LOW)
   → Continuous batching engine handles both; batch jobs fill idle capacity
```

### Quick Reference by Use Case

|Use case|Recommended strategy|Key metric|
|---|---|---|
|Chatbot / conversational AI|Continuous batching|TTFT, ITL|
|Code completion / copilot|Continuous batching + speculative decoding|P95 TTFT|
|Document summarization (async)|Static or dynamic batching|TPS, cost/token|
|Embedding generation (bulk)|Static batching at max batch size|TPS|
|Fraud detection (real-time)|Online, low-latency with small batch|P99 latency|
|Nightly data pipeline|Static batching, scheduled|Wall-clock time, cost|
|RAG with shared system prompts|Continuous batching + prefix caching|TTFT, cache hit rate|
|Multi-model serving|Triton + dynamic batching per model|Per-model SLOs|

---

## 13. Python Code Examples

### Example 1: Static Batching with HuggingFace (offline pipeline)

```python
from transformers import pipeline
import torch

# Offline batch inference — maximize throughput
generator = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda",
    torch_dtype=torch.float16,
)

prompts = [
    "Summarize the following document: ...",
    "Translate to French: ...",
    "Extract key entities from: ...",
    # ... hundreds more
]

# Static batching: process all at once in fixed-size chunks
BATCH_SIZE = 32
results = []

for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i : i + BATCH_SIZE]
    outputs = generator(
        batch,
        max_new_tokens=256,
        batch_size=BATCH_SIZE,
        do_sample=False,
    )
    results.extend(outputs)
```

### Example 2: Online Inference with vLLM (continuous batching)

```python
from vllm import LLM, SamplingParams

# vLLM uses continuous batching internally —
# requests submitted concurrently are automatically batched
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,  # Leave 10% headroom
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)

# For online serving, use vllm.entrypoints.openai.api_server
# For offline batch, call generate() directly:
prompts = ["Tell me about quantum computing", "What is machine learning?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Example 3: Async Online Serving with vLLM AsyncLLMEngine

```python
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enable_prefix_caching=True,  # Cache shared prefixes (e.g., system prompts)
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_streaming(prompt: str, request_id: str):
    """Stream tokens as they are generated — true online inference UX."""
    sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
    
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            # Each iteration yields the next token as soon as it's ready
            yield output.outputs[0].text

async def main():
    # Simulate concurrent online requests — engine handles continuous batching
    tasks = [
        generate_streaming(f"Question {i}: ...", request_id=str(i))
        for i in range(10)
    ]
    # All 10 requests are served concurrently via continuous batching
    for task in tasks:
        async for token in task:
            print(token, end="", flush=True)

asyncio.run(main())
```

### Example 4: Measuring TTFT and Throughput

```python
import time
import asyncio
import httpx

async def measure_ttft(prompt: str, server_url: str) -> dict:
    """Measure Time-to-First-Token for a streaming response."""
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 256,
    }

    start_time = time.perf_counter()
    ttft = None
    total_tokens = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{server_url}/v1/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    total_tokens += 1

    total_time = time.perf_counter() - start_time
    return {
        "ttft_ms": ttft * 1000,
        "total_time_s": total_time,
        "tokens": total_tokens,
        "tps": total_tokens / total_time,
    }

async def load_test(num_concurrent: int, server_url: str):
    """Run concurrent requests and measure p50/p95/p99 TTFT."""
    prompts = [f"Explain concept #{i} in detail." for i in range(num_concurrent)]
    results = await asyncio.gather(*[measure_ttft(p, server_url) for p in prompts])

    ttfts = sorted(r["ttft_ms"] for r in results)
    n = len(ttfts)
    print(f"Concurrency: {num_concurrent}")
    print(f"P50 TTFT: {ttfts[int(n * 0.50)]:.1f}ms")
    print(f"P95 TTFT: {ttfts[int(n * 0.95)]:.1f}ms")
    print(f"P99 TTFT: {ttfts[int(n * 0.99)]:.1f}ms")
    print(f"Avg TPS: {sum(r['tps'] for r in results) / n:.1f}")
```

### Example 5: Batch Inference with Anthropic's Batch API

```python
import anthropic

client = anthropic.Anthropic()

# Use Anthropic's Message Batches API for large-scale offline inference
# Cost-effective: designed for throughput, not latency
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"doc-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": f"Summarize document {i}"}],
            },
        }
        for i in range(1000)
    ]
)

print(f"Batch ID: {batch.id}, Status: {batch.processing_status}")

# Poll for completion (batch jobs may take minutes to hours)
import time
while True:
    result = client.messages.batches.retrieve(batch.id)
    if result.processing_status == "ended":
        break
    time.sleep(30)

# Stream results
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(result.custom_id, result.result.message.content[0].text)
```

---

## 14. Subtle Nuances & Common Pitfalls

### 14.1 Batch Size Saturation is Model- and Hardware-Specific

There is no universal "optimal batch size." The throughput-saturation point depends on model architecture, parameter count, sequence length, GPU VRAM, and memory bandwidth. Always profile empirically on your target hardware. A batch size of 32 may be well past saturation for a 70B model on an A100 but barely warm for a 7B model.

### 14.2 Speculative Decoding Degrades Under High Concurrency

Speculative decoding is highly effective for latency reduction on single-user or low-concurrency workloads. However, it requires running a draft model _and_ the target verifier model simultaneously, consuming GPU compute. Under high batch loads, this compute budget disappears — the verifier's parallel verification step can't fit alongside a large decode batch. Don't design a system that relies on speculative decoding for latency when batch sizes may be large.

### 14.3 Prefill-Decode Resource Contention

When prefill and decode phases run on the same GPU under heavy traffic, they compete for resources. Prefill is compute-bound; decode is memory-bandwidth-bound. This contention causes both phases to underperform. For high-volume production systems, disaggregated serving (dedicated prefill and decode workers) is worth the architectural complexity.

### 14.4 Continuous Batching Doesn't Eliminate All Latency Costs

Continuous batching improves latency on average and dramatically improves throughput, but a newly arriving request still waits until the next scheduler iteration (one decode step ≈ 10–50ms). Under very high load, the queue of waiting requests can grow, increasing TTFT for new arrivals. Monitor queue depth alongside GPU utilization.

### 14.5 Dynamic Batching Timeout Tuning is Workload-Dependent

For dynamic batching, the timeout window requires careful tuning. Too short: batches are often partial, GPU is underutilized. Too long: latency for requests that arrive at the start of an empty window is unnecessarily high. The right value depends on your P95 inter-arrival time. If requests arrive every 50ms on average, a timeout of 20–30ms is usually appropriate.

### 14.6 Padding Waste Can Dominate Cost at Large Batch Sizes

When using static or dynamic batching with very heterogeneous sequence lengths, padding can consume 30–70% of total compute. Before implementing continuous batching (which requires more infrastructure), consider bucketed batching: group requests by approximate sequence length before batching them together. This simple optimization can cut padding waste dramatically with minimal engineering effort.

### 14.7 Memory Pressure Can Cause OOM Mid-Batch

For long-running generation with large batch sizes, the KV cache grows over time. A batch that fits in GPU memory at token step 1 may OOM at token step 500 if you didn't account for KV cache growth. Always reserve headroom (e.g., `gpu_memory_utilization=0.90` in vLLM) and test with realistic output lengths, not just input lengths.

### 14.8 The Incompatibility of Some Optimizations

Some techniques are mutually incompatible or trade off against each other:

- **Large batch size ↔ speculative decoding**: Compete for compute
- **KV quantization ↔ output quality**: May degrade for precision-sensitive tasks
- **Prefix caching ↔ temperature > 0**: Cached results are deterministic; if exact-match caching is used, it overrides sampling
- **Tensor parallelism ↔ inter-GPU latency**: More GPUs = more communication overhead; diminishing returns beyond a certain point

---

## 15. Summary Reference Table

|Dimension|No Batching|Static Batching|Dynamic Batching|Continuous Batching|
|---|---|---|---|---|
|Throughput|Very low|High|Medium-High|Very high|
|Latency|Low (per request)|High (head-of-line)|Medium|Low-Medium|
|GPU utilization|Very low|High|Medium-High|Very high|
|Variable output length|N/A|Poor|Poor|Excellent|
|Padding waste|None|High|Medium|Near-zero|
|Implementation complexity|Trivial|Low|Low|High|
|Best for|Dev/testing|Offline batch jobs|Image gen, uniform outputs|LLM online serving|
|Key frameworks|Any|HuggingFace pipeline|Triton, TorchServe|vLLM, SGLang, TGI|

---

_Report compiled: April 2026. Sources include recent research from ACL 2025, arXiv (2025), and engineering publications from Red Hat, Anyscale, BentoML, Hugging Face, and Hyperstack._