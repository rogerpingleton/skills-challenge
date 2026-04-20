# Caching implementations

Here's a comprehensive breakdown of caching and its impact on inference optimization — everything you need to know as an AI Engineer.

---

# Caching in AI Inference Optimization

## 1. Why Caching Matters

Without optimization, running a 70B parameter model like Llama-2 requires 140GB of GPU memory at FP16 precision, and naive implementations often take 5–15 seconds per response. The challenge stems from three factors: memory bandwidth bottlenecks, sequential token generation, and massive model sizes.

Inference optimization targets memory bandwidth efficiency, request latency, and serving cost for production workloads — and inference is memory-bound while training is compute-bound.

Caching sits at the center of solving all three of those problems.

---

## 2. The Types of Caching

### 2.1 KV Cache (Key-Value Cache) — The Foundation

This is the most fundamental caching mechanism in transformer-based models.

Self-attention depends on all previous tokens; recomputing keys and values for each new token would be prohibitively expensive. The KV cache stores these computations so they can be reused, dramatically speeding up decode.

Without a KV cache, every new token generated requires recomputing attention over the entire prior context from scratch — O(n²) work per step. With it, only the new token's keys and values need to be computed and appended.

**Memory tradeoff:** KV cache memory grows linearly with sequence length, requiring careful memory management for long conversations. A 20-turn conversation sees 10–20x speedup compared to recomputing from scratch.

**Python example (Hugging Face):**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

inputs = tokenizer("Tell me about caching", return_tensors="pt")

# use_cache=True (default) — reuses KV tensors across generation steps
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    use_cache=True  # disabling this causes full recomputation every step
)
```

---

### 2.2 Prefix Caching (Prompt Caching / Automatic Prefix Caching)

This extends basic KV caching _across requests_, not just within a single generation.

The core idea is simple — cache the KV-cache blocks of processed requests, and reuse these blocks when a new request comes in with the same prefix as previous requests.

This is extremely valuable when many requests share a common prefix: a system prompt, a large document, tool definitions in an agentic workflow, etc.

**How vLLM implements it:** vLLM maps logical KV blocks to their hash value and maintains a global hash table of all the physical blocks. All KV blocks sharing the same hash value — such as shared prefix blocks across two requests — can be mapped to the same physical block and share memory space.

**Eviction policy:** When there are no free blocks left, vLLM evicts a KV block based on LRU (Least Recently Used) ordering, prioritizing eviction of blocks at the end of the longest prefix.

**Cost impact at the API level:** Anthropic prefix caching delivers 90% cost reduction at $0.30/M tokens vs. $3.00/M base — with break-even at just 1.4 reads per cached prefix. OpenAI automatic caching provides a 50% discount with no code changes needed for prompts exceeding 1,024 tokens.

**Practical tip for engineers:** Structure your prompts so the static content (system prompt, tool definitions, documents) always appears at the _beginning_, before any dynamic per-user content. Even a single character difference in the prefix will cause a cache miss.

```python
# BAD — dynamic content inserted before static system prompt
messages = [
    {"role": "user", "content": f"User ID: {user_id}\n\n{static_system_prompt}"},
]

# GOOD — static prefix first, dynamic content at the end
messages = [
    {"role": "system", "content": static_system_prompt},  # cacheable
    {"role": "user", "content": f"User ID: {user_id}"},   # dynamic, goes last
]
```

---

### 2.3 PagedAttention — Memory Management for KV Caching

Raw KV caches suffer from memory fragmentation when stored contiguously. PagedAttention solves this.

The core idea of PagedAttention is to partition the KV cache of each request into KV Blocks. Each block contains the attention keys and values for a fixed number of tokens. The PagedAttention algorithm allows these blocks to be stored in non-contiguous physical memory, eliminating fragmentation.

vLLM breaks up prompts into fixed-sized token sequences referred to as blocks — 16 tokens by default. LMCache uses a larger 256-token block by default to reduce the overhead of managing references and to better amortize per-block transfer overhead.

PagedAttention and Prefix Caching together optimize GPU memory by up to 40% and improve throughput up to 3x.

**Enabling APC in vLLM:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,   # Automatic Prefix Caching
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# All requests sharing the same system prompt will benefit from cached KV blocks
responses = llm.generate(prompts, sampling_params)
```

---

### 2.4 Multi-Tier KV Caching (GPU → CPU → Disk)

When GPU memory is exhausted, tiered caching extends the cache hierarchy.

vLLM takes a hierarchical approach to KV caching: first it checks for cache blocks in GPU memory; on a cache miss it progresses to CPU memory; and on another miss it tries to retrieve cache blocks over any configured KV connectors. LMCache works with vLLM over this KV connector interface, working to store or stream cache blocks it locates.

This is relevant when serving long-context workloads (e.g., 128K token documents) where all active KV caches can't fit in GPU VRAM simultaneously.

---

### 2.5 Semantic Caching

This operates at a higher level — caching _outputs_ of similar (not identical) queries.

Model output caching stores and reuses inference results for identical or similar inputs, eliminating redundant computations for frequently requested inferences.

The mechanism works by embedding incoming queries and performing a similarity search against cached (query, response) pairs. If the embedding distance is below a threshold, the cached response is returned without any model call.

Research shows 31% of LLM queries exhibit semantic similarity to previous requests — representing massive efficiency gains for deployments without caching infrastructure. GPTCache achieves 61.6–68.8% cache hit rates with 97%+ positive hit accuracy.

**Python example using GPTCache:**

```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=onnx.dimension)
)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()

# Subsequent semantically similar queries will hit the cache
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

**Caveat:** Static similarity thresholds (e.g., `0.8`) work poorly across diverse query types. Adaptive thresholds improve accuracy across diverse queries, and the SCALM pattern detection approach achieves a 63% improvement in cache hit ratio and 77% reduction in token usage.

---

### 2.6 Intermediate Activation Caching

Intermediate activation caching stores model activations for requests with similar input prefixes, particularly benefiting language models where conversation context can be reused across multiple exchanges. Embedding and feature caching reduces preprocessing overhead while improving overall system efficiency.

This is especially useful in RAG pipelines — cache your document embeddings so they're not recomputed on every query.

```python
import hashlib
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed_document(doc_text: str) -> np.ndarray:
    """Cache embeddings keyed by document content hash."""
    return embedding_model.encode(doc_text)

# Or with an explicit Redis-backed cache for distributed systems:
import redis, pickle

r = redis.Redis()

def get_or_compute_embedding(text: str) -> np.ndarray:
    key = hashlib.sha256(text.encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return pickle.loads(cached)
    embedding = embedding_model.encode(text)
    r.set(key, pickle.dumps(embedding), ex=3600)  # 1hr TTL
    return embedding
```

---

## 3. KV Cache-Aware Routing (Distributed Systems)

In multi-replica deployments, naive load balancing sends requests to any available replica — bypassing warm caches entirely.

In traditional deployments, even if KV caches are enabled inside the model server (like vLLM), the gateway is unaware of the cache state. KV cache-aware routing with llm-d solves this by intelligently directing requests to pods that already hold relevant context in GPU memory, reducing latency, improving throughput, and lowering operational costs.

A demonstrated 87% cache hit rate and 88% faster TTFT (Time to First Token) for warm cache hits underscores the real-world impact of this technology.

This is particularly important for RAG, customer support, and code generation workloads where context reuse is high.

---

## 4. The Interactions and Tradeoffs

Sometimes techniques are symbiotic or incompatible. For example, quantizing the KV cache alleviates a bottleneck in disaggregation, but increasing batch size reduces the compute available for speculation. An inference engineer's challenge is always to create a balanced set of optimizations that delivers more than the sum of its parts.

Here's a practical decision matrix:

|Scenario|Best Caching Strategy|
|---|---|
|Multi-turn chatbot|KV cache + Prefix caching (static system prompt)|
|RAG pipeline|Embedding cache + Prefix caching for document context|
|High-volume FAQ bot|Semantic cache (high query repetition)|
|Agentic workflows|Prefix caching for fixed agent system prompts|
|Long-context documents|Multi-tier KV cache (GPU+CPU+disk)|
|Multi-tenant SaaS|KV cache with `cache_salt` isolation per tenant|

---

## 5. Monitoring What Matters

Key metrics to track in production:

- **Cache Hit Rate** — target 80%+ for prefix caching with stable prompts, 30–60% for semantic caching
- **TTFT (Time to First Token)** — most directly impacted by prefix cache hits
- **KV Cache Memory Utilization** — keep GPU KV cache pressure below ~90% to avoid eviction thrash
- **Eviction Rate** — high eviction rates signal your cache is too small or your eviction policy is wrong

Enable KV caching for multi-turn conversations, monitor memory usage, and implement cache eviction policies for long conversations.

---

## 6. Summary Hierarchy

The full caching stack, from lowest to highest abstraction:

```
┌────────────────────────────────────────┐
│        Semantic Cache (output-level)   │  ← Highest savings, coarsest
├────────────────────────────────────────┤
│   Prefix / Prompt Cache (KV reuse)     │  ← Cross-request KV reuse
├────────────────────────────────────────┤
│  PagedAttention (intra-request KV mgmt)│  ← Memory fragmentation solved
├────────────────────────────────────────┤
│     KV Cache (within-generation)       │  ← Baseline, always on
├────────────────────────────────────────┤
│  Embedding / Activation Cache (RAG)    │  ← Preprocessing layer
└────────────────────────────────────────┘
```

Each layer compounds savings on the one below it. A well-configured stack combining all layers can realistically achieve 10x+ performance improvements in production environments, and up to 90% cost reduction on input token processing for long, stable prompts.