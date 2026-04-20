# Caching architectures

## 1. What Is Caching in AI Engineering?

In traditional software, caching stores the results of expensive operations so they can be reused rather than recomputed. In AI engineering, the same idea applies across several layers of the stack — from individual LLM API calls all the way down to the key-value (KV) tensors inside the transformer attention mechanism.

The motivation is compelling. LLM API calls are expensive (token cost scales linearly with request volume), slow (latency-sensitive applications notice every added round-trip), and often redundant (a significant portion of production traffic is semantically identical queries phrased differently). Effective caching tackles all three problems simultaneously.

A well-designed caching strategy can:

- Reduce LLM API spend by 60–80% in high-repetition workloads
- Cut time-to-first-token (TTFT) dramatically for repeated or similar queries
- Enable offline or low-connectivity operation for static knowledge domains
- Reduce GPU utilisation and infrastructure cost at inference time

---

## 2. Types of Caching Architectures

### 2.1 Exact-Match (Key-Value) Caching

The simplest form. A hash of the full prompt (and model configuration string) is used as the cache key; the LLM response is the value. On a subsequent request, if the hash matches, the stored response is returned immediately without calling the LLM.

**When to use:** Internal tooling, FAQ bots, repeated batch jobs, any scenario where users are expected to send identical prompts verbatim.

**Limitations:** A single word change causes a cache miss. Does not handle natural language variation.

### 2.2 Semantic Caching

Instead of hashing the raw prompt text, the prompt is converted into a vector embedding. New requests are compared against cached embeddings using cosine similarity (or another distance metric). If the similarity score exceeds a threshold (e.g., ≥ 0.80), the cached response is returned.

This means "What is the capital of France?" and "Can you tell me the capital city of France?" both hit the same cache entry — demonstrating real-world speed improvements of 5x or more on similar queries.

**When to use:** Customer support bots, chatbots, RAG applications, any system with high paraphrase rates.

**Limitations:** Embedding generation adds latency on cache misses. Requires a vector store. Similarity thresholds need tuning to avoid false positives. Multi-tenant systems require careful partition isolation to prevent data leakage across users.

**Security note:** When sensitivity of queries varies, use tiered thresholds — stricter (≥ 0.95) for sensitive data, relaxed (≥ 0.75) for general knowledge — and isolate cache partitions per data classification level.

### 2.3 Prompt / Prefix Caching (Provider-Side)

Major LLM providers now offer server-side caching of previously computed attention states for repeated prompt prefixes. This avoids reprocessing the same system prompt, policy block, knowledge base, or few-shot examples on every request.

|Provider|Mechanism|Minimum Tokens|TTL|Notes|
|---|---|---|---|---|
|Anthropic|Developer-controlled, explicit `cache_control`breakpoints|~1,024|Configurable|Guaranteed discount on cache hits|
|OpenAI|Automatic on GPT-4o+|~1,024|5–60 min|Activates automatically, exact prefix only|
|Google|Implicit (no guarantee) + explicit context caching|~4,096|Up to 24 hours|Explicit caching provides guaranteed discounts|

**When to use:** Agents with long system prompts, RAG systems with large document preambles, multi-turn conversations, any app that sends the same static prefix on every call.

**Key pattern:** Put static content (system prompt, tool definitions, reference documents) in the cached prefix. Keep dynamic content (user message, session state) in the fresh suffix. Never place timestamps, user IDs, or session-specific data in the cached portion.

### 2.4 KV-Cache (Model Internals)

During transformer inference, the key and value matrices computed for each token in the attention mechanism can be stored and reused for subsequent tokens in the same sequence. This is the KV-cache — a standard component of all major LLM serving frameworks (vLLM, TensorRT-LLM, etc.).

In distributed inference setups, KV-cache aware routing (as implemented in frameworks like llm-d) directs new requests to the GPU pod that already holds the most relevant cached KV blocks, reducing redundant computation and lowering latency significantly.

**When to use:** Primarily a serving infrastructure concern rather than application-layer. Relevant when operating self-hosted models at scale.

### 2.5 Cache-Augmented Generation (CAG)

CAG is an architectural pattern that pre-loads stable, static knowledge directly into the model's context window (or an adjacent cache), rather than fetching it dynamically via RAG at query time. It treats the cache as an extension of the model's working memory.

**Operational workflow:**

1. **Preparation** — Identify static knowledge domains (legal precedents, product specs, medical protocols).
2. **Caching** — Pre-compute and store the knowledge in a lightweight, accessible format.
3. **Generation** — At inference time, inject cached knowledge directly, skipping retrieval overhead.

**When to use:** Domains with stable, infrequently changing knowledge, where context windows are large enough to hold the required information. Less suited for rapidly changing data or terabyte-scale corpora.

**CAG vs RAG:** CAG wins on latency and coherence for static knowledge. RAG wins for freshness and large, dynamic corpora.

### 2.6 Distributed & Federated Caching

For high-availability production systems, cache state should be distributed across multiple nodes (using Redis Cluster, for example) rather than held in a single process. Federated caching extends this to multi-site or privacy-sensitive scenarios where each node maintains local caches that are synchronised without centralising raw data.

**Key distributed cache design choices:**

- Cluster mode for horizontal scalability
- Replication for fault tolerance and availability
- TTL-based eviction policies for memory management
- Node-local caching for lowest-latency reads

---

## 3. Application Architecture Patterns

The following diagram describes a layered caching architecture for a production LLM application:

```
┌──────────────────────────────────────────────────────────┐
│                      Client / UI                         │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│               AI Gateway / LLM Proxy                     │
│  (LiteLLM, Bifrost, or custom FastAPI)                   │
│                                                          │
│   ┌─────────────────────────────────────────────────┐    │
│   │  Layer 1: Exact-Match Cache (hash lookup)       │    │
│   │  Backend: Redis (in-memory, sub-ms lookup)      │    │
│   └──────────────────────┬──────────────────────────┘    │
│                          │ MISS                          │
│   ┌──────────────────────▼──────────────────────────┐    │
│   │  Layer 2: Semantic Cache (vector similarity)    │    │
│   │  Backend: Redis + vector index, or Milvus/FAISS │    │
│   └──────────────────────┬──────────────────────────┘    │
│                          │ MISS                          │
└──────────────────────────┼───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│              LLM Provider API                            │
│  (Anthropic / OpenAI / etc.)                             │
│                                                          │
│   ┌─────────────────────────────────────────────────┐    │
│   │  Layer 3: Provider-Side Prompt/Prefix Cache     │    │
│   │  (Anthropic cache_control, OpenAI auto-cache)   │    │
│   └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**How a request flows:**

1. Incoming query hashes to a key → checked in Redis exact-match cache.
2. On miss → embed the query → vector similarity search against semantic cache.
3. On miss → forward to LLM provider, with static system prompt marked for provider-side prefix caching.
4. Response stored in both caches; next similar request is served from Layer 2 or Layer 1.

---

## 4. Popular Tools & Services

|Tool / Service|Type|Language|Notes|
|---|---|---|---|
|**Redis**|Key-Value + Vector Store|Any|Battle-tested; supports exact-match and semantic via RediSearch/RedisVL|
|**GPTCache**|Semantic Cache Library|Python|Modular; swap embedding models, vector stores, eviction policies independently. Integrates with LangChain and LlamaIndex|
|**LangChain**|Framework with cache layer|Python|`set_llm_cache()` supports in-memory, SQLite, Redis, and GPTCache backends|
|**LiteLLM**|LLM Gateway|Python|Multi-provider proxy with built-in Redis semantic caching mode|
|**Bifrost**|LLM Gateway (Go)|Any|Dual-layer (exact hash + semantic) caching at the gateway level; OSS|
|**Milvus / Zilliz**|Vector Database|Any|Used as vector backend for GPTCache and custom semantic caches|
|**FAISS**|Vector Index (in-process)|Python|Lightweight, no server required; good for prototyping|
|**vLLM**|LLM Serving Engine|Python|Built-in KV-cache; use with llm-d for KV-cache aware routing|

---

## 5. Python Examples

### 5.1 In-Memory Exact-Match Cache

A simple, zero-dependency implementation using a Python dictionary. Good for prototyping or single-process applications.

```python
import hashlib
import json
from anthropic import Anthropic

client = Anthropic()
_cache: dict[str, str] = {}


def _cache_key(model: str, system: str, user_message: str) -> str:
    payload = json.dumps({"model": model, "system": system, "user": user_message}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def chat_with_cache(user_message: str, system: str = "You are a helpful assistant.") -> str:
    model = "claude-sonnet-4-20250514"
    key = _cache_key(model, system, user_message)

    if key in _cache:
        print("[CACHE HIT]")
        return _cache[key]

    print("[CACHE MISS] — calling API")
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = response.content[0].text
    _cache[key] = answer
    return answer


if __name__ == "__main__":
    q = "What is retrieval-augmented generation?"
    print(chat_with_cache(q))  # API call
    print(chat_with_cache(q))  # Cache hit
```

### 5.2 Redis Exact-Match Cache with LangChain

Uses Redis as a persistent, shared cache. Any process in your application cluster can benefit from a cached response.

```python
# pip install langchain langchain-community langchain-anthropic redis

from langchain_anthropic import ChatAnthropic
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache

# Connect to Redis and set as the global LangChain cache
set_llm_cache(RedisCache(redis_url="redis://localhost:6379", ttl=3600))

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

response1 = llm.invoke("Explain transformer attention in one paragraph.")
print("First call (API):", response1.content)

response2 = llm.invoke("Explain transformer attention in one paragraph.")
print("Second call (cache hit):", response2.content)
```

### 5.3 Semantic Cache with Redis + LangChain

Handles natural language variation — differently-worded but semantically equivalent queries return cached responses.

```python
# pip install langchain langchain-redis langchain-openai redis redisvl

from langchain_redis import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.globals import set_llm_cache

embeddings = OpenAIEmbeddings()  # or any HuggingFace embedding model

semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embeddings=embeddings,
    distance_threshold=0.15,  # lower = stricter; higher = more permissive
    ttl=3600,
)
set_llm_cache(semantic_cache)

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

r1 = llm.invoke("What is the capital of France?")
print("Original:", r1.content)

# Semantically similar — will hit the cache
r2 = llm.invoke("Can you tell me the capital city of France?")
print("Paraphrase (cache hit):", r2.content)

# Check via timing
import time

start = time.time()
llm.invoke("Which city serves as France's capital?")
print(f"Third call took: {time.time() - start:.3f}s")  # Should be ~0.01s
```

### 5.4 GPTCache (Standalone Semantic Cache)

GPTCache gives you full control over the embedding model, vector store, and eviction policy — decoupled from any specific LLM framework.

```python
# pip install gptcache anthropic

from gptcache import cache
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from anthropic import Anthropic

# Initialise GPTCache with ONNX embeddings and FAISS vector store
onnx = Onnx()
data_manager = get_data_manager(
    CacheBase("sqlite"),       # persistent metadata
    VectorBase("faiss", dimension=onnx.dimension),  # vector index
)
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)

anthropic_client = Anthropic()


def ask(question: str) -> str:
    # Check GPTCache first
    cached = get(question)
    if cached:
        print("[GPTCACHE HIT]")
        return cached

    # Call the LLM on a miss
    print("[GPTCACHE MISS]")
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )
    answer = response.content[0].text
    put(question, answer)
    return answer


print(ask("What is gradient descent?"))
print(ask("Can you explain gradient descent?"))  # Should hit cache
```

### 5.5 Anthropic Prompt Caching (Provider-Side)

Mark stable portions of your system prompt for server-side caching. This reduces both cost and latency for every subsequent call that shares the same prefix.

```python
# pip install anthropic

import anthropic

client = anthropic.Anthropic()

# A large, stable system prompt — this is the portion we want to cache.
# In production this might be a full RAG knowledge base, tool definitions,
# or a detailed policy document.
SYSTEM_PROMPT = """
You are an expert Python developer and AI engineer. You have deep knowledge of:
- LLM application architecture
- Retrieval-augmented generation (RAG)
- Caching strategies for AI workloads
- Python frameworks: LangChain, LlamaIndex, FastAPI, Pydantic
- Vector databases: Chroma, Weaviate, Pinecone, Milvus
- Deployment patterns: containerisation, Kubernetes, serverless

Always provide working Python code examples. Be concise but thorough.
Cite trade-offs when comparing approaches.
""" * 20  # Artificially expanded to exceed the 1024-token minimum for demonstration


def ask_with_prompt_cache(user_question: str) -> str:
    """
    The system prompt is sent with cache_control={'type': 'ephemeral'}.
    Anthropic stores the computed KV states for this prefix on its servers.
    Subsequent calls reuse those states, reducing both input token cost and latency.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Mark for caching
            }
        ],
        messages=[{"role": "user", "content": user_question}],
    )

    # Inspect cache usage in response metadata
    usage = response.usage
    print(f"Input tokens:          {usage.input_tokens}")
    print(f"Cache creation tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read tokens:     {getattr(usage, 'cache_read_input_tokens', 0)}")

    return response.content[0].text


# First call: cache is populated (cache_creation_input_tokens will be non-zero)
print("=== Call 1 ===")
print(ask_with_prompt_cache("What caching strategy should I use for a customer support bot?"))

# Second call: cache is read (cache_read_input_tokens will be non-zero, cost is lower)
print("\n=== Call 2 ===")
print(ask_with_prompt_cache("How do I implement semantic caching with Redis?"))
```

**Key rules for effective provider-side caching:**

- Keep the cached prefix stable — do not embed timestamps, user IDs, or session state.
- User-specific content belongs in the message array, not the system prompt.
- The cache is model-specific; switching models invalidates it.

---

## 6. Choosing the Right Strategy

```
Is your knowledge static and fits in the context window?
  └─ YES → Consider Cache-Augmented Generation (CAG)
  └─ NO  → Use RAG + caching

Do users send identical queries verbatim?
  └─ YES → Exact-match cache (Redis hash lookup) is sufficient and fast
  └─ NO  → Add semantic cache layer (GPTCache / RedisSemanticCache)

Do you have a large, stable system prompt (>1,024 tokens)?
  └─ YES → Enable provider-side prompt caching (Anthropic cache_control / OpenAI auto)
  └─ NO  → Not much benefit from prefix caching

Do you operate self-hosted models?
  └─ YES → Tune KV-cache settings in vLLM; consider KV-cache aware routing (llm-d)
  └─ NO  → Focus on application-layer and provider-side caching

Do you need multi-tenant or privacy-sensitive caching?
  └─ YES → Isolate cache partitions by data classification; use stricter similarity thresholds for sensitive data
  └─ NO  → Shared cache with moderate threshold is fine
```

---

## 7. Security Considerations

Caching in multi-tenant AI systems introduces risks that do not exist in single-user deployments.

**Timing attacks:** Response time differences between cache hits and misses can allow an attacker to infer which topics an organisation has recently queried — extracting competitive intelligence through pattern analysis alone, without ever reading cached content.

**Embedding inversion:** Vector embeddings stored in a shared cache contain latent representations of query patterns and domain expertise. Adversarial embedding inversion techniques can reconstruct original queries from these embeddings.

**Mitigation strategies:**

- Partition caches by tenant or data classification level.
- Use strict similarity thresholds (cosine ≥ 0.95) for sensitive data partitions.
- Apply access controls and audit logging to all cache read/write operations.
- Avoid caching personally identifiable information, financial data, or regulated content in shared partitions.
- Conduct regular cache security audits, especially after architectural changes.

---

## 8. Summary

Caching is not a single technique — it is a family of complementary strategies that operate at different layers of an AI application stack.

|Layer|Strategy|Primary Benefit|Best Tool (Python)|
|---|---|---|---|
|Application|Exact-match (hash) cache|Zero-latency identical hits|Redis, `functools.lru_cache`|
|Application|Semantic cache|Handles paraphrases|GPTCache, LangChain + RedisVL|
|Application|CAG (pre-loaded context)|Eliminates RAG overhead|Direct context injection|
|Provider API|Prompt / prefix cache|Reduces cost + TTFT|Anthropic `cache_control`|
|Serving infrastructure|KV-cache|GPU memory reuse|vLLM, TensorRT-LLM|
|Serving infrastructure|KV-cache aware routing|Cross-pod reuse|llm-d (Kubernetes)|

For most Python developers building LLM-powered applications, the highest-leverage starting points are:

1. **Anthropic/OpenAI prompt caching** — free to implement, immediate cost reduction on stable system prompts.
2. **Redis exact-match cache** — straightforward, integrates directly with LangChain in one line.
3. **Redis semantic cache** — adds meaningful cache hit rates for natural-language workloads with minimal extra code.

Add complexity (GPTCache, Bifrost, KV-cache routing) only when simpler strategies have been exhausted and profiling confirms the bottleneck.

---

_Report generated April 2026. Tool versions: anthropic SDK, langchain-redis 0.2.x, GPTCache (Zilliz), Redis Stack._