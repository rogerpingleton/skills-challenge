# Batching techniques

See: Batch vs. online inference strategies
## Batching Techniques in AI Inference Optimization

Batching is one of the most impactful techniques for improving inference throughput. The core idea is simple: instead of processing one request at a time, group multiple inputs together so the GPU can work on them in parallel. But the implementation details reveal a rich set of strategies, each with different tradeoffs.

---

### Why Batching Matters

Modern GPUs are massively parallel processors — they're most efficient when doing the same operation across many data points simultaneously. A single inference request leaves most of the GPU idle. Batching "fills" the GPU, amortizing the fixed overhead (kernel launches, memory transfers, attention head computation) across many requests.

The tension is always **throughput vs. latency**: larger batches = higher throughput, but the first request in a batch has to wait for the others to arrive.

---

### 1. Static Batching

The simplest form. You collect a fixed number of requests, process them together, and return all results.

```python
def static_batch_inference(model, requests: list[str], batch_size: int = 8):
    results = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results
```

**Problem:** Requests have variable lengths. Padding shorter sequences to match the longest wastes compute. A batch with one 500-token sequence and seven 10-token sequences burns GPU memory and FLOPs on padding tokens.

---

### 2. Dynamic Batching

Instead of a fixed batch size, requests are queued and batched based on a **time window or token budget**. A batch is dispatched when either enough requests have arrived or a timeout is hit.

```python
import asyncio

class DynamicBatcher:
    def __init__(self, model, max_batch_size=16, max_wait_ms=20):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue()

    async def infer(self, prompt: str) -> str:
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((prompt, future))
        return await future

    async def batch_loop(self):
        while True:
            batch, futures = [], []
            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    prompt, fut = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(prompt)
                    futures.append(fut)
                except asyncio.TimeoutError:
                    break
            if batch:
                results = run_model(self.model, batch)  # your inference call
                for fut, res in zip(futures, results):
                    fut.set_result(res)
```

This is the backbone of serving frameworks like **NVIDIA Triton** and **Ray Serve**.

---

### 3. Continuous Batching (Iteration-Level Batching)

This is the most important modern technique for LLM serving, introduced prominently by the **Orca** paper (2022) and now used in **vLLM**, **TensorRT-LLM**, and **SGLang**.

**The problem with naive batching for LLMs:** Generation is autoregressive — each request produces tokens one at a time. In a static batch, the entire batch is held until the _longest_ sequence finishes. Short sequences are done early but their GPU slots sit idle.

**Continuous batching** solves this by operating at the _iteration level_ (each forward pass) rather than the request level:

- After every token generation step, completed sequences are **evicted** from the batch.
- New waiting requests are **inserted** into the freed slots immediately.
- The batch size stays near-constant and the GPU stays saturated.

```
Time →
Slot 0: [Req A ████████████████ done]
Slot 1: [Req B ████ done][Req D ██████████]   ← D inserted when B finishes
Slot 2: [Req C ██████████████████████████]
```

This can improve GPU utilization by 5–10x over static batching in real serving workloads.

---

### 4. Chunked Prefill

A refinement of continuous batching. The **prefill phase** (processing the prompt) and **decode phase** (generating tokens) have different compute profiles:

- Prefill is compute-bound (many tokens processed at once).
- Decode is memory-bandwidth-bound (one token at a time, reading the KV cache).

Mixing them in the same batch creates interference. **Chunked prefill** splits long prompts into chunks and schedules them across multiple iterations, interleaving prefill and decode work more evenly. This reduces the "prefill bubble" that causes latency spikes. Used in **vLLM v1** and **SGLang**.

---

### 5. Bucket Batching (Length-Based Grouping)

To combat padding waste without full dynamic batching infrastructure, you can sort and group requests by sequence length into buckets:

```python
from itertools import groupby

def bucket_batch_inference(model, tokenizer, prompts, bucket_size=32):
    # Sort by token length
    tokenized = [(p, len(tokenizer.encode(p))) for p in prompts]
    tokenized.sort(key=lambda x: x[1])

    results = []
    for i in range(0, len(tokenized), bucket_size):
        bucket = [p for p, _ in tokenized[i : i + bucket_size]]
        inputs = tokenizer(bucket, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results
```

Padding waste is minimized because sequences within a bucket are similar in length. Simple to implement and very effective for offline/batch workloads.

---

### Tradeoff Summary

|Technique|Throughput|Latency|Complexity|Best For|
|---|---|---|---|---|
|Static Batching|Medium|High|Low|Offline jobs|
|Dynamic Batching|High|Medium|Medium|General serving|
|Continuous Batching|Very High|Low|High|LLM serving|
|Chunked Prefill|Very High|Very Low|High|Long-context LLMs|
|Bucket Batching|Medium-High|Medium|Low|Offline / encoder models|

---

### In Practice

- For **offline batch jobs** (embeddings, classification): bucket batching + static batching is usually enough.
- For **online LLM serving**: use a framework that implements continuous batching out of the box — **vLLM** is the current default choice in Python, with SGLang being a strong alternative for complex workloads.
- For **edge/embedded**: static batching with a fixed batch size tuned to your latency budget.

These techniques compose well with others like **KV cache quantization**, **speculative decoding**, and **tensor parallelism**— batching is typically the first lever to pull before reaching for those.