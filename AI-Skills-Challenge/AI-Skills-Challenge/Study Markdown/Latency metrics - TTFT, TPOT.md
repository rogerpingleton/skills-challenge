# Latency metrics - TTFT, TPOT

> **Scope:** Inference optimization — understanding, measuring, and tracking latency metrics in production LLM systems.

---

## 1. The Inference Pipeline at a Glance

Before diving into metrics, it helps to understand the two distinct phases of LLM inference:

```
User Request
    │
    ▼
┌─────────────────────────────────┐
│  PREFILL PHASE                  │
│  • Tokenize input               │
│  • Process entire prompt        │
│  • Build KV cache               │
│  • Emit first token             │  ◄── TTFT ends here
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  DECODE PHASE                   │
│  • Auto-regressive generation   │
│  • One token per forward pass   │
│  • KV cache grows each step     │  ◄── TPOT / ITL measured here
└─────────────────────────────────┘
    │
    ▼
Final Token Delivered              ◄── E2E Latency ends here
```

The two phases have fundamentally different performance characteristics and bottlenecks — prefill is **compute-bound**, decode is typically **memory-bandwidth-bound**.

---

## 2. Core Latency Metrics

### 2.1 Time to First Token (TTFT)

**Definition:** The elapsed time from when the user submits a request to when the very first output token is received.

```
TTFT = T(first_token_received) − T(request_sent)
```

TTFT encompasses:

- Network round-trip (client → server)
- Request queuing time (waiting in batch queue)
- Prefill computation (processing the full input prompt)
- KV cache construction

**Why it matters:** TTFT is the primary driver of _perceived_ responsiveness. In a streaming UI, this is how long the screen stays blank. For a chatbot, anything above ~500ms starts to feel sluggish; for a code completion tool, sub-100ms is expected.

**What drives it up:**

- Longer input prompts (prefill cost scales with prompt length)
- High server load / long queue times
- Large model size with insufficient hardware
- Speculative decoding draft overhead

**Typical targets by use case:**

|Use Case|Target TTFT (P99)|
|---|---|
|Code completion|< 100 ms|
|Interactive chatbot|< 500 ms|
|Voice assistant|< 300 ms|
|Batch document jobs|< 6,000 ms|

---

### 2.2 Time Per Output Token (TPOT) / Inter-Token Latency (ITL)

**Definition:** The average time between the generation of consecutive tokens _after_ the first token has been emitted.

```
TPOT = (E2E_latency − TTFT) / (total_output_tokens − 1)
```

> **Tooling note:** Most industry-standard tools (NVIDIA GenAI-Perf, vLLM) _exclude_ TTFT from the TPOT calculation so it measures only the decode phase. Some older tools (LLMPerf, by default) include it — always verify which definition a tool uses.

**Why it matters:** TPOT controls the _streaming feel_ of a response. Human reading speed is roughly 200–250 words per minute, which translates to approximately 4–5 tokens/sec (TPOT ~200–250ms). For a fluent streaming experience you typically want TPOT well under 50ms (≥20 tokens/sec).

**What drives it up:**

- KV cache growth — as output length increases, attention computation grows linearly
- Memory bandwidth saturation
- Suboptimal batching strategies
- GPU memory pressure (evictions, recomputation)

**MLPerf benchmarks (reference targets):**

|Scenario|Model|Target TPOT (P99)|
|---|---|---|
|Interactive|Llama 3.1-8B|≤ 30 ms|
|Server (enterprise)|Llama 2 Chat 70B|≤ 40 ms|
|Large model|Llama 3.1-405B|≤ 175 ms|

---

### 2.3 End-to-End Latency (E2EL)

**Definition:** Total time from request submission to receipt of the final token.

```
E2EL = TTFT + (TPOT × (output_tokens − 1))
```

E2EL is the complete user-perceived wall-clock time. A fast TTFT doesn't save a bad user experience if TPOT is terrible — E2EL captures both.

---

## 3. Additional Metrics You Should Track

The TTFT/TPOT pair is necessary but not sufficient for production observability. A complete picture requires:

### 3.1 Throughput Metrics

**Tokens Per Second (TPS):** Total output tokens generated per second across all concurrent requests. This is your primary capacity metric.

```
TPS = total_output_tokens / benchmark_duration
```

**Requests Per Second (RPS):** How many complete requests the system handles per second. RPS alone is misleading — a system serving 10 token responses looks great in RPS but may be awful at real workloads.

**Per-User TPS:** `output_tokens / E2E_latency` per request. This is the throughput a single user actually experiences, which often diverges significantly from aggregate TPS under load.

### 3.2 Goodput

**Definition:** The fraction of requests that satisfy _all_ defined SLO thresholds simultaneously.

```python
# Example SLO definition
SLO = {"ttft_ms": 500, "tpot_ms": 15, "e2e_latency_ms": 2000}

goodput = sum(
    1 for r in requests
    if r.ttft < SLO["ttft_ms"]
    and r.tpot < SLO["tpot_ms"]
    and r.e2e < SLO["e2e_latency_ms"]
) / len(requests)
```

Goodput is the most honest single number for "are users getting a good experience?" A system can have impressive average latency while still failing 20% of requests.

### 3.3 Percentile Distribution (P50 / P90 / P99)

Never report only the mean. LLM latency distributions are heavy-tailed — a few slow outliers massively inflate averages while median looks fine.

- **P50 (median):** Typical user experience
- **P90:** What most users experience under moderate load
- **P99:** Worst-case SLO boundary; what your slowest 1% of users face

### 3.4 Infrastructure / System Metrics

|Metric|Why It Matters|
|---|---|
|**GPU Memory Utilization**|KV cache evictions cause recomputation; watch for saturation|
|**GPU Compute Utilization**|Prefill is compute-bound; low GPU util = wasted capacity|
|**Request Queue Depth**|Long queues inflate TTFT; indicates under-provisioning|
|**Batch Size**|Too small = GPU underutilized; too large = TTFT spikes|
|**KV Cache Hit Rate**|Higher hit rate = lower prefill cost for repeated prefixes|
|**Token Queue Length**|Decode-phase backpressure indicator|

### 3.5 Error / Quality Signals

- **Error rate:** Timeouts, OOM kills, truncated responses
- **Abort rate:** Requests cancelled mid-stream (often a TTFT symptom)
- **Output token length distribution:** Unexpectedly short = truncation; long = runaway generation

---

## 4. Testing and Benchmarking

### 4.1 Benchmarking Tools

**NVIDIA GenAI-Perf** — best-in-class for NVIDIA hardware; integrates with NIM and TensorRT-LLM.

```bash
genai-perf profile \
  -m meta/llama-3.1-8b-instruct \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --concurrency 50 \
  --num-prompts 1000
```

**vLLM Benchmark Scripts** — the de facto standard for open-source inference servers.

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 10   # requests/sec
```

**LLMPerf** — OpenAI-compatible API benchmarking, useful for cloud endpoints.

```bash
python token_benchmark_ray.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --mean-input-tokens 550 \
  --stddev-input-tokens 150 \
  --mean-output-tokens 150 \
  --stddev-output-tokens 10 \
  --num-concurrent-requests 10 \
  --results-dir "result_outputs"
```

### 4.2 A Python TTFT/TPOT Measurement Harness

For measuring against any OpenAI-compatible endpoint with streaming:

```python
import time
import asyncio
import statistics
from dataclasses import dataclass, field
from openai import AsyncOpenAI

@dataclass
class RequestMetrics:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    output_tokens: int = 0
    token_timestamps: list[float] = field(default_factory=list)

async def measure_single_request(
    client: AsyncOpenAI,
    prompt: str,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> RequestMetrics:
    metrics = RequestMetrics()
    t_start = time.perf_counter()
    first_token = True

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=512,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            t_now = time.perf_counter()
            metrics.token_timestamps.append(t_now)
            metrics.output_tokens += 1
            if first_token:
                metrics.ttft_ms = (t_now - t_start) * 1000
                first_token = False

    metrics.e2e_ms = (time.perf_counter() - t_start) * 1000

    if metrics.output_tokens > 1:
        # TPOT excludes TTFT — measured only over decode phase
        inter_token_gaps = [
            (metrics.token_timestamps[i] - metrics.token_timestamps[i - 1]) * 1000
            for i in range(1, len(metrics.token_timestamps))
        ]
        metrics.tpot_ms = statistics.mean(inter_token_gaps)

    return metrics


async def run_benchmark(
    endpoint: str,
    prompts: list[str],
    concurrency: int = 10,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> dict:
    client = AsyncOpenAI(base_url=endpoint, api_key="placeholder")
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt):
        async with semaphore:
            return await measure_single_request(client, prompt, model)

    results: list[RequestMetrics] = await asyncio.gather(
        *[bounded_request(p) for p in prompts]
    )

    def pct(values, p):
        return sorted(values)[int(len(values) * p / 100)]

    ttfts = [r.ttft_ms for r in results]
    tpots = [r.tpot_ms for r in results if r.tpot_ms > 0]
    e2es  = [r.e2e_ms  for r in results]

    return {
        "n_requests": len(results),
        "ttft": {"p50": pct(ttfts, 50), "p90": pct(ttfts, 90), "p99": pct(ttfts, 99), "mean": statistics.mean(ttfts)},
        "tpot": {"p50": pct(tpots, 50), "p90": pct(tpots, 90), "p99": pct(tpots, 99), "mean": statistics.mean(tpots)},
        "e2e":  {"p50": pct(e2es,  50), "p90": pct(e2es,  90), "p99": pct(e2es,  99), "mean": statistics.mean(e2es)},
        "goodput_500ms_ttft": sum(1 for t in ttfts if t < 500) / len(ttfts),
    }


if __name__ == "__main__":
    prompts = ["Explain the difference between TCP and UDP in simple terms."] * 100
    summary = asyncio.run(run_benchmark("http://localhost:8000/v1", prompts, concurrency=10))
    print(summary)
```

### 4.3 Load Testing Strategy

Run benchmarks across a **concurrency sweep** to find your system's operating envelope:

```python
import asyncio
import pandas as pd
import matplotlib.pyplot as plt

async def concurrency_sweep(endpoint, prompts, levels=[1, 5, 10, 25, 50, 100]):
    rows = []
    for c in levels:
        print(f"Testing concurrency={c} ...")
        result = await run_benchmark(endpoint, prompts[:200], concurrency=c)
        rows.append({
            "concurrency": c,
            "ttft_p99_ms": result["ttft"]["p99"],
            "tpot_p99_ms": result["tpot"]["p99"],
            "e2e_p99_ms":  result["e2e"]["p99"],
            "goodput":     result["goodput_500ms_ttft"],
        })
    return pd.DataFrame(rows)
```

This reveals exactly at what concurrency level your TTFT SLO breaks — a key input for auto-scaling policies.

---

## 5. Observability in Production

### 5.1 The Three Pillars

**Metrics (Prometheus + Grafana):** Numeric time-series for dashboards and alerting.

**Traces (OpenTelemetry):** Distributed traces that show time spent in each stage (queue → prefill → decode → network). Essential for multi-component pipelines (RAG, agents, tool calls).

**Logs (Loki / structured JSON):** Human-readable context — prompts, errors, model parameters.

### 5.2 vLLM's Native Prometheus Endpoint

If you're using vLLM, a `/metrics` endpoint is available out of the box:

```bash
# Start vLLM with metrics enabled
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-metrics
```

Key exported metrics:

|Prometheus Metric|Description|
|---|---|
|`vllm:time_to_first_token_seconds`|TTFT histogram|
|`vllm:time_per_output_token_seconds`|TPOT histogram|
|`vllm:e2e_request_latency_seconds`|End-to-end latency histogram|
|`vllm:gpu_cache_usage_perc`|KV cache utilization|
|`vllm:num_requests_waiting`|Queue depth|
|`vllm:request_success_total`|Successful request counter|
|`vllm:generation_tokens_total`|Total tokens generated|

### 5.3 Prometheus Alert Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: llm_inference_slos
    rules:
      - alert: HighTTFT
        expr: histogram_quantile(0.99, vllm:time_to_first_token_seconds_bucket) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 TTFT has exceeded 500ms"

      - alert: SlowTokenGeneration
        expr: histogram_quantile(0.99, vllm:time_per_output_token_seconds_bucket) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 TPOT has exceeded 50ms"

      - alert: KVCacheNearCapacity
        expr: vllm:gpu_cache_usage_perc > 0.85
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "KV cache above 85% — evictions imminent"

      - alert: RequestQueueBuildup
        expr: vllm:num_requests_waiting > 50
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Request queue depth > 50"
```

### 5.4 OpenTelemetry Tracing with OpenLLMetry

For application-level tracing (vs. server-level Prometheus metrics):

```python
# pip install opentelemetry-sdk traceloop-sdk
from traceloop.sdk import Traceloop
from openai import OpenAI

Traceloop.init(
    app_name="my-llm-service",
    api_endpoint="http://localhost:4318",   # OTel Collector
)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="placeholder")

# All calls are now auto-instrumented with traces
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### 5.5 Recommended Observability Stack

```
┌──────────────────────────────────────────────────────────┐
│  Your Application / vLLM / TensorRT-LLM                  │
│  ├── /metrics  (Prometheus format, TTFT, TPOT, etc.)     │
│  └── OTLP traces/logs → OTel Collector                   │
└──────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
        Prometheus               Jaeger / Tempo
        (metrics store)          (trace store)
              │                      │
              └──────────┬───────────┘
                         ▼
                      Grafana
               (dashboards + alerts)
```

**Managed alternatives:** Arize Phoenix, Langfuse, and Datadog LLM Observability offer pre-built LLM dashboards with minimal setup if you prefer managed tooling over the self-hosted stack.

---

## 6. Metrics Interpretation Reference

|Symptom|Likely Cause|Remediation|
|---|---|---|
|TTFT high, TPOT normal|Long queue time or slow prefill|Scale replicas, reduce batch size, use chunked prefill|
|TPOT high, TTFT normal|KV cache pressure, memory bandwidth limit|Reduce max sequence length, enable quantization|
|Both high under load|System at capacity|Add GPUs, enable tensor parallelism|
|P99 >> P50|Occasional queue spikes or cache evictions|Tune `max_num_seqs`, enable prefix caching|
|TPOT increases with output length|KV cache growth — expected but tunable|Enable sliding window attention, streaming KV offload|
|High abort/error rate|OOM or timeout on long contexts|Reduce `max_model_len`, enable paged attention|

---

## 7. Baseline SLO Templates

Use these as starting points, then tune to your specific use case:

```python
# slo_config.py
SLO_PROFILES = {
    "code_completion": {
        "ttft_p99_ms": 100,
        "tpot_p99_ms": 20,
        "e2e_p99_ms": 2000,
        "min_goodput": 0.99,
    },
    "interactive_chat": {
        "ttft_p99_ms": 500,
        "tpot_p99_ms": 50,
        "e2e_p99_ms": 10000,
        "min_goodput": 0.95,
    },
    "batch_processing": {
        "ttft_p99_ms": 6000,
        "tpot_p99_ms": 200,
        "e2e_p99_ms": 60000,
        "min_goodput": 0.90,
    },
    "voice_assistant": {
        "ttft_p99_ms": 300,
        "tpot_p99_ms": 30,
        "e2e_p99_ms": 5000,
        "min_goodput": 0.99,
    },
}
```

---

## 8. Summary: What to Track and When

|Metric|Track Always|Track in Prod|Benchmark Tool|
|---|:-:|:-:|---|
|TTFT (P50/P99)|✅|✅|All tools|
|TPOT (P50/P99)|✅|✅|All tools|
|E2E Latency|✅|✅|All tools|
|Goodput|✅|✅|Custom / Anyscale|
|TPS / RPS|✅|✅|All tools|
|GPU Cache %|❌|✅|vLLM /metrics|
|Queue Depth|❌|✅|vLLM /metrics|
|KV Cache Hit %|❌|✅|vLLM /metrics|
|Percentile dist.|✅|✅|All tools|
|OTel Traces|❌|✅|OpenLLMetry / Phoenix|

---

_Sources: NVIDIA NIM Benchmarking Docs, BentoML LLM Inference Handbook, MLCommons MLPerf v5.0/5.1, Anyscale Docs, OpenTelemetry LLM Observability Blog, Grafana Labs LLM Observability Guide._