# Monitoring and logging

## A Practical Guide for AI Engineers

## 1. Why This Is Different from Traditional Software

Traditional APM (Application Performance Monitoring) tracks the **Four Golden Signals**: latency, traffic, errors, and saturation. These work beautifully when a "successful" API call means the system did the right thing.

**AI applications break this model.** A `200 OK` response from your LLM tells you inference completed — not whether the output was correct, grounded, safe, or useful. A vector DB might return results in 200ms, but those results may be irrelevant to the user's query.

|Traditional Software|AI Application|
|---|---|
|Deterministic outputs|Probabilistic / non-deterministic outputs|
|Success = correct execution|Success = correct AND useful output|
|Bugs are reproducible|Failures can be one-off or subtle|
|Standard error codes tell you what broke|A 200 response might still be a hallucination|
|Static behavior over time|Model drift can degrade quality silently|

The field that addresses AI-specific observability is called **LLMOps** (an extension of MLOps for large language models), and it requires a separate layer of instrumentation on top of standard infrastructure monitoring.

---

## 2. The Three Pillars: Logs, Metrics, Traces

### Logs

Recorded events at a point in time. In AI systems, logs should capture:

- Prompt inputs and outputs (with PII handling — see Section 9)
- Model version, temperature, and other parameters
- Retrieval results (for RAG systems)
- Agent tool calls and their outputs
- Errors and exceptions

### Metrics

Numerical measurements aggregated over time. Used for dashboards, alerting, and trending (see Section 3).

### Traces

End-to-end execution paths through a system. Critical for multi-step pipelines, RAG, and agent workflows. A trace spans across retrieval → context injection → LLM call → post-processing, showing how time was spent and where failures occurred.

**OpenTelemetry** (OTel) is the emerging standard for collecting all three. The OpenTelemetry GenAI SIG is actively standardizing semantic conventions specifically for LLM telemetry, covering spans, sessions, and agent traces.

---

## 3. Key Metrics to Track

### Performance Metrics (Infrastructure Layer)

|Metric|Description|Why It Matters|
|---|---|---|
|**TTFT** (Time to First Token)|Latency until first token arrives|Perceived responsiveness; high TTFT = queuing/input issues|
|**TPOT** (Time Per Output Token)|Latency per token after first|Streaming smoothness; high TPOT = decoding bottleneck|
|**End-to-End Latency**|Total request time (p50, p95, p99)|Overall UX; p95/p99 reveals tail latency problems|
|**Throughput (RPS / TPS)**|Requests/tokens per second|Capacity planning; TPS is more meaningful than RPS for LLMs|
|**Error Rate**|% of failed/malformed requests|Reliability signal; investigate if > 5%|
|**Token Usage**|Input + output tokens per request|**Directly maps to cost** for API-based LLMs|

### Quality Metrics (AI Layer)

|Metric|Description|Use Case|
|---|---|---|
|**Faithfulness**|Does output derive only from provided context?|RAG hallucination detection|
|**Context Recall**|Did retrieved docs contain the needed info?|RAG retrieval quality|
|**Context Precision**|Signal-to-noise in retrieved chunks|RAG reranking effectiveness|
|**BLEU / ROUGE**|N-gram overlap with reference outputs|Translation, summarization quality|
|**Perplexity**|How "surprised" the model is by outputs|Model fit to data distribution|
|**Bias Score**|Discriminatory treatment across groups|Fairness monitoring|
|**Toxicity Score**|Harmful or offensive content detection|Safety and compliance|

### Business / Operational KPIs

|KPI|Description|
|---|---|
|**Model Drift**|Changes in output distribution over time (can happen when providers silently update weights)|
|**Input Drift**|Changes in user query patterns away from training distribution|
|**Hallucination Rate**|Estimated frequency of factual inaccuracies|
|**User Satisfaction**|Thumbs up/down, CSAT, downstream task success|
|**Cost per Request**|Token cost × model pricing|
|**MTTD / MTTR**|Mean Time to Detect / Resolve incidents|

---

## 4. Logging Architecture & Best Practices

### Log Levels (Use Appropriately)

```
DEBUG   → Detailed trace info for development (prompt internals, intermediate steps)
INFO    → Normal operational events (request received, response sent)
WARNING → Unexpected but recoverable situations (rate limit approaching)
ERROR   → Request-level failures (API timeout, parsing error)
CRITICAL→ System-level failures (service down, data pipeline broken)
```

### Structural Best Practices

**Use structured JSON logging.** This makes logs searchable and parseable by downstream tools like Elasticsearch, Splunk, or Grafana Loki.

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "trace_id": getattr(record, "trace_id", None),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)
```

**Use ISO 8601 timestamps in UTC.** Always. This prevents timezone confusion across distributed systems.

**Implement log rotation and retention.** Keep ~90 days in hot storage for fast access; archive older logs to cold storage. This balances cost against regulatory requirements (GDPR, CCPA, HIPAA).

**Two log tiers:**

- _High-level logs_ — General health overview: request in, response out, latency, status
- _Detailed logs_ — Step-by-step traces for debugging: retrieval results, chain steps, tool calls

---

## 5. Established Frameworks & Tools

### LLM-Specific Observability Platforms

|Tool|Type|Key Strengths|
|---|---|---|
|**Langfuse**|Open source / SaaS|Traces, evals, prompt management, `@observe()` decorator for Python|
|**LangSmith**|SaaS (LangChain)|Deep LangChain/LangGraph integration, prompt playground, A/B testing|
|**Arize Phoenix**|Open source / SaaS|RAG evaluation, embedding drift, LLM-as-a-judge|
|**Opik**|Open source|LLM evaluation, CI/CD integration, golden dataset management|
|**Weights & Biases (W&B)**|SaaS|Strong ML experiment tracking, extending into LLMOps|
|**Helicone**|Open source / SaaS|Lightweight LLM proxy with automatic logging|

### Infrastructure / General Observability

|Tool|Best For|
|---|---|
|**Prometheus + Grafana**|Self-hosted metrics and dashboards; pairs well with `prometheus-client` in Python|
|**OpenTelemetry (OTel)**|Standard for telemetry instrumentation; vendor-neutral|
|**Datadog**|Enterprise APM with growing LLM monitoring capabilities|
|**Grafana Loki**|Log aggregation, especially for teams already on Grafana|
|**Elasticsearch / ELK Stack**|Full-text log search and analytics at scale|

### Python Libraries

```
langfuse          # LLM observability with minimal instrumentation
opentelemetry-sdk # Standard telemetry SDK
prometheus-client # Expose metrics for Prometheus scraping
structlog         # Structured logging for Python
loguru            # Friendlier Python logging with JSON support
```

---

## 6. Methodologies: LLMOps & MLOps

### The LLMOps Lifecycle

```
Develop → Evaluate → Deploy → Monitor → Improve → (repeat)
```

LLMOps extends MLOps with concerns specific to generative AI:

**1. Prompt Management as Code** Treat prompts like source code. Version-control them, track changes with hashes, and run evaluation suites in CI/CD whenever prompts change. A regression in one query type while fixing another is a common failure mode.

**2. Evaluation Pipelines** Maintain "golden datasets" — curated input/output pairs that represent correct behavior. Run LLM-as-a-judge scoring, BLEU/ROUGE, and custom metrics against these on every deploy.

**3. Drift Detection** Two types to monitor:

- **Prompt drift**: Performance degrades over time without prompt changes (e.g., the upstream model provider silently updated weights)
- **Input drift**: User query patterns shift away from training/evaluation distribution, causing the system to encounter cases it wasn't designed for

**4. A/B Testing for Prompts** Comparing prompts is harder than comparing code paths because LLM outputs are non-deterministic and quality is subjective. Use frameworks like LangSmith or Opik that are designed for stochastic comparison.

**5. The Data Flywheel** Capture production traces → filter for failures and negative feedback → curate "hard examples" into golden dataset → improve prompts or fine-tune → deploy → repeat. This is how AI systems improve with production usage.

---

## 7. Python Implementation Examples

### Example 1: Structured Logging with Correlation IDs

```python
import logging
import uuid
import json
import time
from functools import wraps
from contextvars import ContextVar

# Context variable for trace ID propagation
current_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%03dZ"),
            "level": record.levelname,
            "msg": record.getMessage(),
            "trace_id": current_trace_id.get(""),
            "module": record.module,
            **({"exc": self.formatException(record.exc_info)} if record.exc_info else {}),
        })

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

def traced(func):
    """Decorator that assigns a trace ID to each invocation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = current_trace_id.set(str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed in {(time.perf_counter()-start)*1000:.1f}ms")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}", exc_info=True)
            raise
        finally:
            current_trace_id.reset(token)
    return wrapper
```

---

### Example 2: Prometheus Metrics for an LLM Endpoint

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# --- Define Metrics ---
LLM_REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"]  # labels
)

LLM_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM end-to-end request latency",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

LLM_TTFT = Histogram(
    "llm_time_to_first_token_seconds",
    "Time from request to first token",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

TOKEN_USAGE = Counter(
    "llm_tokens_total",
    "Total tokens used",
    ["model", "type"]  # type: input | output
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently in-flight LLM requests"
)

# --- Instrument your LLM call ---
def call_llm(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    status = "success"

    try:
        # Replace with your actual LLM client call
        response = your_llm_client.complete(prompt, model=model)

        TOKEN_USAGE.labels(model=model, type="input").inc(response.usage.input_tokens)
        TOKEN_USAGE.labels(model=model, type="output").inc(response.usage.output_tokens)

        return response
    except Exception as e:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        LLM_LATENCY.labels(model=model).observe(duration)
        LLM_REQUEST_COUNT.labels(model=model, status=status).inc()
        ACTIVE_REQUESTS.dec()

# Start metrics server (scrape at /metrics on port 8001)
start_http_server(8001)
```

---

### Example 3: Langfuse Tracing with the `@observe` Decorator

```python
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import openai  # Drop-in replacement for openai client

@observe()  # Automatically creates a trace
def rag_pipeline(user_query: str) -> str:
    # This span is automatically nested under the parent trace
    context = retrieve_context(user_query)
    answer = generate_answer(user_query, context)
    
    # Attach custom metadata to the trace
    langfuse_context.update_current_trace(
        metadata={"query_type": classify_query(user_query)},
        tags=["rag", "production"],
    )
    return answer

@observe(name="retrieval")
def retrieve_context(query: str) -> list[str]:
    # All LLM calls via langfuse.openai are auto-instrumented
    results = vector_db.search(query, top_k=5)
    langfuse_context.update_current_observation(
        output=results,
        metadata={"num_results": len(results)}
    )
    return results

@observe(name="generation")
def generate_answer(query: str, context: list[str]) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based on context only."},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content
```

---

### Example 4: Model Drift Detection

```python
import numpy as np
from scipy.stats import ks_2samp
from collections import deque

class DriftDetector:
    """
    Simple statistical drift detector using the Kolmogorov-Smirnov test.
    Compares a sliding window of recent response scores against a baseline.
    """
    def __init__(self, baseline_scores: list[float], window_size: int = 200, threshold: float = 0.05):
        self.baseline = np.array(baseline_scores)
        self.window = deque(maxlen=window_size)
        self.threshold = threshold  # p-value threshold

    def record(self, score: float):
        self.window.append(score)

    def check_drift(self) -> dict:
        if len(self.window) < 50:
            return {"drift_detected": False, "reason": "insufficient_data"}

        stat, p_value = ks_2samp(self.baseline, list(self.window))
        drift = p_value < self.threshold

        return {
            "drift_detected": drift,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "window_mean": round(np.mean(self.window), 4),
            "baseline_mean": round(np.mean(self.baseline), 4),
        }

# Usage
baseline = [0.82, 0.79, 0.85, ...]  # Historical faithfulness scores
detector = DriftDetector(baseline_scores=baseline)

# In your production loop:
for response in production_stream:
    score = evaluate_faithfulness(response)
    detector.record(score)
    result = detector.check_drift()
    if result["drift_detected"]:
        alert_team(result)
```

---

## 8. Alerting Strategy

Avoid alert fatigue by starting with a small, high-signal set. Expand only when you understand your system's baselines.

**Tier 1 — Page immediately:**

- End-to-end error rate > 5% for 5 minutes
- Service health endpoint down for 1 minute
- Token spend spike > 3× baseline in 10 minutes (cost runaway / infinite loop)

**Tier 2 — Notify (non-paging):**

- p95 latency > 2× 7-day baseline
- Drift detector fires (quality degradation)
- Input distribution shift detected
- Any single agent recursion depth > 3 (runaway loop indicator)

**Tier 3 — Dashboard only:**

- Daily token usage and cost trending
- Faithfulness / context recall scores
- User satisfaction scores

---

## 9. Privacy, Security & Compliance

AI applications often process sensitive data. This creates specific logging obligations:

**PII Handling.** Never log raw user inputs without a scrubbing pass. Build a redaction pipeline before log emission:

```python
import re

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
}

def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{label.upper()}]", text)
    return text
```

**Regulatory requirements:**

- **GDPR / CCPA** — Mandate traceability in AI decision-making. Insufficient logs can expose organizations to legal penalties.
- **EU AI Act (in effect 2024)** — Mandates continuous monitoring of high-risk AI applications. Logging is no longer optional for regulated uses.
- **OWASP LLM Top 10** — Specifically calls out `LLM07: Sensitive Information Disclosure` as a risk of exposing PII through poorly configured logs.

**Security recommendations:**

- Encrypt logs at rest and in transit
- Implement access controls and audit logging _on the logs themselves_
- Use data masking or differential privacy for monitoring insights without exposing raw data
- Apply the same security standards to your observability tools as to the AI apps they monitor

---

## 10. The Continuous Improvement Loop

Monitoring is not a fire-and-forget setup. The goal is a **virtuous feedback cycle**:

```
Production Traces
       ↓
Filter for failures + negative feedback
       ↓
Curate "hard examples" into golden dataset
       ↓
Improve prompts or fine-tune models
       ↓
Run evaluation suite (CI blocks regression)
       ↓
Deploy improved system
       ↓
Back to Production Traces ↑
```

This "data flywheel" is how AI systems get measurably better over time rather than degrading silently. Every failure is a signal. Without this loop, you're flying blind.

**Key habits for AI Engineers:**

1. Instrument from day one — observability retrofitted to production is painful and incomplete
2. Treat prompts as code — version control, CI evaluation, regression testing
3. Define your "golden dataset" early — you can't measure improvement without a baseline
4. Set cost alerts before latency alerts — token runaway is often your first production incident
5. Review traces weekly — manual spot-checks catch what automated metrics miss

---

## Quick Reference: Tool Decision Tree

```
Need to trace multi-step LLM pipelines?
  → Langfuse, LangSmith, or Arize Phoenix

Need infrastructure metrics (latency, throughput, error rate)?
  → Prometheus + Grafana, or Datadog

Need log aggregation and search?
  → ELK Stack, Grafana Loki, or Splunk

Need RAG-specific evaluation (faithfulness, context recall)?
  → Arize Phoenix, Opik, or Ragas (Python library)

Need standard telemetry instrumentation (vendor-neutral)?
  → OpenTelemetry SDK (opentelemetry-sdk in Python)

Need cost monitoring for API-based LLMs?
  → Helicone (proxy-based), or Langfuse (SDK-based)
```

---

_Last updated: April 2026_