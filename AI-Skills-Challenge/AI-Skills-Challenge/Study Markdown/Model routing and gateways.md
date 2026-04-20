# Model routing and gateways

## 1. What Is Model Routing?

**Model routing** is the practice of intelligently selecting which LLM (or which instance of an LLM) should handle a given request, rather than hardcoding a single model.

Think of it like a traffic director at an intersection. Instead of every car going to the same road, the director evaluates where each car came from, how heavy traffic is, and which destination is fastest — then sends it the right way.

In practice, a router evaluates a request and selects a model based on:

- **Complexity** — Is this a simple FAQ or a multi-step reasoning task?
- **Cost** — Can a cheaper, smaller model handle this adequately?
- **Latency** — Which model is fastest right now?
- **Availability** — Is the primary provider up? Any rate limits hit?
- **Task type** — Code generation, summarization, and RAG may each benefit from different models.

---

## 2. What Is an LLM Gateway?

An **LLM gateway** is a routing and control layer that sits between your application and model providers. It normalizes different provider APIs (OpenAI, Anthropic, Google, etc.) behind a single unified interface, and adds production-grade capabilities on top.

```
Your Application
      │
      ▼
┌─────────────────────────────────────┐
│           LLM Gateway               │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Routing  │  │ Observability    │ │
│  │ Logic    │  │ (logs, traces)   │ │
│  └──────────┘  └──────────────────┘ │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Caching  │  │ Auth / RBAC      │ │
│  └──────────┘  └──────────────────┘ │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Failover │  │ Cost Controls    │ │
│  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  OpenAI  │  Anthropic  │  Gemini  │  Azure  │  Ollama │
└──────────────────────────────────────────────────────┘
```

A gateway does more than route — it also:

- **Normalizes APIs** so you write code once and switch providers without rewriting
- **Handles failover** automatically when a provider is down or rate-limited
- **Centralizes secrets** so individual services never hold raw API keys
- **Tracks cost** per team, feature, or customer
- **Caches responses** semantically to avoid redundant API calls
- **Enforces policies** like PII filtering, content guardrails, and budget caps

---

## 3. Why You Need Both

Routing and gateways are complementary, not competing:

|Concern|Routing|Gateway|
|---|---|---|
|Which model fits this task?|✅|—|
|Failover when a provider is down|Partial|✅|
|Cost tracking across providers|—|✅|
|Unified API interface|Partial|✅|
|A/B testing models|✅|✅|
|Semantic caching|—|✅|
|PII scrubbing / guardrails|—|✅|

In production, you typically use **routing logic inside a gateway** — the gateway handles infrastructure concerns while routing handles intelligent model selection.

---

## 4. Routing Strategies

### 4.1 Round-Robin / Simple Shuffle

Distributes requests across a pool of deployments with no preference. Best when models are equivalent and you just want to spread load or avoid rate limits.

### 4.2 Latency-Based Routing

Measures response time from each model/provider and prefers the currently fastest one. Good for real-time applications where user experience is sensitive to delays.

### 4.3 Usage-Based / Least-Busy Routing

Tracks tokens-per-minute (TPM) or requests-per-minute (RPM) and routes to the least-utilized deployment. Prevents hot-spotting but adds Redis overhead — not recommended for very high RPS.

### 4.4 Cost-Based Routing

Calculates the expected cost of a request (based on token count estimates) and routes to the cheapest viable model. Highly effective for RAG or classification pipelines where you don't always need GPT-4 level quality.

### 4.5 Complexity-Based / Semantic Routing

Uses a small, fast classification model (or embeddings) to assess task complexity or type, then sends the request to an appropriate model tier. This is the most powerful strategy — studies show it can achieve 90% of the quality of always-routing-to-a-strong-model at around 10% of the cost.

### 4.6 Priority / Fallback Ordering

Assigns explicit priority ranks to deployments. Order=1 is always tried first; failures cascade down to order=2, order=3, etc. Essential for SLA-critical systems.

---

## 5. When to Use What

|Scenario|Recommended Approach|
|---|---|
|Prototyping, single provider|Direct API calls — no gateway needed yet|
|Multi-provider access, early production|LiteLLM Python SDK with simple-shuffle|
|High traffic, latency-sensitive|High-performance gateway (Bifrost, Helicone)|
|Cost optimization on mixed-complexity tasks|Complexity-based routing (RouteLLM, LiteLLM semantic)|
|Enterprise: audit trails, RBAC, budgets|Full LLM gateway (Portkey, Kong, LiteLLM Proxy)|
|Already on Cloudflare/Vercel infra|Native gateway for that platform|
|Agentic / multi-step pipelines|Gateway + task-aware routing per agent node|
|On-premises / data residency requirements|Self-hosted gateway (Bifrost, LiteLLM Proxy)|

---

## 6. Python Examples

### 6.1 Basic Multi-Provider Routing with LiteLLM

The simplest use case: one unified interface across providers, with automatic failover.

```python
# pip install litellm
import os
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

def call_llm(prompt: str, provider: str = "openai") -> str:
    """Route to a specific provider using a single interface."""
    model_map = {
        "openai": "openai/gpt-4o-mini",
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "gemini": "gemini/gemini-1.5-flash",
    }
    model = model_map.get(provider, "openai/gpt-4o-mini")
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Same code, different providers
print(call_llm("Summarize quantum entanglement in one sentence.", "openai"))
print(call_llm("Summarize quantum entanglement in one sentence.", "anthropic"))
```

---

### 6.2 Complexity-Based Routing (Strong vs. Weak Model)

Route simple tasks to a cheap model and complex tasks to a powerful one.

```python
# pip install litellm
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "fast-model",          # Logical name your code uses
            "litellm_params": {
                "model": "openai/gpt-4o-mini",    # Actual model
                "api_key": "your-openai-key",
            },
        },
        {
            "model_name": "powerful-model",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-6",
                "api_key": "your-anthropic-key",
            },
        },
    ],
    fallbacks=[{"fast-model": ["powerful-model"]}],  # Fallback chain
    num_retries=2,
)


def estimate_complexity(prompt: str) -> str:
    """
    Simple heuristic router: score prompt complexity and pick a model tier.
    In production, replace with an embedding similarity classifier or
    a trained routing model (e.g., RouteLLM's matrix factorization router).
    """
    complexity_signals = [
        "explain", "analyze", "compare", "debate", "critique",
        "write code", "implement", "design", "architecture",
        "multi-step", "reasoning", "proof",
    ]
    simple_signals = [
        "what is", "define", "list", "summarize briefly",
        "translate", "yes or no", "when was",
    ]

    prompt_lower = prompt.lower()
    complexity_score = sum(1 for s in complexity_signals if s in prompt_lower)
    simplicity_score = sum(1 for s in simple_signals if s in prompt_lower)

    # Also factor in prompt length
    if len(prompt) > 500:
        complexity_score += 2

    if complexity_score > simplicity_score:
        return "powerful-model"
    return "fast-model"


def smart_completion(prompt: str) -> dict:
    model = estimate_complexity(prompt)
    print(f"[Router] Selected: {model}")
    response = router.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "model_used": model,
        "content": response.choices[0].message.content,
    }


# Simple query → fast model
result = smart_completion("What is the capital of France?")
print(result)

# Complex query → powerful model
result = smart_completion(
    "Analyze and compare the architectural tradeoffs between "
    "transformer-based and state-space models for long-context tasks."
)
print(result)
```

---

### 6.3 Latency-Based Routing with Fallback

Automatically prefer the fastest responding deployment, with cascading fallback.

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "your-openai-key",
            },
            "model_info": {"order": 1},  # Try first
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o-deployment",
                "api_base": "https://your-endpoint.openai.azure.com/",
                "api_key": "your-azure-key",
                "api_version": "2024-02-15-preview",
            },
            "model_info": {"order": 2},  # Try second if order=1 fails
        },
    ],
    routing_strategy="latency-based-routing",
    num_retries=3,
    timeout=30,
    fallbacks=[{"gpt-4o": ["claude-fallback"]}],  # Cross-provider fallback
)

# Extend model_list with the fallback target
router.model_list.append({
    "model_name": "claude-fallback",
    "litellm_params": {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "api_key": "your-anthropic-key",
    },
})

response = router.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello, which model am I talking to?"}],
)
print(response.choices[0].message.content)
```

---

### 6.4 Task-Type Semantic Routing

Route based on what the user is trying to accomplish — different models for different capabilities.

```python
from dataclasses import dataclass
from enum import Enum
import litellm

litellm.set_verbose = False


class TaskType(Enum):
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


@dataclass
class ModelConfig:
    model: str
    provider_key_env: str
    description: str


TASK_MODEL_MAP: dict[TaskType, ModelConfig] = {
    TaskType.CODE: ModelConfig(
        model="openai/gpt-4o",
        provider_key_env="OPENAI_API_KEY",
        description="Best for code generation and debugging",
    ),
    TaskType.REASONING: ModelConfig(
        model="anthropic/claude-sonnet-4-6",
        provider_key_env="ANTHROPIC_API_KEY",
        description="Best for multi-step reasoning and analysis",
    ),
    TaskType.CREATIVE: ModelConfig(
        model="anthropic/claude-sonnet-4-6",
        provider_key_env="ANTHROPIC_API_KEY",
        description="Best for creative writing and brainstorming",
    ),
    TaskType.SUMMARIZATION: ModelConfig(
        model="openai/gpt-4o-mini",
        provider_key_env="OPENAI_API_KEY",
        description="Fast and cheap for summarization",
    ),
    TaskType.GENERAL: ModelConfig(
        model="openai/gpt-4o-mini",
        provider_key_env="OPENAI_API_KEY",
        description="Default for general queries",
    ),
}


def classify_task(prompt: str) -> TaskType:
    """Classify the task type from the prompt using keyword heuristics."""
    p = prompt.lower()
    if any(k in p for k in ["write code", "implement", "function", "debug", "class", "script"]):
        return TaskType.CODE
    if any(k in p for k in ["analyze", "compare", "reason", "evaluate", "pros and cons", "tradeoffs"]):
        return TaskType.REASONING
    if any(k in p for k in ["write a story", "poem", "creative", "imagine", "brainstorm"]):
        return TaskType.CREATIVE
    if any(k in p for k in ["summarize", "tldr", "brief summary", "condense"]):
        return TaskType.SUMMARIZATION
    return TaskType.GENERAL


def route_and_complete(prompt: str, system: str = "") -> dict:
    task = classify_task(prompt)
    config = TASK_MODEL_MAP[task]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = litellm.completion(model=config.model, messages=messages)

    return {
        "task_type": task.value,
        "model_used": config.model,
        "reason": config.description,
        "content": response.choices[0].message.content,
    }


# Examples
print(route_and_complete("Write a Python function to parse JSON from a REST API response."))
print(route_and_complete("Summarize the key points of transformer architecture."))
print(route_and_complete("Write a short poem about distributed systems."))
```

---

### 6.5 Cost-Aware Routing with Budget Controls

Track per-request cost and enforce a budget ceiling before spending escalates.

```python
import litellm
from litellm import completion, completion_cost

# Track spend across the session
session_spend = 0.0
BUDGET_LIMIT_USD = 1.00  # $1 cap for this session


def budget_aware_completion(
    prompt: str,
    preferred_model: str = "openai/gpt-4o",
    fallback_model: str = "openai/gpt-4o-mini",
) -> dict:
    global session_spend

    if session_spend >= BUDGET_LIMIT_USD:
        raise RuntimeError(
            f"Budget limit of ${BUDGET_LIMIT_USD:.2f} reached. "
            f"Current spend: ${session_spend:.4f}"
        )

    # If we're at 80% of budget, downgrade to the cheaper model
    model = fallback_model if session_spend > BUDGET_LIMIT_USD * 0.8 else preferred_model
    messages = [{"role": "user", "content": prompt}]

    response = completion(model=model, messages=messages)
    cost = completion_cost(completion_response=response)
    session_spend += cost

    return {
        "model_used": model,
        "cost_usd": round(cost, 6),
        "session_total_usd": round(session_spend, 6),
        "budget_remaining_usd": round(BUDGET_LIMIT_USD - session_spend, 6),
        "content": response.choices[0].message.content,
    }


result = budget_aware_completion("Explain gradient descent in three sentences.")
print(f"Model: {result['model_used']}")
print(f"Cost: ${result['cost_usd']}")
print(f"Session total: ${result['session_total_usd']}")
print(f"Content: {result['content']}")
```

---

### 6.6 Building a Custom Gateway Wrapper

A reusable gateway class that handles routing, retries, logging, and cost tracking in one place.

```python
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import litellm
from litellm import completion, completion_cost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_gateway")


@dataclass
class GatewayConfig:
    default_model: str = "openai/gpt-4o-mini"
    fallback_model: str = "anthropic/claude-haiku-4-5-20251001"
    max_retries: int = 3
    timeout_seconds: int = 30
    budget_limit_usd: float = 10.0
    enable_cost_tracking: bool = True
    log_requests: bool = True


@dataclass
class GatewayStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_requests: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    cost_by_model: dict = field(default_factory=dict)


class LLMGateway:
    """
    A minimal production-grade LLM gateway with routing, retries,
    fallback, cost tracking, and structured logging.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.stats = GatewayStats()

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> dict:
        model = model or self.config.default_model

        if system:
            messages = [{"role": "system", "content": system}] + messages

        self._check_budget()

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._execute(messages, model, attempt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed on {model}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # Primary model exhausted — try fallback
        logger.warning(f"Falling back to {self.config.fallback_model}")
        self.stats.fallback_requests += 1
        try:
            return self._execute(messages, self.config.fallback_model, attempt=0, **kwargs)
        except Exception as e:
            self.stats.failed_requests += 1
            raise RuntimeError(f"All models failed. Last error: {last_error}") from e

    def _execute(self, messages: list[dict], model: str, attempt: int, **kwargs) -> dict:
        start = time.monotonic()

        response = completion(
            model=model,
            messages=messages,
            timeout=self.config.timeout_seconds,
            **kwargs,
        )

        latency_ms = (time.monotonic() - start) * 1000
        cost = completion_cost(completion_response=response) if self.config.enable_cost_tracking else 0.0

        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.total_cost_usd += cost
        self.stats.total_latency_ms += latency_ms
        self.stats.cost_by_model[model] = self.stats.cost_by_model.get(model, 0.0) + cost

        if self.config.log_requests:
            logger.info(
                f"model={model} attempt={attempt} "
                f"latency={latency_ms:.1f}ms cost=${cost:.6f}"
            )

        return {
            "content": response.choices[0].message.content,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
            "usage": dict(response.usage),
        }

    def _check_budget(self):
        if self.stats.total_cost_usd >= self.config.budget_limit_usd:
            raise RuntimeError(
                f"Budget exhausted: ${self.stats.total_cost_usd:.4f} "
                f">= limit ${self.config.budget_limit_usd}"
            )

    def get_stats(self) -> dict:
        avg_latency = (
            self.stats.total_latency_ms / self.stats.successful_requests
            if self.stats.successful_requests > 0 else 0
        )
        return {
            "total_requests": self.stats.total_requests,
            "success_rate": (
                self.stats.successful_requests / self.stats.total_requests
                if self.stats.total_requests > 0 else 0
            ),
            "fallback_rate": (
                self.stats.fallback_requests / self.stats.total_requests
                if self.stats.total_requests > 0 else 0
            ),
            "total_cost_usd": round(self.stats.total_cost_usd, 6),
            "avg_latency_ms": round(avg_latency, 1),
            "cost_by_model": {k: round(v, 6) for k, v in self.stats.cost_by_model.items()},
        }


# Usage
gateway = LLMGateway(GatewayConfig(budget_limit_usd=5.0, log_requests=True))

result = gateway.complete(
    messages=[{"role": "user", "content": "What is the difference between RAG and fine-tuning?"}],
    system="You are a concise AI engineer assistant.",
)
print(result["content"])
print("\nGateway Stats:", gateway.get_stats())
```

---

## 7. Gateway Tool Comparison (2025)

|Tool|Language|Best For|Open Source|Latency Overhead|
|---|---|---|---|---|
|**LiteLLM**|Python|Python teams, prototyping, 100+ providers|✅|~3–40ms|
|**Bifrost**|Go|Production at scale, lowest latency|✅ (Apache 2.0)|~0.011ms|
|**Helicone**|Rust|Observability-first teams|✅|~50ms|
|**OpenRouter**|Managed|Simplest multi-model access, prototyping|❌|~40ms|
|**Portkey**|Managed|Enterprise controls, 100+ models|Partial|~50ms|
|**Kong AI Gateway**|Go|Orgs already on Kong API Gateway|Partial|Low|
|**Cloudflare AI Gateway**|Managed|Cloudflare infrastructure users|❌|Edge-optimized|

**For Python-native engineers:** LiteLLM is the natural starting point. Its SDK integrates directly into existing Python codebases, supports 100+ providers, and provides a proxy server mode when you need centralized control. For scale beyond ~500 RPS, consider Bifrost or Helicone.

---

## 8. Architecture Patterns

### Pattern 1: Direct SDK Routing (Development / Small Scale)

```
App Code → LiteLLM Python SDK → Provider APIs
```

Best for: prototyping, internal tools, small teams. No separate infrastructure.

### Pattern 2: Sidecar Gateway (Microservices)

```
Service A ──┐
Service B ──┼──► LiteLLM Proxy / Bifrost ──► Provider APIs
Service C ──┘
```

Best for: multiple services sharing one gateway. Centralized keys, cost tracking, and logging.

### Pattern 3: Gateway + Observability Platform

```
App → Gateway (Bifrost/Helicone) → Provider APIs
                  │
                  ▼
         Observability (Langfuse / Maxim / Datadog)
```

Best for: production systems where model quality, cost trends, and latency need monitoring.

### Pattern 4: Intelligent Multi-Tier Routing

```
Request → Task Classifier → [simple | complex | code | creative]
                                  │
               ┌──────────────────┼────────────────────┐
               ▼                  ▼                     ▼
          Fast/Cheap          Mid-tier             Powerful/Expensive
       (gpt-4o-mini)      (claude-haiku)          (claude-sonnet)
```

Best for: cost-optimized pipelines with mixed workloads.

---

## 9. Decision Guide: Choosing the Right Approach

```
Are you in production with real users?
  ├── No  → Use LiteLLM SDK directly; add routing logic as needed
  └── Yes →
        Do you need multi-team access, RBAC, or audit trails?
          ├── Yes → Deploy a full gateway (LiteLLM Proxy, Portkey, Bifrost)
          └── No  →
                Do you need < 1ms gateway overhead?
                  ├── Yes → Use Bifrost or Helicone
                  └── No  → LiteLLM Proxy is sufficient

Are you optimizing for cost?
  ├── Yes → Implement complexity-based routing (weak/strong model pairs)
  └── No  → Use simple-shuffle or latency-based routing for reliability

Are you building agentic / multi-step pipelines?
  └── Consider task-aware routing per agent node + gateway for failover
```

---

## 10. Key Takeaways

**Model routing and LLM gateways are now essential infrastructure** for any serious AI engineering work. As enterprise LLM spend has surpassed $8.4 billion, the teams that treat this layer as a first-class concern build more reliable, cheaper, and more maintainable systems.

The core principles to internalize:

1. **Never hardcode a single provider.** Provider outages, model deprecations, and pricing changes happen. Abstraction costs almost nothing; tight coupling costs a lot.
    
2. **Start with LiteLLM for Python work.** It covers the majority of use cases with minimal setup and integrates naturally with your existing code.
    
3. **Use complexity-based routing to cut costs.** Studies consistently show 30–85% cost reduction while preserving 90%+ of output quality by routing simple tasks to smaller models.
    
4. **Add a gateway before you scale.** Retrofitting observability, RBAC, and cost controls after you've shipped is painful. Wire them early.
    
5. **Trace everything from day one.** Without per-request logs of which model was used, what it cost, and how long it took, you are flying blind.
    
6. **Test your fallbacks.** Inject failures. Throttle API keys. Verify that your failover logic actually fires under pressure.
    

---

_Report current as of April 2026. Gateway landscape evolves rapidly — verify benchmarks against your specific workload and provider mix before committing to infrastructure choices._