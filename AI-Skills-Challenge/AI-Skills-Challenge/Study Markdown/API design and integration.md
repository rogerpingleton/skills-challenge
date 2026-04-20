# API design and integration

APIs (Application Programming Interfaces) are the connective tissue of modern AI systems. As an AI Engineer, you'll constantly be consuming external APIs (LLMs, vector DBs, data sources) and designing your own (to expose AI capabilities to other services). Here's what you need to know.

---

## 1. Core API Design Principles

### REST vs. Other Paradigms

**REST** is the dominant style. It's stateless, uses HTTP verbs, and organizes resources around URLs.

|Verb|Purpose|Example|
|---|---|---|
|`GET`|Retrieve|`GET /models`|
|`POST`|Create / invoke|`POST /completions`|
|`PUT`|Replace|`PUT /agents/{id}`|
|`PATCH`|Partial update|`PATCH /agents/{id}`|
|`DELETE`|Remove|`DELETE /sessions/{id}`|

You'll also encounter:

- **GraphQL** — client-specified queries; useful when consumers need flexible data shapes
- **gRPC** — binary, high-performance; common in ML inference pipelines
- **WebSockets** — persistent connections for streaming responses (e.g., token streaming from an LLM)

### Resource Naming

Resources should be **nouns, not verbs**, and **plural**.

```
# Good
POST /v1/completions
GET  /v1/embeddings
GET  /v1/agents/{agent_id}/sessions

# Bad
POST /runCompletion
GET  /getEmbedding
```

### Versioning

Always version your APIs. The standard approach is a path prefix:

```
/v1/completions
/v2/completions   ← breaking changes go here, v1 still works
```

---

## 2. Request & Response Design

### Standard Response Envelope

Consistency matters enormously when consumers are building on your API.

```python
# A consistent response structure
{
    "status": "success",        # or "error"
    "data": { ... },            # the actual payload
    "metadata": {
        "request_id": "req_abc123",
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 85,
            "total_tokens": 205
        }
    }
}
```

### Error Responses

Use HTTP status codes correctly _and_ return structured error bodies:

```python
# 4xx = client error, 5xx = server error
{
    "status": "error",
    "error": {
        "code": "rate_limit_exceeded",
        "message": "You have exceeded 60 requests/min.",
        "retry_after": 30
    }
}
```

Common codes you'll use in AI APIs:

|Code|Meaning|
|---|---|
|`200`|OK|
|`201`|Created|
|`400`|Bad request (malformed input)|
|`401`|Unauthorized (bad API key)|
|`422`|Unprocessable entity (valid JSON, bad semantics)|
|`429`|Rate limit exceeded|
|`500`|Server error|
|`503`|Model/service unavailable|

---

## 3. Authentication & Security

### API Keys (most common in AI services)

```python
import httpx

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

**Never hardcode keys.** Load them from environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Other Auth Patterns

- **OAuth 2.0** — for user-delegated access (e.g., a user grants your AI app access to their Google Drive)
- **JWT (JSON Web Tokens)** — stateless auth for internal services
- **mTLS** — mutual TLS for service-to-service in high-security environments

---

## 4. Consuming External AI APIs

This is where you'll spend most of your time. Key practices:

### Retry Logic with Exponential Backoff

LLM APIs are often rate-limited or intermittently slow. Don't just retry immediately.

```python
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60)
)
def call_llm_api(payload: dict) -> dict:
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30.0
    )
    response.raise_for_status()
    return response.json()
```

### Streaming Responses

For chat and completions, **streaming is essential** for UX — you don't want to block waiting for the full response.

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain transformers."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Timeouts and Circuit Breakers

Always set timeouts. For longer inference tasks, consider a circuit breaker pattern.

```python
# Using httpx with explicit timeouts
client = httpx.Client(
    timeout=httpx.Timeout(
        connect=5.0,    # connection timeout
        read=60.0,      # read timeout (LLMs can be slow)
        write=10.0,
        pool=5.0
    )
)
```

---

## 5. Building Your Own AI APIs

When you expose an AI capability as a service (e.g., a RAG endpoint, an agent API), you'll typically use **FastAPI** in Python — it's become the standard for AI microservices.

### A Minimal RAG API Endpoint

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

app = FastAPI(title="RAG API", version="1.0")

# --- Request/Response Models (always use Pydantic) ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[dict] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    request_id: str
    usage: dict

# --- Endpoint ---

@app.post("/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # 1. Embed the query
        embedding = await get_embedding(request.query)

        # 2. Retrieve from vector DB
        docs = await vector_db.search(
            embedding,
            top_k=request.top_k,
            filter=request.filter_metadata
        )

        # 3. Generate answer
        answer, usage = await generate_answer(request.query, docs)

        return QueryResponse(
            answer=answer,
            sources=[d.metadata for d in docs],
            request_id=generate_request_id(),
            usage=usage
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal inference error")
```

### Input Validation

FastAPI + Pydantic handle this elegantly. Validate aggressively at the boundary — bad data going into an LLM call wastes money and produces garbage output.

```python
from pydantic import BaseModel, Field, validator

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: str = Field(default="claude-sonnet-4-6")

    @validator("model")
    def validate_model(cls, v):
        allowed = {"claude-sonnet-4-6", "claude-opus-4-6"}
        if v not in allowed:
            raise ValueError(f"Model must be one of {allowed}")
        return v
```

---

## 6. Rate Limiting & Pagination

### Rate Limiting Your Own API

Use a library like `slowapi` (FastAPI) to protect your service:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/completions")
@limiter.limit("20/minute")
async def create_completion(request: Request, body: CompletionRequest):
    ...
```

### Pagination for Listing Endpoints

For any endpoint returning lists (e.g., conversation history, agent runs), use **cursor-based pagination** over offset pagination — it's more stable with live data.

```python
class PaginatedResponse(BaseModel):
    data: list[dict]
    next_cursor: Optional[str]   # pass this back in next request
    has_more: bool

# GET /v1/sessions?cursor=abc123&limit=20
```

---

## 7. Async-First Design

AI operations are I/O-heavy (network calls to LLM providers, vector DBs, etc.). Use `async`/`await` throughout.

```python
import asyncio
import anthropic

async def run_parallel_completions(prompts: list[str]) -> list[str]:
    """Fan out multiple LLM calls concurrently."""
    client = anthropic.AsyncAnthropic()

    async def single_call(prompt: str) -> str:
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    results = await asyncio.gather(*[single_call(p) for p in prompts])
    return results
```

---

## 8. Observability

You can't improve what you can't measure. Every AI API call should be logged with enough context to debug failures.

```python
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def trace_llm_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        request_id = generate_request_id()

        logger.info({
            "event": "llm_call_start",
            "request_id": request_id,
            "model": kwargs.get("model"),
            "prompt_length": len(str(kwargs.get("messages", "")))
        })

        try:
            result = await func(*args, **kwargs)
            logger.info({
                "event": "llm_call_success",
                "request_id": request_id,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "tokens_used": result.usage.total_tokens
            })
            return result
        except Exception as e:
            logger.error({
                "event": "llm_call_error",
                "request_id": request_id,
                "error": str(e),
                "latency_ms": (time.perf_counter() - start) * 1000
            })
            raise
    return wrapper
```

---

## Quick Reference Summary

|Topic|Key Takeaway|
|---|---|
|**REST design**|Nouns not verbs, version with `/v1/`|
|**Auth**|Bearer tokens, never hardcode keys|
|**Error handling**|Structured errors + correct HTTP codes|
|**Retries**|Exponential backoff, especially for LLM APIs|
|**Streaming**|Use it for any user-facing completions|
|**Your own APIs**|FastAPI + Pydantic is the Python standard|
|**Async**|Default to `async`/`await` for all I/O|
|**Observability**|Log request IDs, latency, and token usage|

As an AI Engineer, consuming LLM APIs well (retries, streaming, async) will occupy most of your day-to-day work. Building your own APIs (FastAPI, validation, rate limiting) becomes critical once you're packaging AI capabilities into products or microservices.