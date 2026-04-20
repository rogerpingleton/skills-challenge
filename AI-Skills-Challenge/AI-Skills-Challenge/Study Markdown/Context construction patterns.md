# Context construction patterns

## Introduction

Where prompt engineering focused on crafting a single well-worded instruction, **context engineering** is a comprehensive systems discipline that manages everything an LLM encounters during inference — prompts, retrieved documents, memory systems, tool descriptions, conversation state, and more.

As Andrej Karpathy described it: context engineering is the _"delicate art and science of filling the context window with just the right information for the next step."_ Think of the LLM as a CPU and the context window as RAM — your job is to be the operating system that curates what fits.

> **Why it matters:** A 2025 Chroma study of 18 frontier models found that every single one performed worse as input length grew — some dropping from 95% to 60% accuracy past a threshold. More context is **not** always better. How you construct it is everything.

---

## The Four Core Patterns

LangChain's context engineering guide categorises approaches into four buckets: **Write**, **Select**, **Compress**, and **Isolate**. Each addresses a different dimension of the context problem.

---

### Pattern 1: Write — Externalising State

**Writing** context means saving information _outside_ the context window so it can be selectively retrieved later. This combats the finite nature of the context window and enables long-lived agents.

**Sub-patterns:**

- **Scratchpad / working memory** — a temporary store the agent writes intermediate results to
- **Persistent memory** — facts or instructions saved across sessions (episodic, semantic, procedural)
- **Tool-call logs** — recording what an agent did for reflection or audit

```python
import json
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()

# --- Persistent Memory Store ---
MEMORY_FILE = Path("agent_memory.json")

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"facts": [], "preferences": {}}

def save_memory(memory: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))

def write_fact(memory: dict, fact: str) -> None:
    """Write a new fact into persistent memory."""
    memory["facts"].append(fact)
    save_memory(memory)
    print(f"[Memory] Stored: {fact}")

# --- Scratchpad Pattern ---
class AgentScratchpad:
    """Temporary working memory for a single agent run."""

    def __init__(self):
        self._entries: list[dict] = []

    def write(self, step: str, result: str) -> None:
        self._entries.append({"step": step, "result": result})

    def render(self) -> str:
        """Serialise the scratchpad into a string for injection into context."""
        if not self._entries:
            return "No prior steps."
        lines = [f"Step {i+1} — {e['step']}: {e['result']}"
                 for i, e in enumerate(self._entries)]
        return "\n".join(lines)


# Example usage
memory = load_memory()
write_fact(memory, "User prefers Python over JavaScript")
write_fact(memory, "User's project uses FastAPI")

scratchpad = AgentScratchpad()
scratchpad.write("Fetched schema", "Users table has id, name, email columns")
scratchpad.write("Validated input", "Email format is valid")

system_prompt = f"""You are a helpful AI engineer assistant.

## Persistent Memory (facts from past sessions)
{chr(10).join(f'- {f}' for f in memory['facts'])}

## Current Task Scratchpad
{scratchpad.render()}
"""

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=512,
    system=system_prompt,
    messages=[{"role": "user", "content": "How should I structure my user registration endpoint?"}]
)
print(response.content[0].text)
```

---

### Pattern 2: Select — Retrieval-Augmented Generation (RAG)

**Selecting** context means pulling the _right_ information into the window from an external store. Rather than stuffing everything in, you retrieve only what is relevant to the current query.

**Sub-patterns:**

- **Semantic search** (vector databases) — dense retrieval via embeddings
- **Keyword / BM25 search** — sparse retrieval for precise term matching
- **Hybrid retrieval** — combines both for best-of-both-worlds precision

```python
import numpy as np
from anthropic import Anthropic

client = Anthropic()

# --- Minimal in-memory vector store (production: use Chroma, Weaviate, Pinecone) ---

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def embed_text(text: str) -> list[float]:
    """
    Production: replace with a real embedding model call, e.g.:
        openai.embeddings.create(input=text, model="text-embedding-3-small")
    Here we use a mock that hashes words to simulate a vector.
    """
    import hashlib
    words = text.lower().split()
    vec = np.zeros(64)
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 64
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


class SimpleVectorStore:
    def __init__(self):
        self.documents: list[dict] = []

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self.documents.append({
            "id": doc_id,
            "text": text,
            "embedding": embed_text(text),
            "metadata": metadata or {}
        })

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = embed_text(query)
        scored = [
            {**doc, "score": cosine_similarity(query_vec, doc["embedding"])}
            for doc in self.documents
        ]
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# --- Build a knowledge base ---
store = SimpleVectorStore()
store.add("doc1", "FastAPI uses async/await for non-blocking request handling.")
store.add("doc2", "Pydantic models validate request and response schemas in FastAPI.")
store.add("doc3", "SQLAlchemy is the recommended ORM for database access in Python.")
store.add("doc4", "Alembic handles database migrations when using SQLAlchemy.")
store.add("doc5", "JWT tokens are commonly used for stateless authentication in APIs.")


def answer_with_rag(user_query: str) -> str:
    # 1. Select relevant chunks
    results = store.search(user_query, top_k=3)
    context_block = "\n".join(
        f"[{r['id']}] {r['text']}" for r in results
    )

    # 2. Inject only the relevant context
    messages = [
        {
            "role": "user",
            "content": (
                f"## Relevant documentation\n{context_block}\n\n"
                f"## Question\n{user_query}"
            )
        }
    ]
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system="Answer questions using only the provided documentation. Cite source IDs.",
        messages=messages,
    )
    return response.content[0].text


print(answer_with_rag("How do I validate request data in FastAPI?"))
```

---

### Pattern 3: Compress — Summarisation and Trimming

**Compressing** context reduces token count while preserving signal. This is essential for long-running agents where conversation history and tool outputs accumulate.

**Sub-patterns:**

- **Rolling summarisation** — replace old turns with a summary
- **Hard trimming** — drop messages older than N turns (simple but lossy)
- **Selective compression** — use an LLM to distil only task-relevant facts

```python
from anthropic import Anthropic
from dataclasses import dataclass, field

client = Anthropic()

@dataclass
class CompressedConversation:
    """
    Maintains a conversation that auto-compresses when it grows too large.
    Uses a rolling summary + recent window approach.
    """
    summary: str = ""
    recent_messages: list[dict] = field(default_factory=list)
    max_recent: int = 6          # keep last N turns verbatim
    compress_threshold: int = 10  # compress when total turns exceed this

    def add_turn(self, role: str, content: str) -> None:
        self.recent_messages.append({"role": role, "content": content})
        if len(self.recent_messages) > self.compress_threshold:
            self._compress()

    def _compress(self) -> None:
        """Summarise the oldest messages, keep only the most recent window."""
        to_compress = self.recent_messages[:-self.max_recent]
        self.recent_messages = self.recent_messages[-self.max_recent:]

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_compress
        )
        prompt = (
            f"Existing summary:\n{self.summary}\n\n"
            f"New conversation to add:\n{history_text}\n\n"
            "Produce a concise updated summary preserving all key facts, "
            "decisions, and user preferences. Be dense and factual."
        )
        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        self.summary = resp.content[0].text
        print(f"[Compressed] Summary updated ({len(to_compress)} turns compressed)")

    def build_messages(self, new_user_message: str) -> list[dict]:
        """Construct the message list to send to the API."""
        messages = []

        # Prepend summary as a synthetic assistant message if it exists
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Conversation summary so far]\n{self.summary}"
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have that context."
            })

        messages.extend(self.recent_messages)
        messages.append({"role": "user", "content": new_user_message})
        return messages


# Simulation
conv = CompressedConversation(max_recent=4, compress_threshold=6)

turns = [
    ("user",      "My name is Alex and I'm building a Python REST API."),
    ("assistant", "Great, I can help with that. Are you using FastAPI or Flask?"),
    ("user",      "FastAPI. I need authentication."),
    ("assistant", "For FastAPI, JWT with python-jose is a common choice."),
    ("user",      "Also I need async database access."),
    ("assistant", "Use SQLAlchemy with asyncpg for async Postgres."),
    ("user",      "Should I use Alembic for migrations?"),
    ("assistant", "Yes, Alembic integrates well with SQLAlchemy."),
]

for role, content in turns:
    conv.add_turn(role, content)

messages = conv.build_messages("Can you recap what stack we decided on?")
print(f"\nFinal message count sent to API: {len(messages)}")
```

---

### Pattern 4: Isolate — Sub-Agents and Context Partitioning

**Isolating** context means splitting work across separate agents, each with its own focused context window. Anthropic's multi-agent research found that many agents with isolated contexts outperformed a single agent with a large combined context.

**Sub-patterns:**

- **Specialist sub-agents** — each handles a narrow sub-task with only relevant context
- **Orchestrator / worker split** — a planner routes tasks; workers execute
- **Parallel fan-out** — multiple agents run simultaneously on independent chunks

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

# --- Specialist Sub-Agent Pattern ---

async def security_agent(code: str) -> str:
    """Isolated agent with ONLY security context."""
    resp = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system=(
            "You are a security code reviewer. "
            "Focus exclusively on: injection vulnerabilities, auth issues, "
            "secrets in code, and insecure dependencies. "
            "Respond with a JSON list of findings."
        ),
        messages=[{"role": "user", "content": f"Review:\n```python\n{code}\n```"}]
    )
    return resp.content[0].text


async def performance_agent(code: str) -> str:
    """Isolated agent with ONLY performance context."""
    resp = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system=(
            "You are a Python performance reviewer. "
            "Focus exclusively on: algorithmic complexity, unnecessary I/O, "
            "missing async/await, and memory leaks. "
            "Respond with a JSON list of findings."
        ),
        messages=[{"role": "user", "content": f"Review:\n```python\n{code}\n```"}]
    )
    return resp.content[0].text


async def style_agent(code: str) -> str:
    """Isolated agent with ONLY style context."""
    resp = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system=(
            "You are a Python style reviewer. "
            "Focus exclusively on: PEP 8 violations, naming conventions, "
            "missing type hints, and docstrings. "
            "Respond with a JSON list of findings."
        ),
        messages=[{"role": "user", "content": f"Review:\n```python\n{code}\n```"}]
    )
    return resp.content[0].text


async def orchestrator_review(code: str) -> dict:
    """
    Orchestrator: fan out to specialists in parallel,
    then synthesise results with a fresh, focused context.
    """
    # Parallel fan-out — each agent gets its own isolated context
    security, performance, style = await asyncio.gather(
        security_agent(code),
        performance_agent(code),
        style_agent(code),
    )

    # Synthesis agent receives only the findings — not the original code
    synthesis_prompt = (
        f"Security findings:\n{security}\n\n"
        f"Performance findings:\n{performance}\n\n"
        f"Style findings:\n{style}\n\n"
        "Produce a prioritised summary of the top 3 issues to fix first."
    )
    synth_resp = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=400,
        system="You are a senior engineer synthesising code review results.",
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    return {
        "security": security,
        "performance": performance,
        "style": style,
        "summary": synth_resp.content[0].text,
    }


# Example
sample_code = """
import sqlite3, os

password = "admin123"  # hardcoded

def get_user(username):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
    return cursor.fetchall()
"""

# results = asyncio.run(orchestrator_review(sample_code))
# print(results["summary"])
```

---

## Pattern 5: Prompt Layering — Separating System vs. User Context

A foundational structural pattern: cleanly separate **system context** (stable instructions, persona, constraints) from **user context** (dynamic data, retrieved docs, tool results).

```python
from anthropic import Anthropic
from datetime import date

client = Anthropic()


def build_system_prompt(user_role: str, capabilities: list[str]) -> str:
    """
    Layer 1: Stable system instructions — cached by the API across calls.
    These rarely change and benefit from prompt caching.
    """
    caps = "\n".join(f"- {c}" for c in capabilities)
    return f"""You are a senior AI engineer assistant.
Today's date: {date.today()}
User role: {user_role}

## Your capabilities
{caps}

## Constraints
- Always cite sources when referencing documentation
- Prefer Python examples unless asked otherwise
- Flag security concerns proactively
"""


def build_user_message(
    query: str,
    retrieved_docs: list[str] | None = None,
    tool_results: list[str] | None = None,
) -> str:
    """
    Layer 2: Dynamic user context — assembled fresh per request.
    Combines retrieved knowledge with the actual question.
    """
    parts: list[str] = []

    if retrieved_docs:
        docs_block = "\n\n".join(retrieved_docs)
        parts.append(f"## Retrieved documentation\n{docs_block}")

    if tool_results:
        tools_block = "\n".join(tool_results)
        parts.append(f"## Tool results\n{tools_block}")

    parts.append(f"## Question\n{query}")
    return "\n\n".join(parts)


# Usage
system = build_system_prompt(
    user_role="Senior Python Developer",
    capabilities=["code review", "architecture advice", "debugging", "performance tuning"]
)

user_msg = build_user_message(
    query="Should I use asyncio.gather or TaskGroup for parallel API calls in Python 3.12?",
    retrieved_docs=[
        "asyncio.TaskGroup (Python 3.11+) provides structured concurrency with automatic "
        "cancellation of sibling tasks on failure.",
        "asyncio.gather() collects results from multiple coroutines but has looser "
        "cancellation semantics."
    ],
    tool_results=["Python version detected: 3.12.2"]
)

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=512,
    system=system,
    messages=[{"role": "user", "content": user_msg}]
)
print(response.content[0].text)
```

---

## Pattern 6: Context Budget Management

Explicitly track and enforce token budgets per context section to prevent context rot and stay within limits.

```python
from anthropic import Anthropic

client = Anthropic()

# Rough token estimator (production: use tiktoken or the Anthropic tokenizer)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextBudget:
    """
    Enforces a token budget across named context sections.
    Truncates sections that exceed their allocation.
    """

    def __init__(self, total_budget: int = 4096):
        self.total = total_budget
        self.sections: dict[str, dict] = {}

    def allocate(self, name: str, max_tokens: int, priority: int = 5) -> None:
        """Reserve a named section with a token ceiling and a priority (1=highest)."""
        self.sections[name] = {"max_tokens": max_tokens, "priority": priority, "content": ""}

    def set(self, name: str, content: str) -> None:
        """Assign content to a section, truncating to its budget."""
        section = self.sections[name]
        tokens = estimate_tokens(content)
        if tokens > section["max_tokens"]:
            # Trim to budget (character approximation)
            char_limit = section["max_tokens"] * 4
            content = content[:char_limit] + "\n...[truncated]"
            print(f"[Budget] '{name}' truncated: {tokens} → {section['max_tokens']} tokens")
        section["content"] = content

    def render(self) -> str:
        """Render all sections sorted by priority."""
        ordered = sorted(self.sections.items(), key=lambda x: x[1]["priority"])
        parts = []
        total_used = 0
        for name, section in ordered:
            if not section["content"]:
                continue
            chunk = f"## {name}\n{section['content']}"
            used = estimate_tokens(chunk)
            if total_used + used > self.total:
                print(f"[Budget] Dropping section '{name}' — budget exhausted")
                continue
            parts.append(chunk)
            total_used += used
        print(f"[Budget] Total estimated tokens used: {total_used}/{self.total}")
        return "\n\n".join(parts)


# Example
budget = ContextBudget(total_budget=2000)
budget.allocate("system_instructions",  max_tokens=400, priority=1)
budget.allocate("retrieved_docs",       max_tokens=800, priority=2)
budget.allocate("conversation_history", max_tokens=600, priority=3)
budget.allocate("user_query",           max_tokens=200, priority=1)

budget.set("system_instructions",  "You are a Python expert. Be concise and accurate.")
budget.set("retrieved_docs",       "FastAPI doc: " + "x" * 2000)  # will be truncated
budget.set("conversation_history", "User asked about routers. Assistant explained path params.")
budget.set("user_query",           "How do I add middleware to FastAPI?")

context = budget.render()
```

---

## Summary: Choosing the Right Pattern

|Pattern|Best For|Watch Out For|
|---|---|---|
|**Write**|Long-running agents, cross-session memory|Stale or contradictory facts accumulating|
|**Select (RAG)**|Large knowledge bases, factual Q&A|Retrieval misses; embedding quality matters|
|**Compress**|Long conversations, high-cost workflows|Summary LLM losing nuanced details|
|**Isolate**|Complex multi-step tasks, parallelism|Orchestration overhead; inter-agent communication cost|
|**Prompt Layering**|All production systems|Mixing stable/dynamic content reduces cacheability|
|**Budget Management**|Cost-sensitive or latency-sensitive apps|Hard to tune budgets without monitoring|

---

## Key Principles

1. **Context is a scarce resource.** Treat every token as having a cost — in latency, money, and model attention.
2. **Structure beats volume.** A well-organised 2,000-token context outperforms a poorly structured 8,000-token one.
3. **Separate concerns.** System instructions, retrieved knowledge, history, and user queries belong in distinct, clearly labelled sections.
4. **Monitor context rot.** Add token-count logging from day one; performance degradation from over-full contexts is silent and insidious.
5. **Compress proactively.** Don't wait for the window to fill — summarise conversation history on a rolling basis.
6. **Isolate for scale.** When a task grows complex, decompose it into specialist agents rather than enlarging a single context.

---

_Report generated April 2026. References: LangChain Context Engineering Guide, Chroma Context Rot Research (2025), Anthropic Multi-Agent Research, ACE Framework (arXiv:2510.04618)._