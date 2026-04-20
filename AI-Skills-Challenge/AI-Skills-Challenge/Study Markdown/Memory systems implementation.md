
# Memory systems implementation

## The Core Problem: LLMs Are Stateless

LLMs that power AI agents are stateless and don't have memory as a built-in feature. LLMs learn and remember information during their training phase and store it in their model weights, but they don't immediately learn and remember what you just said. Therefore, every time you interact with an LLM, each time is essentially a fresh start.

Most agentic frameworks conflate context windows with memory. A context window holds recent conversation turns during a session but resets when the task ends. Real memory persists across sessions and lets agents learn from past interactions.

Even as models like Claude 3.7 Sonnet (200K tokens) push the boundaries of context length, these improvements merely delay rather than solve the fundamental limitation.

Memory systems are the engineering solution to this problem.

---

## The Four Memory Types (The Standard Taxonomy)

The field converges on four types, drawn from cognitive science:

### 1. In-Context Memory (Working / Short-Term Memory)

This is the live context window — what the agent can "see" right now. It holds the current conversation, tool call results, intermediate reasoning, and any injected history.

Short-term memory is typically managed using a sliding window approach within the language model's prompt input to maintain recent context.

**Implementation:** You manage this directly by constructing your `messages` list on each API call. Techniques include conversation buffering, summarization of older turns, and "pinning" critical facts to the system prompt.

```python
# Simple sliding window short-term memory
from collections import deque
from anthropic import Anthropic

client = Anthropic()

class ShortTermMemory:
    def __init__(self, max_turns: int = 10):
        self.turns = deque(maxlen=max_turns * 2)  # user + assistant pairs
    
    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
    
    def get_messages(self) -> list:
        return list(self.turns)

stm = ShortTermMemory(max_turns=10)
stm.add("user", "My name is Alex and I'm building a RAG pipeline.")
stm.add("assistant", "Great, happy to help with your RAG pipeline, Alex.")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=stm.get_messages() + [{"role": "user", "content": "What was I building?"}]
)
```

### 2. Episodic Memory (Long-Term: Event-Based)

Episodic memory allows AI agents to recall specific past experiences, similar to how humans remember individual events. This type of memory is useful for case-based reasoning, where an AI learns from past events to make better decisions in the future. Episodic memory is often implemented by logging key events, actions and their outcomes in a structured format that the agent can access when making decisions.

Episodic memory in AI agents stores detailed, time-based records of past interactions. It typically consists of conversation logs, tool usage, and environmental changes, all saved with timestamps and metadata. This allows agents to maintain continuity across sessions — for example, recalling a previous customer support issue and referencing it naturally in future interactions.

### 3. Semantic Memory (Long-Term: Factual / Knowledge)

Semantic memory holds structured factual knowledge: user preferences, domain facts, entity relationships, and general world knowledge relevant to the agent's scope. A customer service agent that knows a user prefers concise answers and operates in the legal industry is drawing on semantic memory. This is often implemented as entity profiles updated incrementally over time, combining relational storage for structured fields with vector storage for fuzzy retrieval.

Unlike episodic memory, which records individual interactions, semantic memory extracts and preserves key information — such as turning a past interaction about a peanut allergy into a permanent fact like "User Allergy: Peanuts."

### 4. Procedural Memory (Long-Term: Skills / Rules)

Procedural memory encodes how to do things — workflows, decision rules, and learned behavioral patterns. In practice, this shows up as system prompt instructions, few-shot examples, or agent-managed rule sets that evolve through experience. A coding assistant that has learned to always check for dependency conflicts before suggesting library upgrades is expressing procedural memory.

---

## The Two Write Paths: Explicit vs. Implicit

Explicit memory (hot path) describes the agent memory system's ability to autonomously recognize important information and decide to explicitly remember it via tool calling. Implicit memory (background) describes when memory management is programmatically defined in the system at specific times during or after a conversation — for instance, batch processing the entire conversation after a session ends.

In practice, **explicit** = the agent calls a `save_memory` tool when it decides something is important. **Implicit** = your infrastructure runs a background job to extract and store memories after each session. Most production systems use both.

---

## Storage Backends

Implementations typically use vector databases — such as Faiss, Qdrant, or Milvus — to store embeddings and provide rapid, relevance-based retrieval for memory modules.

Here's how the storage options map to memory types:

|Memory Type|Storage Backend|Retrieval Method|
|---|---|---|
|Working/STM|In-process list / Redis|Sequential|
|Episodic|Vector DB (Qdrant, Chroma, Pinecone)|Semantic similarity|
|Semantic|Vector DB + Relational (Postgres)|Hybrid: exact + semantic|
|Procedural|System prompt / Key-value store|Lookup|

---

## A Practical Pattern: RAG-Based Long-Term Memory

The most common pattern in production is to use a vector database as the long-term memory store and retrieve relevant chunks at the start of each agent turn.

```python
import json
from anthropic import Anthropic
from datetime import datetime

# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions

client = Anthropic()
chroma_client = chromadb.Client()
ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="agent_memory",
    embedding_function=ef
)

def save_memory(user_id: str, content: str, memory_type: str = "episodic"):
    """Persist a memory to the vector store."""
    memory_id = f"{user_id}_{datetime.now().timestamp()}"
    collection.add(
        documents=[content],
        metadatas=[{"user_id": user_id, "type": memory_type, "timestamp": datetime.now().isoformat()}],
        ids=[memory_id]
    )

def retrieve_memories(user_id: str, query: str, n: int = 5) -> list[str]:
    """Retrieve the top-n relevant memories for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where={"user_id": user_id}
    )
    return results["documents"][0] if results["documents"] else []

def run_agent(user_id: str, user_message: str) -> str:
    # 1. Retrieve relevant long-term memories
    memories = retrieve_memories(user_id, user_message)
    memory_block = "\n".join(f"- {m}" for m in memories)
    
    # 2. Build system prompt with injected memories
    system_prompt = f"""You are a helpful assistant with access to the user's history.

Relevant memories:
{memory_block if memory_block else 'No relevant memories found.'}

Use these memories to personalize your response."""

    # 3. Call the LLM
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    reply = response.content[0].text
    
    # 4. Save this interaction as a new memory (implicit write)
    save_memory(user_id, f"User said: {user_message}. Agent replied: {reply[:200]}")
    
    return reply

# Usage
save_memory("user_alex", "User prefers Python. Works on RAG pipelines.", memory_type="semantic")
save_memory("user_alex", "User previously asked about Chroma vs Pinecone.", memory_type="episodic")

print(run_agent("user_alex", "Which vector DB should I use for my project?"))
```

---

## The Explicit Memory (Tool-Use) Pattern

Here the agent decides for itself what to remember by calling tools:

```python
import json
from anthropic import Anthropic

client = Anthropic()

# Simulated in-memory store for this example
memory_store: list[str] = []

tools = [
    {
        "name": "save_memory",
        "description": "Save an important piece of information to long-term memory for future recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact or insight to remember."},
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "description": "The type of memory."
                }
            },
            "required": ["content", "memory_type"]
        }
    },
    {
        "name": "recall_memories",
        "description": "Search long-term memory for relevant information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."}
            },
            "required": ["query"]
        }
    }
]

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    if tool_name == "save_memory":
        memory_store.append(f"[{tool_input['memory_type']}] {tool_input['content']}")
        return f"Memory saved: {tool_input['content']}"
    elif tool_name == "recall_memories":
        # In production: do semantic search here
        relevant = [m for m in memory_store if tool_input["query"].lower() in m.lower()]
        return json.dumps(relevant) if relevant else "No relevant memories found."
    return "Unknown tool."

def agent_loop(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages,
            system="You are a helpful assistant. Use save_memory to store important facts and recall_memories before answering questions about past context."
        )
        
        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))
        
        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = handle_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        messages.append({"role": "user", "content": tool_results})

print(agent_loop("I prefer dark mode and I'm allergic to peanuts. Remember this."))
print(agent_loop("What do you know about my preferences?"))
```

---

## The Critical Engineering Challenges

The difficulty lies in optimizing the system to avoid slower response times while simultaneously solving the complex problem of determining what information is obsolete and should be permanently deleted. Latency: constantly processing whether the agent now needs to retrieve new information from or offload data to the memory bank can lead to slower response times. Forgetting: this seems to be the hardest challenge for developers at the moment — how do you automate a mechanism that decides when and what information to permanently delete?

The naive approach to agent memory management — simply appending every new conversation turn into a vector database — inevitably leads to catastrophic systemic failure. As the data corpus grows over weeks or months of deployment, agents experience debilitating retrieval noise, severe context dilution, and latency spikes.

Memory without forgetting is as problematic as no memory at all. Memory entries should carry timestamps, source provenance, and explicit expiration conditions. Implement decay strategies so older, less relevant memories don't pollute retrieval as your store grows.

**Key design principles for production:**

- **Distill, don't dump.** Extract structured memory objects (facts, preferences, outcomes) rather than storing raw transcripts.
- **Write asynchronously.** Enterprise-grade architectures uniformly rely on asynchronous, background consolidation paradigms to avoid adding latency to the user-facing response path.
- **Hybrid retrieval.** "What did this user say about billing in the last 30 days?" requires both semantic matching and a date filter. Use vector similarity + metadata filters together.
- **Handle conflicts.** If the system detects a conflict — for example, "User prefers React" and "User is building entirely in Vue" — an arbiter LLM decides whether the new statement is a duplicate, a refinement, or a complete pivot, and compresses the old memory into a temporal reflection summary.

---

## The Ecosystem: Key Libraries

Commercial frameworks — both proprietary and open-source — now offer built-in memory support. Examples include Mem0, LangChain, AutoGPT, Haystack, MemAI, and OctoTools.

- **Mem0** — Dedicated memory orchestration layer. Manages the full memory lifecycle: extraction, storage, deduplication, decay, and retrieval. Integrates with most vector DBs. Best for production use cases.
- **LangGraph** — Represents agent workflows as directed graphs where state is maintained as typed dictionaries using Python's TypedDict, ensuring type safety while allowing flexible representation of complex state.
- **Letta (MemGPT)** — Agents that explicitly manage their own memory via tool calls; good for long-horizon tasks.
- **Vector DBs:** Chroma (local/dev), Qdrant (production), Pinecone (managed), Weaviate (hybrid search).

---

## Multi-Agent Memory Considerations

The Multi-Agent Core enables multiple AI agents — planners, retrievers, reasoners, verifiers — to collaborate within a shared memory substrate. Procedural memory for task flow templates, Episodic memory for context continuity, and Semantic memory for trusted factual grounding. This distributed memory exchange avoids hallucinations, improves interpretability, and enhances accountability.

Mem0 manages the memory lifecycle, from extracting information from agent interactions to storing and retrieving it efficiently, and handles memory operations such as automatic filtering to prevent memory bloat, decay mechanisms that remove irrelevant information over time, and cost optimization features that reduce LLM expenses through prompt injection and semantic caching.

---

## Summary: Design Checklist for AI Engineers

Before building any agentic memory system, answer these questions for each memory type:

1. **What gets stored?** Raw turns, or distilled facts/preferences/outcomes?
2. **When is it written?** Hot path (explicit tool call) or background job (implicit)?
3. **How is it retrieved?** Semantic search, exact lookup, hybrid, or graph traversal?
4. **How does it decay or expire?** TTL? Scoring? LLM-based conflict resolution?
5. **Who owns it?** Per-user, per-agent, or shared across a multi-agent crew?

Memory is increasingly the differentiator that transforms a generic LLM wrapper into a genuinely intelligent, persistent agent — and it's one of the most actively evolving areas of AI engineering right now.