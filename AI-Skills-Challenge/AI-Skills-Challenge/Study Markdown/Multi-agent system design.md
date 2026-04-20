
# Multi-agent system design

## 1. The Paradigm Shift: From Single to Multi-Agent

Traditional LLMs wait for prompts and produce standalone answers. Agentic systems, by contrast, distribute intelligence across multiple decision points. This is the core shift: from a model that _responds_ to one that _acts_.

Consider a traditional single-agent system: one language model processes input, generates output, and that's the end of the interaction. A multi-agent system breaks that monolith apart. Multi-agent systems distribute work across specialists. While one agent qualifies leads, another analyzes customer sentiment, and a third handles competitive research — all simultaneously.

### Why Single Agents Hit a Wall

Single-agent systems face several fundamental limitations. Context Window Constraints: even with expanding context windows, a single agent attempting to handle everything becomes overwhelmed — specialized agents can focus on relevant information for their domain. Hallucination Compounding: single agents handling diverse domains are more prone to errors, while multi-agent systems allow agents to check each other's work. Scalability Limitations: a monolithic agent becomes a bottleneck, whereas multiple agents can work in parallel on different aspects of a problem.

The analogy from Google's engineering blog puts it well: a single agent tasked with too many responsibilities becomes a "Jack of all trades, master of none." As the complexity of instructions increases, adherence to specific rules degrades, and error rates compound, leading to more hallucinations. Multi-Agent Systems allow you to build the AI equivalent of a microservices architecture.

---

## 2. Core Components of a Multi-Agent System

Every agent in a MAS shares the same fundamental anatomy:

- **LLM/Policy Core** — the reasoning engine
- **Tools** — functions the agent can invoke (search, code exec, APIs)
- **Memory** — short-term (context window) and long-term (vector DB, key-value stores)
- **Environment** — the shared space agents act within (databases, APIs, message queues)

An agent is not only a generator of text; it is a controller that translates intent into procedures carried out in the world — software repositories, browsers, enterprise systems, or physical robots.

---

## 3. The Four Core Architectural Patterns

Four architectural patterns form the foundation of most multi-agent applications: subagents, skills, handoffs, and routers. Each takes a different approach to task coordination, state management, and sequential unlocking.

### Pattern 1: Subagents (Orchestrator + Workers)

In the subagents pattern, a supervisor agent coordinates specialized subagents by calling them as tools. The main agent maintains conversation context while subagents remain stateless, providing strong context isolation. The main agent decides which subagents to invoke, what input to provide, and how to combine results. Subagents don't remember past interactions. This architecture provides centralized control where all routing passes through the main agent, which can invoke multiple subagents in parallel.

**Python example using LangGraph:**

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# --- Specialist subagents as tools ---
@tool
def research_agent(query: str) -> str:
    """Search the web and summarize findings on a topic."""
    agent_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    result = agent_llm.invoke(
        f"You are a research specialist. Answer this thoroughly: {query}"
    )
    return result.content

@tool
def code_agent(task: str) -> str:
    """Write and explain Python code for a given task."""
    agent_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    result = agent_llm.invoke(
        f"You are a Python expert. Write clean, commented code for: {task}"
    )
    return result.content

@tool
def review_agent(content: str) -> str:
    """Review and critique content for accuracy and completeness."""
    agent_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    result = agent_llm.invoke(
        f"You are a critical reviewer. Review this and flag issues:\n{content}"
    )
    return result.content

# --- Orchestrator ---
orchestrator = llm.bind_tools([research_agent, code_agent, review_agent])

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def orchestrator_node(state: AgentState):
    response = orchestrator.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("orchestrator", orchestrator_node)
graph.set_entry_point("orchestrator")
graph.add_edge("orchestrator", END)
app = graph.compile()
```

### Pattern 2: Handoffs (Routing / Triage)

Use handoffs when routing itself is part of the workflow and you want the chosen specialist to own the next part of the interaction.

The router/handoff pattern requires one routing call (~200 tokens) plus one specialist call (~1,000 tokens) — 1,200 tokens total. That is why it is the most common pattern in production.

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Literal

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

class State(TypedDict):
    query: str
    route: str
    response: str

# Router: classify intent
def router_node(state: State) -> State:
    result = llm.invoke(
        f"Classify this query as one of: [billing, technical, general].\n"
        f"Reply with only the category word.\nQuery: {state['query']}"
    )
    return {"route": result.content.strip().lower()}

def billing_node(state: State) -> State:
    result = llm.invoke(
        f"You are a billing specialist. Answer: {state['query']}"
    )
    return {"response": result.content}

def technical_node(state: State) -> State:
    result = llm.invoke(
        f"You are a technical support engineer. Answer: {state['query']}"
    )
    return {"response": result.content}

def general_node(state: State) -> State:
    result = llm.invoke(f"Answer this general question: {state['query']}")
    return {"response": result.content}

def route_selector(state: State) -> Literal["billing", "technical", "general"]:
    return state["route"]

graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("billing", billing_node)
graph.add_node("technical", technical_node)
graph.add_node("general", general_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_selector)
for node in ["billing", "technical", "general"]:
    graph.add_edge(node, END)

app = graph.compile()
result = app.invoke({"query": "My invoice is wrong this month."})
print(result["response"])
```

### Pattern 3: Sequential Pipeline (Assembly Line)

Agents pass outputs to the next stage in order — ideal for document processing, content pipelines, or data transformation. The `SequentialAgent` primitive handles orchestration via shared session state: use `output_key` to write to shared state so the next agent knows exactly where to pick up the work.

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def run_pipeline(raw_input: str) -> dict:
    """Three-stage pipeline: extract → analyze → report."""

    # Stage 1: Extractor
    extracted = llm.invoke(
        f"Extract the key facts and entities from this text:\n{raw_input}"
    ).content

    # Stage 2: Analyzer
    analysis = llm.invoke(
        f"Analyze these extracted facts for patterns and insights:\n{extracted}"
    ).content

    # Stage 3: Reporter
    report = llm.invoke(
        f"Write a concise executive summary based on this analysis:\n{analysis}"
    ).content

    return {"extracted": extracted, "analysis": analysis, "report": report}
```

### Pattern 4: Parallel Fan-Out

For multi-domain tasks, patterns with parallel execution are most efficient. Subagents processes 67% fewer tokens overall due to context isolation — each subagent works only with relevant context.

```python
import asyncio
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

async def analyze_aspect(topic: str, aspect: str) -> tuple[str, str]:
    result = await llm.ainvoke(
        f"Analyze the {aspect} of: {topic}. Be specific and concise."
    )
    return aspect, result.content

async def parallel_analysis(topic: str) -> dict:
    """Fan out to multiple specialist agents simultaneously."""
    aspects = ["technical feasibility", "market opportunity", "risks", "timeline"]
    tasks = [analyze_aspect(topic, a) for a in aspects]
    results = await asyncio.gather(*tasks)
    return dict(results)

# Run
report = asyncio.run(parallel_analysis("Deploying LLM agents in healthcare"))
for aspect, analysis in report.items():
    print(f"\n## {aspect.title()}\n{analysis}")
```

---

## 4. Communication Patterns Between Agents

How agents exchange information is as important as what they do. There are three primary models:

|Pattern|How it Works|Best For|
|---|---|---|
|**Shared State**|Agents read/write to a common dict or DB|Sequential pipelines|
|**Message Passing**|Agents communicate via structured messages|Async / parallel workflows|
|**Tool Invocation**|One agent calls another as a function|Orchestrator-worker|

The universal anti-pattern across all topologies is passing unstructured free text between agents. Structured output schemas using Pydantic validation at every agent boundary reduce variance and improve auditability.

```python
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Enforce structured outputs at agent boundaries
class ResearchOutput(BaseModel):
    summary: str
    key_facts: list[str]
    confidence: float
    sources_needed: bool

structured_llm = llm.with_structured_output(ResearchOutput)
result = structured_llm.invoke("Research the current state of agentic AI frameworks")
# result is a validated ResearchOutput object — safe to pass to the next agent
```

---

## 5. Memory Architecture

Agents need multiple memory layers:

- **In-context (ephemeral)** — the current conversation window
- **External short-term** — Redis or similar for session state
- **Long-term (persistent)** — vector databases like Chroma or Pinecone for semantic retrieval
- **Shared state** — a key-value store accessible across agents in the same workflow

```python
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import FakeEmbeddings  # replace with real embeddings

# Shared long-term memory store
vectorstore = Chroma(embedding_function=FakeEmbeddings(size=1536))

def remember(content: str, metadata: dict):
    vectorstore.add_texts([content], metadatas=[metadata])

def recall(query: str, k: int = 3) -> list[str]:
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]

# Agents call remember() to persist findings and recall() to retrieve context
```

---

## 6. The Framework Landscape (2026)

The agentic framework landscape consolidated around three dominant open-source options: LangGraph, Microsoft's AutoGen, and CrewAI. Each framework embodies different design philosophies that determine appropriate use cases.

The orchestration model differs significantly: LangGraph uses a directed graph with conditional edges; CrewAI uses role-based crews with process types; OpenAI SDK uses explicit handoffs; AutoGen/AG2 uses conversational GroupChat; and Google ADK uses a hierarchical agent tree.

**Framework decision guide:**

|Framework|Orchestration Model|State|Model Support|Best For|
|---|---|---|---|---|
|**LangGraph**|Directed graph|Checkpointed|Any|Complex conditional workflows|
|**CrewAI**|Role-based crews|Sequential task outputs|Any|Role-playing teams, fastest to start|
|**AutoGen**|Group chat|Conversation history|Any|Research, multi-agent dialogue|
|**OpenAI Agents SDK**|Explicit handoffs|Ephemeral|OpenAI only|Clean, production-grade pipelines|
|**Google ADK**|Hierarchical tree|Pluggable backends|Gemini-optimized|GCP deployments|

**MCP (Model Context Protocol)** is now the cross-framework standard for tool integration. MCP by Anthropic standardizes how agents access tools and external resources — no more custom integrations for every connection.

---

## 7. Real-World Production Examples

Genentech built agent ecosystems on AWS to automate complex research workflows, with a system that coordinates 10+ specialized agents, each expert in molecular analysis, regulatory compliance, or clinical trial design. Amazon used Amazon Q Developer to coordinate agents that modernized thousands of legacy Java applications.

Capital One's Chat Concierge uses a coordinating agent to orchestrate specialists across auto finance workflows, with hallucination and error mitigation handled at the coordination layer before outputs reach customers. Amazon's healthcare multi-agent system uses hierarchical orchestration with specialized domain expert sub-agents — a validation agent for medication directions achieved a 33% reduction in near-miss medication events.

Multi-Agent Research Assistants on platforms like AutoGen and CrewAI assign specialized roles to multiple agents — retrievers, summarizers, synthesizers, and citation formatters — under a central orchestrator. These systems are being used for literature reviews, grant preparation, and patent search pipelines, outperforming single-agent systems by enabling concurrent sub-task execution and long-context management.

---

## 8. Key Engineering Pitfalls and How to Avoid Them

Common failure modes include: too many agents too soon (start with 2 agents, get coordination right, add a third only when you have a real use case); overlapping tool responsibilities (if both your research agent and general agent can search the web, the coordinator will pick the wrong one — each agent needs exclusive ownership of its tools); and no recursion limit (without `recursion_limit`, a handoff loop between two agents runs until your token budget is gone).

Every agent in the chain adds 1–3 seconds of LLM latency. A 3-agent sequential pipeline adds 3–9 seconds minimum. For batch workflows this is fine; for real-time chat, it is a dealbreaker. A bug in a multi-agent system means reconstructing message flows across 3–10 agents, figuring out which agent made the wrong decision, and understanding how that decision propagated through the pipeline. Invest in logging every routing decision, every agent input/output, and every tool call. Without observability, multi-agent systems are black boxes.

**Minimum production checklist:**

```python
# 1. Always set recursion limits
config = {"recursion_limit": 15}

# 2. Validate inter-agent messages with Pydantic
class AgentMessage(BaseModel):
    agent_id: str
    task: str
    result: str
    confidence: float

# 3. Log every decision
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent.orchestrator")
logger.info(f"Routing to: {selected_agent}, reason: {routing_rationale}")

# 4. Add timeouts to tool calls
import asyncio
async def safe_tool_call(tool_fn, input, timeout=30):
    try:
        return await asyncio.wait_for(tool_fn(input), timeout=timeout)
    except asyncio.TimeoutError:
        return {"error": "Tool call timed out"}
```

---

## 9. The Decision Framework: When to Go Multi-Agent

Most teams go multi-agent too early. Start simple. Graduate to complexity only when the simple pattern demonstrably fails.

Go **multi-agent** when you have:

- Tasks that are too long for one context window
- Distinct domains requiring truly different expertise or tools
- Steps that can genuinely run in parallel
- A need for agents to independently verify each other's outputs

Stay **single-agent** when:

- The task fits in one context window
- Routing logic can be solved with a dynamic system prompt
- Tools can be grouped into one agent without conflicts
- Latency is a primary constraint

---

## 10. The State of the Field (April 2026)

Carnegie Mellon benchmarks show leading agents complete only 30–35% of multi-step tasks — reliability engineering is becoming the critical differentiator. This means your most important work as an AI Engineer right now is not building more complex systems, but making existing ones more robust: better evals, structured inter-agent contracts, observability from day one, and graceful failure handling.

Production agent systems require observability instrumentation from the start. Retrofitting tracing into existing agent systems proves difficult due to the deep integration required with agent decision points. Plan observability architecture during initial agent design, instrumenting every tool invocation, reasoning step, and memory access.

The core insight that ties everything together: governance enforced architecturally — locked SQL, mandatory compliance gate nodes, microVM isolation — outperforms governance as policy. Compliance is architecture, not configuration. Design your agent boundaries and contracts first; the intelligence fills in around them.