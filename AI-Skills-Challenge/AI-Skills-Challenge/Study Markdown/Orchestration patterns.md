# Orchestration patterns

## What Is Orchestration in AI Engineering?

**Orchestration** in AI Engineering refers to the coordination of one or more AI agents (typically LLM-backed) to complete complex, multi-step tasks that exceed the capabilities of a single model call. It encompasses the management of:

- **Task decomposition** — breaking a goal into sub-tasks
- **Agent coordination** — determining which agent acts, when, and with what input
- **State and context management** — passing relevant information between agents
- **Tool and API integration** — calling external systems as part of the workflow
- **Error handling and fallback** — recovering gracefully when agents fail

LLM orchestration has evolved from simple prompt chaining to sophisticated multi-agent systems where specialized agents collaborate, critique each other, and route work dynamically. By 2025–2026, it is considered a foundational discipline in production AI system design.

---

## Why Orchestration Patterns Matter

Single-agent approaches suffer from well-documented limitations:

- **Context overload**: Stuffing too many tools and instructions into one agent degrades adherence and increases hallucinations.
- **Quality variance**: Research has demonstrated that single-agent LLM systems produce highly inconsistent outputs, while multi-agent orchestrated systems can achieve near-zero quality variance across repeated trials.
- **Maintainability**: A monolithic agent is hard to test, debug, and iterate on. Specialized agents allow targeted improvements.
- **Scalability**: Independent agents can be swapped, upgraded, or scaled without redesigning the whole system.

Orchestration patterns give engineers a shared vocabulary and proven blueprints for solving these problems reliably.

---

## Complexity Spectrum: When to Orchestrate

Not every task needs a multi-agent system. Use the lowest complexity that reliably meets your requirements:

|Level|Description|When to Use|
|---|---|---|
|**Direct model call**|A single LLM call with a crafted prompt|Classification, summarization, single-step tasks|
|**Single agent with tools**|One agent that loops through tool calls|Varied queries in a single domain with dynamic tool use|
|**Multi-agent orchestration**|Specialized agents coordinate via a defined pattern|Cross-functional tasks, parallel specialization, security isolation|

---

## Core Orchestration Patterns

### 1. Sequential (Pipeline)

**Also known as:** prompt chaining, linear delegation, pipeline.

Each agent processes the output of the previous one in a fixed, predefined order. This creates a transformation pipeline where each stage adds specialized value.

**When to use:**

- Multi-stage processes with clear linear dependencies
- Progressive refinement workflows (draft → review → polish)
- Data transformation where each stage builds on the last

**When to avoid:**

- Stages that could run in parallel
- Workflows requiring backtracking or dynamic routing

#### Python Example

```python
from anthropic import Anthropic

client = Anthropic()

def run_agent(system_prompt: str, user_input: str) -> str:
    """Run a single agent with a system and user prompt."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    return response.content[0].text


def sequential_pipeline(raw_text: str) -> dict:
    """
    A three-stage sequential pipeline:
      1. Extractor  — pulls key facts from raw text
      2. Analyst    — identifies themes and insights
      3. Writer     — produces a polished summary
    """
    # Stage 1: Extract key facts
    facts = run_agent(
        system_prompt="You are a fact extractor. List all key facts from the text as bullet points.",
        user_input=raw_text,
    )
    print("Stage 1 - Extracted Facts:\n", facts)

    # Stage 2: Analyze themes (receives Stage 1 output)
    analysis = run_agent(
        system_prompt="You are an analyst. Given these facts, identify the 3 most important themes and why.",
        user_input=facts,
    )
    print("\nStage 2 - Analysis:\n", analysis)

    # Stage 3: Write final report (receives Stage 2 output)
    report = run_agent(
        system_prompt="You are a professional writer. Turn this analysis into a concise 2-paragraph executive summary.",
        user_input=analysis,
    )
    print("\nStage 3 - Report:\n", report)

    return {"facts": facts, "analysis": analysis, "report": report}


if __name__ == "__main__":
    sample = """
    Q3 revenue reached $4.2B, up 18% YoY. Customer churn fell to 3.2% from 5.1%.
    The APAC region grew 34%, driven by new enterprise contracts. R&D spend increased
    by 22%, primarily in AI infrastructure. Two new products shipped: DataPilot and
    CloudSync. Headcount grew from 8,400 to 9,100.
    """
    sequential_pipeline(sample)
```

---

### 2. Concurrent (Fan-out / Fan-in)

**Also known as:** parallel, scatter-gather, map-reduce, ensemble reasoning.

Multiple agents work on the same input simultaneously, each from a different perspective or specialization. Results are aggregated — via voting, weighted merging, or a synthesizing LLM call — into a final output.

**When to use:**

- Tasks benefiting from multiple independent perspectives
- Time-sensitive scenarios where parallel processing reduces latency
- Brainstorming, ensemble reasoning, quorum-based decisions

**When to avoid:**

- Agents must build on each other's work sequentially
- Resource (quota/cost) constraints make parallelism impractical

#### Python Example

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

SPECIALIST_AGENTS = {
    "security": "You are a cybersecurity expert. Analyze the following code for security vulnerabilities only.",
    "performance": "You are a performance engineer. Analyze the following code for performance issues only.",
    "maintainability": "You are a senior software engineer. Analyze the following code for maintainability and readability issues only.",
}


async def run_agent_async(name: str, system_prompt: str, code: str) -> tuple[str, str]:
    """Run a single agent asynchronously."""
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Review this code:\n\n```python\n{code}\n```"}],
    )
    return name, response.content[0].text


async def concurrent_code_review(code: str) -> dict:
    """
    Fan-out: three specialist agents analyze the same code in parallel.
    Fan-in: a synthesizer agent merges their findings.
    """
    # Fan-out — run all agents concurrently
    tasks = [
        run_agent_async(name, prompt, code)
        for name, prompt in SPECIALIST_AGENTS.items()
    ]
    results = dict(await asyncio.gather(*tasks))

    print("Specialist reviews received:", list(results.keys()))

    # Fan-in — synthesize all reviews
    combined = "\n\n".join(f"### {k.capitalize()} Review\n{v}" for k, v in results.items())
    synthesis_prompt = (
        "You are a tech lead. Given these specialist code reviews, produce a "
        "prioritized list of the top 5 actionable improvements with brief explanations."
    )
    _, final_report = await run_agent_async("synthesizer", synthesis_prompt, combined)

    results["synthesis"] = final_report
    return results


if __name__ == "__main__":
    sample_code = """
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        conn = get_db_connection()
        result = conn.execute(query)
        data = []
        for row in result:
            data.append(row)
        return data
    """
    asyncio.run(concurrent_code_review(sample_code))
```

---

### 3. Hierarchical (Orchestrator–Subagent)

**Also known as:** manager–worker, planner–executor, nested agents.

A top-level orchestrator agent decomposes a high-level goal into sub-tasks and delegates them to specialized subagents. The orchestrator synthesizes results and decides the next steps. This mirrors a management hierarchy.

**When to use:**

- Complex, multi-domain goals that require planning
- Tasks where the steps are not fully known upfront (dynamic plans)
- Systems where specialists should be isolated from each other

**When to avoid:**

- Simple, well-defined tasks a single agent can handle
- When the overhead of coordination outweighs benefits

#### Python Example

```python
import json
from anthropic import Anthropic

client = Anthropic()

# Specialist agents
def research_agent(topic: str) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system="You are a research specialist. Provide concise, factual information on the given topic.",
        messages=[{"role": "user", "content": topic}],
    )
    return r.content[0].text


def writing_agent(brief: str) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a professional writer. Write clear, engaging content based on the provided brief.",
        messages=[{"role": "user", "content": brief}],
    )
    return r.content[0].text


def fact_check_agent(content: str) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system="You are a fact-checker. Identify any suspicious claims in the text and flag them.",
        messages=[{"role": "user", "content": content}],
    )
    return r.content[0].text


TOOLS = [
    {
        "name": "research",
        "description": "Research a topic and return factual information.",
        "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    },
    {
        "name": "write",
        "description": "Write content based on a provided brief or research notes.",
        "input_schema": {"type": "object", "properties": {"brief": {"type": "string"}}, "required": ["brief"]},
    },
    {
        "name": "fact_check",
        "description": "Fact-check a piece of content and flag suspicious claims.",
        "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    """Route tool calls to the appropriate subagent."""
    if name == "research":
        return research_agent(inputs["topic"])
    elif name == "write":
        return writing_agent(inputs["brief"])
    elif name == "fact_check":
        return fact_check_agent(inputs["content"])
    return "Unknown tool."


def orchestrator(goal: str) -> str:
    """
    Hierarchical orchestrator that plans and delegates to subagents using tool use.
    """
    messages = [{"role": "user", "content": goal}]
    system = (
        "You are an orchestrator managing a team of specialist agents. "
        "Use the available tools to decompose the user's goal into tasks. "
        "Delegate to research, write, and fact_check agents as needed. "
        "Return a final polished result when done."
    )

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "Done."

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → Calling subagent: {block.name}({list(block.input.keys())})")
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    result = orchestrator(
        "Write a short, fact-checked blog post about the environmental impact of lithium-ion batteries."
    )
    print("\nFinal Output:\n", result)
```

---

### 4. Maker–Checker (Generator–Critic)

**Also known as:** reflective agents, self-refinement, generator–validator.

One agent generates output; a second (or the same agent in a different role) critically reviews it against defined criteria. The cycle continues until quality thresholds are met or a maximum number of iterations is reached. This pattern is a cornerstone of high-reliability agentic systems.

**When to use:**

- Tasks with strict quality, safety, or correctness requirements
- Code generation (generate → test/review → fix)
- Content that must meet stylistic or factual standards

**When to avoid:**

- Latency-sensitive tasks where one pass is sufficient
- Well-understood tasks where a single-agent prompt is reliable

#### Python Example

```python
from anthropic import Anthropic

client = Anthropic()
MAX_ITERATIONS = 3


def generate(task: str, previous_feedback: str = "") -> str:
    prompt = task
    if previous_feedback:
        prompt += f"\n\nPrevious attempt was rejected for the following reason:\n{previous_feedback}\nPlease revise."
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a Python developer. Write clean, well-documented Python code.",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


def check(code: str, criteria: str) -> dict:
    """
    Returns {"approved": bool, "feedback": str}
    """
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=(
            "You are a code reviewer. Evaluate the code strictly against the given criteria. "
            "Respond ONLY with a JSON object: {\"approved\": true/false, \"feedback\": \"...\"}"
        ),
        messages=[{"role": "user", "content": f"Criteria:\n{criteria}\n\nCode:\n{code}"}],
    )
    text = r.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    return json.loads(text)


import json

def maker_checker(task: str, criteria: str) -> str:
    """Run the maker-checker loop until approved or max iterations hit."""
    feedback = ""
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}: Generating...")
        draft = generate(task, feedback)
        print(f"Draft produced ({len(draft)} chars). Checking...")
        result = check(draft, criteria)
        if result.get("approved"):
            print("✓ Approved!")
            return draft
        feedback = result.get("feedback", "Did not meet criteria.")
        print(f"✗ Rejected: {feedback}")

    print("Max iterations reached. Returning last draft.")
    return draft


if __name__ == "__main__":
    task = "Write a Python function that reads a CSV file and returns a list of dicts."
    criteria = (
        "1. Uses the csv module from stdlib only (no pandas).\n"
        "2. Includes type hints.\n"
        "3. Has a docstring.\n"
        "4. Handles FileNotFoundError gracefully."
    )
    final = maker_checker(task, criteria)
    print("\nFinal Code:\n", final)
```

---

### 5. Router (Dynamic Dispatch)

**Also known as:** intent classification, task routing, conditional delegation.

A router agent classifies the incoming request and dynamically dispatches it to the most appropriate specialist agent. This avoids the overhead of engaging all agents on every task.

**When to use:**

- Systems with clearly delineated domains (e.g., billing vs. technical support vs. HR)
- When prompt complexity would be unreasonable for a single catch-all agent
- Gateways for multi-domain chatbots or assistants

**When to avoid:**

- Requests that genuinely require multiple domains simultaneously
- When routing classification itself is error-prone

#### Python Example

```python
from anthropic import Anthropic

client = Anthropic()

AGENTS = {
    "billing": "You are a billing specialist. Help users with invoices, payments, and subscriptions.",
    "technical": "You are a technical support engineer. Help users with bugs, errors, and product usage.",
    "sales": "You are a sales consultant. Help users understand pricing, plans, and upgrades.",
    "general": "You are a helpful general assistant for any queries not covered by specialists.",
}


def router(user_message: str) -> str:
    """Classify the intent and return the appropriate agent key."""
    valid = ", ".join(AGENTS.keys())
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16,
        system=(
            f"Classify the user message into exactly one category: {valid}. "
            "Reply with ONLY the category name, nothing else."
        ),
        messages=[{"role": "user", "content": user_message}],
    )
    category = r.content[0].text.strip().lower()
    return category if category in AGENTS else "general"


def dispatch(user_message: str) -> str:
    """Route the message and run the appropriate specialist agent."""
    agent_key = router(user_message)
    print(f"  → Routed to: {agent_key}")
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=AGENTS[agent_key],
        messages=[{"role": "user", "content": user_message}],
    )
    return r.content[0].text


if __name__ == "__main__":
    queries = [
        "My invoice from last month seems incorrect, I was charged twice.",
        "The app crashes whenever I try to export a PDF.",
        "What's the difference between your Pro and Enterprise plans?",
        "Can you recommend a good book?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        answer = dispatch(q)
        print(f"Answer: {answer[:200]}...")
```

---

### 6. Human-in-the-Loop (HITL)

**Also known as:** human escalation, approval gate, supervised autonomy.

At defined checkpoints, the system pauses and surfaces its current state to a human for review, approval, correction, or override. The human's input is injected back into the workflow before execution continues. This is essential for high-stakes or irreversible actions.

**When to use:**

- Actions that are irreversible (sending emails, executing financial transactions, deploying code)
- Compliance-sensitive workflows
- Early-stage deployments where trust in the agent is still being established

**When to avoid:**

- High-volume, low-risk tasks where human review creates bottlenecks
- Latency-sensitive pipelines

#### Python Example

```python
from anthropic import Anthropic

client = Anthropic()


def plan_action(goal: str) -> str:
    """Generate an action plan for the given goal."""
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=(
            "You are an automation agent. Given a goal, produce a numbered action plan "
            "of concrete steps you would execute. Be specific."
        ),
        messages=[{"role": "user", "content": goal}],
    )
    return r.content[0].text


def execute_action(step: str) -> str:
    """Simulate executing a single step (replace with real tool calls)."""
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system="You are an executor. Confirm you have completed the step and describe the outcome briefly.",
        messages=[{"role": "user", "content": f"Execute: {step}"}],
    )
    return r.content[0].text


def hitl_workflow(goal: str) -> None:
    """
    Human-in-the-loop: plan is shown to a human for approval before each step executes.
    """
    print(f"Goal: {goal}\n")
    plan = plan_action(goal)
    print("Proposed Action Plan:\n", plan)

    # Parse numbered steps (simple heuristic)
    steps = [line.strip() for line in plan.split("\n") if line.strip() and line.strip()[0].isdigit()]

    for step in steps:
        print(f"\n--- Pending Step ---\n{step}")
        approval = input("Approve this step? (y/n/edit): ").strip().lower()

        if approval == "n":
            print("Step skipped by human.")
            continue
        elif approval == "edit":
            step = input("Enter revised step: ").strip()

        result = execute_action(step)
        print(f"Result: {result}")

    print("\nWorkflow complete.")


if __name__ == "__main__":
    hitl_workflow("Send a campaign email to all users who haven't logged in for 30 days.")
```

---

## Pattern Comparison

|Pattern|Control Flow|Parallelism|Latency|Complexity|Best For|
|---|---|---|---|---|---|
|Sequential|Fixed linear|No|High (cumulative)|Low|Progressive refinement pipelines|
|Concurrent|Parallel + aggregate|Yes|Low (parallel)|Medium|Multi-perspective analysis|
|Hierarchical|Dynamic / planned|Optional|Medium–High|High|Complex, multi-domain goals|
|Maker–Checker|Iterative loop|No|Variable|Medium|Quality-critical generation|
|Router|Single dispatch|No|Low|Low|Domain routing / chatbots|
|HITL|Human-gated|No|Very High|Low–Medium|Irreversible / high-stakes actions|

---

## Key Frameworks in 2025–2026

The following frameworks are widely used for implementing these patterns in Python:

- **LangGraph** (LangChain) — Graph-based state machine for building stateful, multi-actor agent workflows. Benchmarks show it as the fastest orchestration framework with the most efficient state management.
- **CrewAI** — Role-based multi-agent framework modeled on human team collaboration.
- **AutoGen** (Microsoft) — Conversational multi-agent framework supporting group chat and code execution.
- **Google ADK** — Provides first-class `SequentialAgent`, `ParallelAgent`, and `LlmAgent` primitives.
- **LlamaIndex Workflows** — Event-driven orchestration with first-class RAG integration.
- **Anthropic SDK** — Native Python SDK used in all examples above; no framework required for custom orchestration logic.

---

## Best Practices

1. **Start simple.** Begin with a sequential chain or single agent with tools. Add complexity only when simpler approaches demonstrably fail.
2. **Instrument everything.** Log all agent inputs, outputs, tool calls, and handoffs. Distributed AI systems are hard to debug without traces.
3. **Set iteration limits.** All loops (maker–checker, agentic tool-call cycles) must have a maximum iteration cap to prevent infinite loops.
4. **Use testable interfaces.** Design each agent so it can be tested in isolation. Use LLM-as-judge or scoring rubrics for evaluating non-deterministic outputs.
5. **Design HITL checkpoints explicitly.** Identify exactly which steps require human approval, whether approval is mandatory or optional, and how human feedback re-enters the workflow.
6. **Manage context carefully.** Pass only the minimal, relevant state between agents. Bloated context windows increase cost and degrade quality.
7. **Prefer async for concurrent patterns.** Use `asyncio` and async clients to run parallel agent calls efficiently without blocking.
8. **Write immutable audit logs.** For every critical decision, record the input, prompt, output, and decision path for post-hoc review and compliance.

---

## References

- Microsoft Azure Architecture Center — _AI Agent Orchestration Patterns_ (February 2026)
- Google Developers Blog — _Developer's Guide to Multi-Agent Patterns in ADK_ (December 2025)
- arXiv 2511.15755 — _Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response_ (January 2026)
- arXiv 2601.13671 — _The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption_(January 2026)
- Anthropic SDK Documentation — https://docs.anthropic.com
- LangGraph Documentation — https://langchain.com/langgraph