# Tool integration

## 1. What Is Tool Integration?

At its core, tool integration is the mechanism that transforms an LLM from a text generator into an agent that can _act_. Tool calling (also called function calling) is the process where an LLM decides to invoke an external capability — for example, calling a weather API, running code, or querying a database. Without tool use, an LLM can only generate text. With it, it can search the web, query databases, send emails, create tickets, execute code, and interact with any system that exposes an API.

External tools and integrations act as the agent's enterprise toolkit, enabling it to interact with enterprise systems and perform tangible tasks. Without these integrations, the agent would merely be an AI-enabled conversational helper — nothing more than a chatbot.

---

## 2. How the Tool-Calling Loop Works

The agent loop consists of repeated cycles: the agent decides to take an **action** (call a tool), the environment returns a response, the agent processes it as an **observation**, and then decides whether to take another action or respond to the user.

The mechanics at the API level:

1. You define tools as structured JSON schemas with names, descriptions, and parameter definitions
2. You send the user message + tool definitions to the LLM
3. The LLM either responds in text or emits a structured `tool_use` block
4. Your code executes the tool and returns the result
5. The result is appended to conversation history and the loop continues

The loop continues until the agent responds with no tool calls. At each step, the LLM is called with the accumulated conversation history — made up of the initial user input and any previous LLM responses and tool outputs. If a function is called, the result is added to the conversation history; if no tool is called, the LLM response is returned.

---

## 3. Implementing Tool Integration in Python

### 3.1 Defining a Tool Schema (Anthropic/Claude)

The two required components for any tool are the **function implementation** and the **tool schema**. The tool schema is a structured description of the tool — it tells the LLM what the tool does, when to use it, and what parameters it takes.

```python
import anthropic
import json

client = anthropic.Anthropic()

# --- Tool implementation ---
def get_stock_price(ticker: str) -> dict:
    """Fake stock price lookup — replace with real API call."""
    prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 420.10}
    price = prices.get(ticker.upper())
    if price is None:
        return {"error": f"Ticker '{ticker}' not found."}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}

def run_python_code(code: str) -> str:
    """Execute sandboxed Python and return stdout."""
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return output.getvalue() or "(no output)"
    except Exception as e:
        return f"Error: {e}"

# --- Tool schemas ---
tools = [
    {
        "name": "get_stock_price",
        "description": "Retrieves the current price for a stock ticker symbol. Use this when the user asks about stock prices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL, GOOGL."
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "run_python_code",
        "description": "Executes Python code in a sandboxed environment and returns stdout. Use for calculations or data transformations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Valid Python code to execute."}
            },
            "required": ["code"]
        }
    }
]

# --- Tool dispatcher ---
TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "run_python_code": run_python_code,
}

def dispatch_tool(name: str, inputs: dict):
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**inputs)
```

### 3.2 The Agentic Loop

```python
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Final text response
            for block in response.content:
                if block.type == "text":
                    return block.text

        elif response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"[Tool Call] {block.name}({block.input})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"[Tool Result] {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            break

# Example run
answer = run_agent("What is the price of Apple stock? Then calculate 189.50 * 100.")
print(answer)
```

---

## 4. Pydantic-Based Tool Registration (Production Pattern)

Rather than hand-writing JSON schemas, use Pydantic to generate them automatically — this is the pattern used by LangChain, OpenAI Agents SDK, and others.

```python
from pydantic import BaseModel, Field
from typing import Callable
import inspect, json

class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    fn: Callable

    class Config:
        arbitrary_types_allowed = True

def tool(description: str):
    """Decorator to register a Python function as an agent tool."""
    def decorator(fn: Callable) -> ToolSpec:
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            prop = {"type": type_map.get(annotation, "string")}
            # Pull description from docstring heuristic
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            properties[param_name] = prop

        schema = {
            "name": fn.__name__,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
        return ToolSpec(name=fn.__name__, description=description,
                        input_schema=schema["input_schema"], fn=fn)
    return decorator

# Usage
@tool("Fetch current weather for a city. Returns temperature in Fahrenheit.")
def get_weather(city: str, units: str) -> dict:
    return {"city": city, "temp_f": 78.0, "condition": "sunny"}

print(json.dumps(get_weather.input_schema, indent=2))
```

---

## 5. The Model Context Protocol (MCP)

MCP is the emerging industry standard that replaces ad-hoc tool integrations. MCP is an open standard for connecting AI assistants to the systems where data lives — content repositories, business tools, and development environments. It provides a universal, open standard for connecting AI systems with data sources, replacing fragmented integrations with a single protocol.

MCP follows a client-server model. The MCP client is typically the AI agent or application that needs information, and the MCP server is a lightweight connector exposing a specific data source or service via the MCP standard. Developers can run multiple MCP servers — one for each repository, database, API, or tool — and the agent can connect to all of them through a unified protocol.

This design cleanly separates the AI's reasoning from the integration logic, allowing the agent to focus on _what_ to request, while the MCP servers handle _how_ to fulfill it.

A minimal Python MCP server looks like:

```python
# mcp_server_example.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

app = Server("my-data-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_database",
            description="Run a read-only SQL query against the products database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT statement."}
                },
                "required": ["sql"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_database":
        sql = arguments["sql"]
        # Execute against your DB here
        results = [{"id": 1, "name": "Widget", "price": 9.99}]
        return [TextContent(type="text", text=json.dumps(results))]
```

MCP has arguably brought agentic AI into the mainstream much faster than the industry expected — by making it easier for developers to connect agents to many different sources of data, it's now possible to provide agentic systems with detailed and rich context that would otherwise require significant time and investment.

---

## 6. Tool Design: The Most Impactful Engineering Decision

The quality of your tool descriptions is the single biggest factor in whether your agent uses tools correctly. Poor descriptions lead to wrong tool selection, bad parameter inference, and hallucinated arguments.

**Good vs. bad tool descriptions:**

```python
# BAD - vague, no guidance on when to use it
{
    "name": "search",
    "description": "Searches for things.",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
}

# GOOD - tells the model exactly when and how to use it
{
    "name": "search_products",
    "description": (
        "Search the product catalog by keyword. Use this when the user asks "
        "about specific products, wants to find items by name or category, "
        "or is comparison shopping. Do NOT use for order status or account info. "
        "Returns a list of up to 10 matching products with IDs, names, and prices."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords. Use natural language — e.g. 'blue running shoes under $100'."
            },
            "max_results": {
                "type": "integer",
                "description": "Max products to return. Defaults to 5. Max is 10.",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

---

## 7. Best Practices

### Tool Design

- **One tool, one responsibility.** Avoid tools that do multiple things — the model can't reason about ambiguous tools.
- **Name tools like verbs**: `search_products`, `create_ticket`, `send_email` — not `products`, `ticket`, `email`.
- **Document failure modes** in the description: "Returns an error dict if the ticker is not found."
- Standardize tool interfaces, parameter definitions, capability descriptions, and usage constraints across your system. This ensures that tools developed across diverse organizational units maintain consistent structural patterns and semantic clarity.

### Validation & Security

- Validate every tool call against its JSON Schema before execution. Libraries like Pydantic make this straightforward. Also perform authorization checks — verify that the tool call is permitted given the current user's permissions.
- Rate limit your tools to prevent agents from making excessive tool calls. A bug or adversarial input could cause an infinite tool-calling loop that racks up costs or overwhelms downstream services.
- Tool results should be sanitized before being sent back to the model, especially if they contain user-generated content that could constitute a prompt injection.
- Enforce least-privilege principles across all tool integrations. An agent that should only read from a calendar should not gain write permissions across your entire environment.

### Reliability & Observability

- Always log tool calls with inputs, outputs, latency, and errors. This is your primary debugging surface.
- Use pre-defined metrics like tool selection accuracy, tool parameter accuracy, and multi-turn function call accuracy to systematically evaluate your agent's capability to correctly identify appropriate tools and populate parameters with accurate values.
- Implement **retry logic with exponential backoff** for transient API failures.
- Return structured error dicts from tools rather than raising exceptions — let the LLM decide how to recover.

```python
# Resilient tool wrapper pattern
import time
from functools import wraps

def resilient_tool(max_retries=3, backoff_base=1.5):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    wait = backoff_base ** attempt
                    print(f"[Tool Error] {fn.__name__} attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                    time.sleep(wait)
            return {"error": f"Tool '{fn.__name__}' failed after {max_retries} attempts: {last_error}"}
        return wrapper
    return decorator

@resilient_tool(max_retries=3)
def call_external_api(endpoint: str) -> dict:
    # ... real HTTP call here
    pass
```

### Tool Chaining & State

- For reliable chaining of tool calls, implement state management between steps. Store intermediate results in a session object that persists across the agent loop. This prevents the agent from losing context between steps and allows you to resume chains that are interrupted by errors or timeouts.

```python
# Stateful agent session example
from dataclasses import dataclass, field

@dataclass
class AgentSession:
    user_id: str
    conversation: list = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)  # keyed by tool call id
    tool_call_count: int = 0
    max_tool_calls: int = 25  # hard ceiling to prevent runaway agents

    def can_call_tool(self) -> bool:
        return self.tool_call_count < self.max_tool_calls

    def record_tool_call(self, tool_name: str, inputs: dict, result):
        self.tool_call_count += 1
        self.intermediate_results[f"{tool_name}_{self.tool_call_count}"] = {
            "inputs": inputs,
            "result": result
        }
```

### Dynamic Tool Loading

- In production systems, rather than loading all tools at the start, dynamically adjust the tool list at each step of the workflow based on the user's permissions, conversation context, or current workflow phase. This reduces token overhead and prevents the model from being confused by irrelevant tools.

```python
def get_tools_for_context(user_role: str, current_step: str) -> list:
    base_tools = [search_tool, calculator_tool]
    if user_role == "admin":
        base_tools.append(delete_record_tool)
    if current_step == "checkout":
        base_tools.append(payment_tool)
    return base_tools
```

---

## 8. Tool Categories Reference

|Category|Examples|Notes|
|---|---|---|
|**Data retrieval**|Database queries, vector search, API GET calls|Most common; usually read-safe|
|**Code execution**|Python sandbox, shell commands|High risk; require sandboxing|
|**External services**|Email, Slack, CRM, calendar|Require auth & rate limiting|
|**File I/O**|Read/write local or cloud files|Scope carefully; path traversal risk|
|**Web/Search**|Web search, page fetch|Output sanitization critical|
|**Agent handoff**|Delegate to specialized sub-agent|Core to multi-agent patterns|
|**Memory**|Write to/read from long-term store|Enables statefulness across sessions|

---

## 9. MCP Security Considerations

Security researchers have identified multiple outstanding security issues with MCP, including prompt injection, tool permissions that allow for combining tools to exfiltrate data, and lookalike tools that can silently replace trusted ones.

Key mitigations:

- **Vet MCP servers** before connecting — treat third-party MCP servers like third-party code dependencies
- **Explicit consent flows** before any tool that writes or deletes data
- Implement appropriate access controls and data protections. Hosts must obtain explicit user consent before invoking any tool, and tools must be treated with appropriate caution since they represent arbitrary code execution.
- Audit log every tool invocation with the full input/output for compliance and forensics

---

## 10. Evaluation

Amazon built golden datasets for regression testing of tool use, generated synthetically using LLMs from historical API invocation logs. Using metrics like tool selection accuracy, tool parameter accuracy, and multi-turn function call accuracy, their teams systematically evaluate the agent's capability to correctly identify appropriate tools, populate parameters with accurate values, and maintain coherent tool invocation sequences across conversations.

Your evaluation checklist for any tool-using agent:

- **Tool selection accuracy** — did the agent choose the right tool?
- **Parameter accuracy** — were the inputs to the tool correctly extracted?
- **Recovery behavior** — does the agent handle tool errors gracefully?
- **Loop termination** — does the agent know when to stop calling tools?
- **Parallel call efficiency** — can it call independent tools simultaneously?

---

## Summary

Tool integration is the foundational capability that makes agents genuinely useful. The core loop is simple — define schemas, dispatch calls, feed results back — but production quality requires: precise tool descriptions, robust validation, security at every boundary, stateful session management, and rigorous evaluation. MCP is rapidly becoming the standard for enterprise tool connectivity, so it's worth investing in understanding its architecture even if you start with direct function calling.