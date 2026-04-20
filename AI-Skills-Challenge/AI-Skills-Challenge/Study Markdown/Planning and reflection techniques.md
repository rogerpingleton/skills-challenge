
# Planning and reflection techniques

## What Are They and Why Do They Matter?

As opposed to zero-shot prompting of a large language model, agents allow for more complex interaction and orchestration. In particular, agentic systems have a notion of planning, loops, reflection, and other control structures that heavily leverage the model's inherent reasoning capabilities to accomplish a task end-to-end.

What makes workflows truly agentic are the iteration and feedback loops built into the process. Instead of generating output in a single pass, agentic workflows involve cycles where the agent takes an action, observes the result, and uses that observation to inform the next action. This mirrors how humans actually solve complex problems — we try something, see what happens, learn from the result, and adjust our approach.

As an AI Engineer, these techniques are the core cognitive machinery you're wiring together when you build agents. They determine how your agent thinks before acting, how it recovers from failure, and how it avoids getting stuck.

---

## The Cognitive Loop: PRAR

Before diving into individual techniques, understand the loop they all operate within. Agentic reasoning frameworks define how digital agents think within the Perception → Reasoning → Action → Reflection (PRAR) loop. Each pattern introduces a unique "thinking rhythm."

All planning and reflection techniques are specializations of this loop.

---

## Planning Techniques

### 1. Chain-of-Thought (CoT)

The simplest and most foundational technique. You prompt the model to verbalize its intermediate reasoning steps before producing a final answer.

Chain-of-thought prompting works by explicitly asking the cognitive agent to show its work or explain its reasoning as it progresses towards a solution. This step-by-step breakdown not only improves the accuracy of the final answer but also provides insight into the model's decision-making process. One of the key advantages of this technique is its ability to enhance the reliability and verifiability of AI-generated responses — by exposing the intermediate steps, it becomes easier to identify errors, logical flaws, or alternative approaches.

**When to use it:** Simple, single-turn tasks where you want structured reasoning without multi-step tool calls. Best for math, logic, classification, and analysis tasks.

**Python example:**

```python
from anthropic import Anthropic

client = Anthropic()

def chain_of_thought(question: str) -> str:
    system = """You are a reasoning agent. Always think step-by-step.
    Format your response as:
    Reasoning:
    1. [step]
    2. [step]
    ...
    Final Answer: [answer]"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

result = chain_of_thought(
    "A company's revenue grew 20% in Q1 and then dropped 15% in Q2. "
    "If Q4 last year was $1M, what is Q2 this year's revenue?"
)
print(result)
```

---

### 2. ReAct (Reasoning + Acting)

The ReAct framework was a pivotal innovation that granted LLM agents the capability to engage dynamically with the external world. The agent first generates an internal thought — verbalized reasoning where it breaks down the primary task, identifies missing information, and plans its next step. Based on this thought, the agent takes an action, typically issuing a command to an external tool. The external tool executes the command and returns an observation. The agent then uses this new observation to refine its reasoning and determine the next thought-action cycle. This continuous cycle allows the agent to gather information iteratively, correct mistakes, and ultimately produce a well-supported, factually accurate final answer.

**When to use it:** Any task requiring real-world interaction — web search, database queries, API calls, file I/O, code execution.

**Python example (minimal ReAct loop):**

```python
import anthropic
import json

client = anthropic.Anthropic()

def web_search(query: str) -> str:
    # Stub — replace with real search
    return f"Search results for '{query}': [result 1, result 2]"

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

tool_map = {"web_search": web_search, "calculator": calculator}

def react_agent(user_question: str, max_iterations: int = 5) -> str:
    messages = [{"role": "user", "content": user_question}]
    
    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "end_turn":
            # Extract final text answer
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = tool_map[block.name](**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break
    
    return "Max iterations reached."

answer = react_agent("What is the square root of the number of days in a leap year?")
print(answer)
```

---

### 3. Plan-and-Execute (Two-Phase Planning)

Plan-and-Execute is the architectural opposite of ReAct. Instead of reasoning through every step, the agent plans the entire strategy first, then executes sequentially. This approach mirrors human project management: define goals → outline subtasks → perform each one in order.

A dedicated Planner module emits a single global plan (often as a DAG), which is dispatched by an Executor. All tool dependencies and orderings are resolved up-front. This variant decouples plan generation from execution, enabling efficient scheduling, predictability, and architectural security.

**When to use it:** Long-horizon tasks with many predictable subtasks (research pipelines, document processing workflows, code generation projects). Also good when you want human review of the plan before execution begins.

**Python example:**

```python
from anthropic import Anthropic
from dataclasses import dataclass

client = Anthropic()

@dataclass
class Plan:
    steps: list[str]
    
def generate_plan(goal: str) -> Plan:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a planning agent. Given a goal, output a numbered 
        JSON list of concrete steps to accomplish it. 
        Respond ONLY with valid JSON like: {"steps": ["step1", "step2", ...]}""",
        messages=[{"role": "user", "content": f"Goal: {goal}"}]
    )
    data = json.loads(response.content[0].text)
    return Plan(steps=data["steps"])

def execute_step(step: str, context: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are an execution agent. Complete the given step using prior context.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nStep to execute: {step}"}]
    )
    return response.content[0].text

def plan_and_execute(goal: str) -> dict:
    plan = generate_plan(goal)
    print(f"Plan generated with {len(plan.steps)} steps:\n")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")
    
    context = ""
    results = []
    for i, step in enumerate(plan.steps, 1):
        print(f"\nExecuting step {i}: {step}")
        result = execute_step(step, context)
        results.append({"step": step, "result": result})
        context += f"\nStep {i} ({step}): {result}"
    
    return {"plan": plan.steps, "results": results}

output = plan_and_execute(
    "Write a technical blog post outline about vector databases for Python developers"
)
```

---

### 4. Tree of Thoughts (ToT)

Tree of Thoughts (ToT) is a reasoning framework that allows agents to explore multiple ideas or solution paths simultaneously, evaluate them, and converge on the best option. Think of it as an AI "brainstorming tree" where each branch represents a possible thought or decision path.

Tree-of-Thoughts becomes valuable when early decisions significantly constrain later possibilities. It's computationally expensive but finds better solutions on hard problems.

Graph-of-Thoughts (GoT) takes the tree structure further, allowing for arbitrary connections and transformations between thoughts. Instead of a strict hierarchy, GoT enables thoughts to be aggregated (combining the best parts of several ideas) or refined in iterative loops. This non-linear, flexible structure is ideal for highly complex synthesis tasks and creative problem-solving, where ideas need to be merged and refined over time.

**When to use it:** Strategic planning, architecture decisions, creative problem-solving, any problem where exploring alternatives matters and you can afford the extra LLM calls.

**Python example (simplified ToT):**

```python
from anthropic import Anthropic

client = Anthropic()

def generate_thoughts(problem: str, n: int = 3) -> list[str]:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"""Generate {n} distinct approaches to solve the problem. 
        Return as JSON: {{"thoughts": ["approach1", "approach2", ...]}}""",
        messages=[{"role": "user", "content": problem}]
    )
    data = json.loads(response.content[0].text)
    return data["thoughts"]

def score_thought(problem: str, thought: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system="""Score this approach to the problem from 0.0 to 1.0 based on 
        feasibility, completeness, and quality. 
        Return ONLY JSON: {"score": 0.0, "reason": "..."}""",
        messages=[{"role": "user", "content": f"Problem: {problem}\nApproach: {thought}"}]
    )
    data = json.loads(response.content[0].text)
    return data["score"], data["reason"]

def tree_of_thoughts(problem: str, breadth: int = 3, depth: int = 2) -> str:
    current_best = None
    current_score = -1
    
    for d in range(depth):
        thoughts = generate_thoughts(
            problem if d == 0 else f"{problem}\n\nBest so far: {current_best}",
            n=breadth
        )
        for thought in thoughts:
            score, reason = score_thought(problem, thought)
            print(f"  Score {score:.2f}: {thought[:60]}... ({reason})")
            if score > current_score:
                current_score = score
                current_best = thought
    
    return current_best

best_approach = tree_of_thoughts(
    "Design a rate limiting strategy for a public API serving 10k req/sec",
    breadth=3,
    depth=2
)
print(f"\nBest approach: {best_approach}")
```

---

### 5. Language Agent Tree Search (LATS)

LATS is a single-agent method that synergizes planning, acting, and reasoning by using trees. This technique, inspired by Monte Carlo Tree Search, represents a state as a node and taking an action as traversing between nodes. It uses LM-based heuristics to search for possible options, then selects an action using a state evaluator. LATS implements a self-reflection reasoning step that dramatically improves performance. When an action is taken, both environmental feedback as well as feedback from a language model is used to determine if there are any errors in reasoning and propose alternatives.

This is ToT + ReAct + Reflection fused together. It's the most powerful single-agent technique but also the most expensive.

---

## Reflection Techniques

Reflection is the mechanism by which an agent evaluates its own outputs and reasoning — enabling self-correction without retraining.

### 1. Basic Reflection (Generate → Critique → Revise)

Reflection patterns enable AI agents to examine their own reasoning processes, evaluate their performance, and improve their future decision-making. The cycle is: Generate (the agent produces an initial response), Reflect (the agent examines its own output critically), Regenerate (the agent produces an improved version incorporating lessons learned from the self-evaluation).

**Python example:**

```python
from anthropic import Anthropic

client = Anthropic()

def reflect_and_revise(task: str, max_rounds: int = 2) -> str:
    # Initial generation
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": task}]
    )
    output = response.content[0].text
    
    for round_num in range(max_rounds):
        # Critique phase
        critique_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system="""You are a critical reviewer. Identify specific weaknesses, 
            errors, or missing elements in the response. Be concrete and actionable.
            Format: {"issues": ["issue1", "issue2"], "score": 0-10}""",
            messages=[{
                "role": "user",
                "content": f"Task: {task}\n\nResponse to critique:\n{output}"
            }]
        )
        critique = json.loads(critique_response.content[0].text)
        print(f"Round {round_num+1} score: {critique['score']}/10")
        
        if critique["score"] >= 9:
            print("Quality threshold met, stopping.")
            break
        
        # Revision phase
        issues_text = "\n".join(f"- {issue}" for issue in critique["issues"])
        revise_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="Revise the response to fix the identified issues.",
            messages=[{
                "role": "user",
                "content": f"Task: {task}\n\nPrevious response:\n{output}\n\nIssues to fix:\n{issues_text}"
            }]
        )
        output = revise_response.content[0].text
    
    return output

result = reflect_and_revise(
    "Explain the tradeoffs between SQL and NoSQL databases for a high-traffic e-commerce app."
)
```

---

### 2. Reflexion (Memory-Based Reflection Across Attempts)

Reflexion introduces a different paradigm based on episodic memory and self-reflection. After each attempt at a task, the agent generates a verbal critique of its performance and stores that reflection for future trials. The architecture consists of three components: an Actor that generates reasoning traces and actions, an Evaluator that scores the trajectory's quality, and a Self-Reflection module that analyzes failures and produces concrete guidance for improvement. When the agent fails at a task, it doesn't just restart with the same approach — it explicitly reflects on what went wrong, generates actionable feedback, and uses that insight in subsequent attempts.

This is especially powerful for tasks with automated evaluators (unit tests, API validators, rule checkers).

**Python example:**

```python
from anthropic import Anthropic
from dataclasses import dataclass, field

client = Anthropic()

@dataclass
class ReflexionAgent:
    task: str
    evaluator_fn: callable  # Returns (passed: bool, feedback: str)
    memory: list[str] = field(default_factory=list)
    max_attempts: int = 4
    
    def _build_system_prompt(self) -> str:
        base = "You are a coding agent. Write correct Python code for the task."
        if self.memory:
            reflections = "\n".join(f"- {m}" for m in self.memory)
            base += f"\n\nPast attempt reflections (do NOT repeat these mistakes):\n{reflections}"
        return base
    
    def _reflect(self, attempt: str, feedback: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system="Analyze why this attempt failed. Be specific and concise (1-2 sentences).",
            messages=[{
                "role": "user",
                "content": f"Task: {self.task}\nAttempt:\n{attempt}\nFailure reason: {feedback}"
            }]
        )
        return response.content[0].text
    
    def run(self) -> str | None:
        for attempt_num in range(self.max_attempts):
            print(f"\nAttempt {attempt_num + 1}/{self.max_attempts}")
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": self.task}]
            )
            code = response.content[0].text
            
            passed, feedback = self.evaluator_fn(code)
            print(f"  Passed: {passed} | Feedback: {feedback}")
            
            if passed:
                return code
            
            # Reflect on failure and store in memory
            reflection = self._reflect(code, feedback)
            self.memory.append(reflection)
            print(f"  Reflection: {reflection}")
        
        return None  # All attempts failed

# Example evaluator (unit test based)
def evaluate_code(code: str) -> tuple[bool, str]:
    try:
        namespace = {}
        exec(code, namespace)
        fn = namespace.get("fibonacci")
        assert fn(0) == 0
        assert fn(1) == 1
        assert fn(10) == 55
        return True, "All tests passed"
    except AssertionError as e:
        return False, f"Test assertion failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"

agent = ReflexionAgent(
    task="Write a Python function `fibonacci(n)` that returns the nth Fibonacci number.",
    evaluator_fn=evaluate_code,
    max_attempts=4
)
result = agent.run()
```

---

### 3. Self-Ask (Recursive Sub-question Decomposition)

This pattern involves the agent asking itself clarifying sub-questions before answering — similar to Socratic reasoning. "What is the user asking?" "What information do I need to answer this?" "How do I verify it?" This recursive questioning leads to more logical and transparent answers.

This is great for research-heavy tasks or when the agent must handle complex multi-part questions where hidden assumptions can cause failures.

---

### 4. Critic-Refine (Separate Critic Agent)

Rather than one agent critiquing itself, you spin up a dedicated critic model. After producing an output, the Critic module reviews it and either approves or requests refinement. This loop powers advanced hybrid frameworks like Reflexion or Critic-Refine architectures.

In multi-agent systems this is implemented as a reviewer agent role — a separate system prompt, sometimes a different model, that only evaluates and never produces the primary output.

---

## Choosing the Right Technique

The right planning approach depends on your problem's characteristics. Simple, deterministic tasks often need nothing more than direct prompting or chain-of-thought — the computational overhead of sophisticated planning would waste resources. ReAct excels when tasks require real-world interaction and feedback. Tree-of-Thoughts becomes valuable when early decisions significantly constrain later possibilities.

Here's a practical decision guide:

|Scenario|Recommended Technique|
|---|---|
|Single-turn reasoning, math, logic|Chain-of-Thought|
|Tool use, API calls, search|ReAct|
|Long task with predictable subtasks|Plan-and-Execute|
|Creative/strategic problem with many options|Tree of Thoughts|
|Iterative tasks with automated eval (e.g., code)|Reflexion|
|High-stakes long-horizon tasks|LATS (ReAct + ToT + Reflection)|
|Structured multi-part questions|Self-Ask|
|Content quality assurance|Critic-Refine|

---

## Combining Techniques

These patterns are not mutually exclusive. The most sophisticated agent systems often combine multiple patterns to achieve their goals.

A common production pattern is **Plan-and-Execute + ReAct + Reflexion**:

1. A **Planner** decomposes the goal into subtasks (Plan-and-Execute).
2. Each subtask is executed by a **ReAct** agent with tools.
3. After each subtask, a **Reflexion** step evaluates the result and updates a running memory of what worked and what didn't.
4. A **Critic** agent reviews the final assembled output.

---

## Key Engineering Pitfalls

In multi-step reasoning loops like ReAct, a single error in an early step propagates downstream, leading to "cascading failures." Future work must focus on robust error-recovery mechanisms to validate reasoning steps before execution occurs. Autonomous agents frequently suffer from getting stuck in repetitive loops, continuously retrying a failed action without modifying their strategy.

As an AI Engineer, guard against:

- **Always set `max_iterations`** on any ReAct loop — infinite loops are a real failure mode.
- **Reflection without memory is useless** — the reflection output must be injected into the next prompt, not discarded.
- **Token cost compounds** with every loop — ToT and LATS can be 10–20x more expensive than a single-pass call; use them only when the task justifies it.
- **Evaluator quality determines Reflexion quality** — if your evaluator gives vague feedback ("it failed"), your reflections will also be vague. Invest in precise evaluators.
- **Cascading errors** — validate intermediate outputs in Plan-and-Execute pipelines rather than passing bad state forward.

---

These techniques form the cognitive backbone of modern agentic systems. In practice, you'll often start with CoT or ReAct for most tasks, then reach for Plan-and-Execute or Reflexion when you hit reliability walls on longer-horizon workflows.