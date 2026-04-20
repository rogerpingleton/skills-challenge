
# Tradeoffs - cost vs performance vs licensing

## Foundation Models & Model Selection: An AI Engineer's Guide

### What is a Foundation Model?

A foundation model is a large, pre-trained model (GPT-4o, Claude Sonnet, Llama 4, Gemini, Mistral, DeepSeek, etc.) that can be adapted to a wide range of downstream tasks via prompting, RAG, or fine-tuning. As an AI engineer, your job is not to _build_ these models — it's to build _applications_ around them. The mental model shift is: from "how do I build a model" to "how do I ship a product powered by a model."

By 2026, engineers need to select and combine diverse models — not just LLMs like GPT, Claude, Gemini, Mistral, or Llama, but also vision, audio, reasoning agents, and tool-augmented models, choosing based on factors such as cost, latency, throughput, memory constraints, and task specialization.

---

### 1. The Core Decision: Proprietary (API) vs. Open-Weight vs. Open-Source

This is the first and most consequential choice you'll make.

**Proprietary / Closed Models** (OpenAI GPT-5, Claude, Gemini)

These are accessed via API. You pay per token and never touch the weights.

- **Pros:** Best-in-class performance for frontier tasks, fast time-to-market, no infrastructure to manage, IP indemnification from the provider
- **Cons:** Higher costs accumulate quickly at scale, limited customizability to protect IP, and deep integration risks vendor lock-in with substantial financial and technical hurdles for migration
- **Data risk:** If you're not focused on protecting your data, you may inadvertently enter into agreement terms allowing some use of it to train third-party models. For companies considering inputting very sensitive company information into a third-party model, it is critical to carefully evaluate these risks.

**Open-Weight Models** (Llama 4, Mistral, Qwen 3, DeepSeek R1, Grok)

Weights are released publicly — you can download, self-host, and fine-tune.

- **Pros:** No API dependency, no token meter, no vendor lock-in. You can download weights, run locally, fine-tune on proprietary data, and deploy offline.
- **Cons:** You own the infrastructure cost and operational burden. Requires ML/DevOps expertise.
- **License complexity:** Open weights models release the weights and parameters but typically not the underlying training data or training algorithms. Other AI models considered open weights include Mistral's Pixtral Large, xAI's Grok-1, and Alibaba's Qwen 3.

**True Open-Source Models**

Existing open-source software licenses like MIT and Apache-2.0 weren't designed with AI models in mind, while most model-specific licenses are either too complex, overly restrictive, or legally ambiguous. The list of models meeting the OSI's full open-source AI definition (weights + training data + code) remains short.

**The Hybrid Approach (What Most Enterprises Actually Do)**

A hybrid approach is winning with enterprises: they look to closed-source frontier models for the most sophisticated applications and open-source smaller models for edge and specialized use cases.

---

### 2. Cost vs. Performance

This is the most immediate engineering tradeoff and it has several layers.

**Token-based API pricing** scales dangerously. A feature that looks cheap in a demo can become expensive in production once you factor in:

- Input + output token costs (e.g., long system prompts repeated on every call)
- Context window utilization
- Number of calls per user session
- Agentic loops where one user action triggers many model calls

**The open-source cost shift:** Open-source models shift costs from API consumption to infrastructure — hardware requirements for running state-of-the-art models require significant GPU resources. This can be cheaper at scale, but it requires upfront capital and operational expertise.

**Real-world example:** Midjourney reduced monthly spend from $2.1M to under $700K by moving to TPU v6e infrastructure, achieving $16.8M in annualized savings. Fine-tuned LoRA adapters can nearly double accuracy over base models, and fine-tuned smaller models often match or exceed larger models on specific tasks.

**Practical cost strategies:**

- **Model routing:** Route simple queries to cheap/fast models (e.g., Haiku, Gemini Flash) and complex ones to powerful models (GPT-4o, Claude Opus). This is now a standard architectural pattern.
- **Prompt caching:** Many providers offer significant discounts for cached prefix tokens — essential if you have long, repeated system prompts.
- **Quantized open-weight models:** Running a quantized Llama 4 70B locally is often cheaper at high volume than paying per-token at frontier prices.

---

### 3. Latency

Latency is a product quality issue, not just a performance metric. A model that's 20% more accurate but 4x slower can make a product unusable.

Key concepts:

- **Time to First Token (TTFT):** How quickly the response starts streaming. Critical for chat UIs.
- **Tokens per second (throughput):** How fast the full response completes.
- **Context window size:** Larger context = slower inference. Filling a 200K token window is very different in latency profile from a 4K window.

A 5-second retrieval plus a 3-second inference time doesn't make a usable product. In agentic systems, this compounds — a 5-step agent chain where each step takes 3 seconds means a 15-second wait before any result.

**Tradeoff:** Smaller models are faster and cheaper but less capable. The art is finding the smallest model that's _good enough_for your task, not the best model possible.

---

### 4. Licensing in Depth

Licensing is often overlooked by engineers until it becomes a legal problem.

|License Type|Examples|Commercial Use|Fine-tune & Redistribute|IP Indemnity|
|---|---|---|---|---|
|Proprietary API|GPT-5, Claude, Gemini|Yes (per ToS)|No|Often yes (limited)|
|Apache 2.0|Qwen 3, Grok-1, Mistral|Yes|Yes|No|
|MIT|DeepSeek R1|Yes|Yes|No|
|Custom (Llama)|Llama 4|Yes (<700M MAU)|Yes|No|
|Copyleft (GPL)|Some older models|Restricted|Derivative must be open|No|

**Watch out for:**

- **Usage caps:** Meta's Llama license restricts commercial use for companies above 700M monthly active users. Fine for startups; potentially an issue for large enterprises.
- **No indemnity on open models:** Several well-known open AI models have been released under permissive licenses that include a disclaimer of warranty, limitation of liability, and no indemnity — significantly worse protection than what proprietary models typically offer.
- **Geopolitical risk:** DeepSeek is trained by a Chinese lab. Several US federal agencies and defense contractors have restricted or banned its use, regardless of the MIT license. This is an emerging compliance dimension for regulated industries.

---

### 5. Context Window

Context window size determines what you can fit in a single inference call — system prompt, conversation history, retrieved documents, tool outputs, etc.

- GPT-4o, Claude Sonnet: 200K tokens
- Gemini 2.0 Flash: 1M tokens
- Llama 4 Scout: 10M tokens (experimental)
- Mistral 7B: 32K tokens

Larger windows don't mean "more is better" — long contexts introduce **lost-in-the-middle** problems where models attend poorly to content in the middle of a very long context. They also dramatically increase latency and cost. Context window size is a ceiling, not a recommendation.

---

### 6. Multimodal Capabilities

Many production systems now need to handle more than text. Your model selection needs to account for:

- **Vision:** Can it process images, screenshots, PDFs, charts?
- **Audio:** Can it transcribe or reason about audio?
- **Structured output:** Does it reliably produce JSON/XML for downstream parsing?
- **Code:** Does it support tool use, code execution, or function calling?

Gemini's multi-modal architecture provides significant advantages for applications involving multi-modal data like robotics, visual analysis, or integrated AI systems, but it also introduces additional complexity in deployment and maintenance.

---

### 7. Safety, Alignment & Output Reliability

Different models have different safety profiles, which matters for your product.

Claude emphasizes safety and user alignment, often producing more conservative outputs that avoid controversial or harmful content. While this enhances trust and reduces risk, it can sometimes limit creative or open-ended responses — a tradeoff in scenarios requiring bold or innovative language.

For production systems, you care about:

- **Hallucination rate:** How often does the model confidently produce false information?
- **Instruction following:** Does it reliably follow complex, structured prompts?
- **Refusal behavior:** Does it refuse too aggressively (breaking legitimate use cases) or too permissively (creating safety incidents)?
- **Output consistency:** Does it produce stable structured output (JSON) reliably, or does it frequently malformat?

---

### 8. Data Privacy & Sovereignty

This is increasingly a first-class concern, especially in healthcare, legal, finance, and government.

On-premises or open-source deployment is needed if you're working with private data, air-gapped networks, or need compliance (HIPAA, SOC 2, FedRAMP).

Considerations:

- **Cloud API:** Data leaves your infrastructure. Who trains on it? Is there a DPA (Data Processing Agreement)?
- **Self-hosted open weights:** Data stays on your infrastructure. Harder to operate, but full sovereignty.
- **Regulatory requirements:** EU's AI Act now requires transparency, safety assessments, and documentation of model capabilities. US executive orders mandate risk disclosures and AI auditing for federal contractors using high-capacity models.

---

### 9. Vendor Lock-in & Portability

This is a long-term architectural risk. If your application is tightly coupled to one provider's API quirks, prompt formats, or proprietary features, switching is expensive.

Mitigation strategies:

- Abstract your model calls behind a provider-agnostic interface (e.g., LiteLLM, LangChain's model abstraction layer).
- Use standard formats for prompt templates so they're portable.
- Maintain your own evaluation suite so you can benchmark a new model against your actual tasks quickly.

---

### 10. Model Selection Workflow in Practice

A sound model selection workflow includes: evaluating all components in a system, creating an evaluation guideline, and defining evaluation methods and data — not just relying on public benchmarks.

**Don't trust leaderboards blindly.** Public benchmarks like MMLU or HumanEval measure general capability. Your task is specific. A model that ranks #3 on a benchmark may outperform the #1 model on your exact use case.

A practical selection process:

1. **Define your task clearly** — summarization, extraction, code generation, agentic reasoning, etc.
2. **Build a golden dataset** of 50–200 real examples with expected outputs
3. **Shortlist 3–4 models** based on rough capability + cost tier
4. **Run evals** on your golden set — use a scoring rubric or AI-as-judge for open-ended tasks
5. **Measure latency and cost** at realistic call volumes
6. **Check licensing and compliance** for your deployment context
7. **Pick the smallest model that passes your quality bar** — not the best model you found

---

### Quick Reference: Decision Heuristics

|Situation|Recommendation|
|---|---|
|Prototype / low volume|Proprietary API (fast, low ops overhead)|
|High volume, cost-sensitive|Self-hosted open-weight (Llama, Mistral, Qwen)|
|Air-gapped / HIPAA / FedRAMP|Self-hosted open-weight only|
|Best reasoning quality needed|Frontier proprietary (Claude Opus, GPT-5, Gemini Ultra)|
|Fast / cheap / simple tasks|Small models: Haiku, Gemini Flash, Mistral 7B|
|Need IP indemnification|Proprietary with enterprise contract|
|Need fine-tuning on private data|Open-weight (LoRA/QLoRA)|
|Multi-modal (vision + text)|GPT-4o, Gemini, or Llama 4 (multimodal)|

---

The bottom line: choose not just for now, but for what you'll need 12 months from now. Proprietary gives you velocity. Hybrid gives you optionality. Most mature production systems end up routing across multiple models rather than betting everything on one.