# Tokenization

Tokenization is one of the most foundational — and frequently misunderstood — topics in AI engineering. Here's a comprehensive breakdown.

---

## What is tokenization?

Before a language model can process text, the text must be converted into numbers. Tokenization is the process of splitting raw text into **tokens** — discrete units that the model operates on — and then mapping each token to an integer ID via a vocabulary lookup.

A token is not always a word. Depending on the algorithm, tokens can be whole words, subwords, individual characters, or even multi-word phrases.

---

## The major algorithms

**Byte-Pair Encoding (BPE)** — used by GPT models, LLaMA, and many others. Starts from individual bytes or characters and iteratively merges the most frequent adjacent pairs into new tokens. The vocabulary is built from training data frequency statistics. The result: common words become single tokens, rare/unknown words are split into subword pieces.

**WordPiece** — used by BERT. Similar to BPE but merges are driven by maximizing likelihood of the training data rather than raw frequency. Produces very similar results in practice.

**SentencePiece** — used by T5, LLaMA 2, Gemma, and others. Language-agnostic; operates directly on raw Unicode text without pre-tokenization. Supports both BPE and unigram language model variants. Crucially, it handles whitespace as a real character, so it's more robust across languages.

**Tiktoken** — OpenAI's BPE implementation (used in GPT-3.5/4, Claude uses a similar approach). Byte-level BPE, meaning the base vocabulary is all 256 bytes, so it can represent _any_ text without unknowns.

Here's a quick interactive explorer to make this concrete:> Try typing different inputs — numbers, code, non-English text — and switch between BPE and character-level to see the token count change.

---

## The subtle nuances every AI engineer needs to know

### 1. Token count ≠ word count

The ratio varies wildly. Common English words are typically 1 token. Rare words, technical jargon, and non-English text often fragment into 2–5 tokens per word. Some numbers serialize character by character. Code can be especially unpredictable.

This matters because **you're billed per token and rate-limited per token**, not per word or character. A prompt that looks short can be surprisingly expensive.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

examples = [
    "Hello, world!",
    "Tokenization",          # common word → 1 token
    "Uncharacteristically",  # rare word → fragments
    "2024-01-15",            # dates can fragment
    "def tokenize(text):",   # code is efficient
    "こんにちは",              # Japanese: ~1 char/token
    "مرحبا",                  # Arabic
]

for s in examples:
    toks = enc.encode(s)
    print(f"{len(toks):3d} tokens | {s}")
```

### 2. The fertility problem (especially for non-English text)

English is heavily over-represented in most training corpora, so English vocabulary is well-optimized. Non-English text, especially CJK (Chinese/Japanese/Korean), Arabic, and low-resource languages, can have 3–8× higher token counts for the same semantic content. This has two concrete consequences:

- **Cost**: a Turkish or Thai query may cost 4× more than the English equivalent
- **Context window asymmetry**: a model with a 128k token context "fits" far less information in non-English languages

For multilingual applications, factor this into your context management and cost modeling.

### 3. Tokenization boundaries affect model behavior

Because attention operates on token-level representations, _where you split matters_. A model may handle `tokenization`differently than `token` + `##ization`. This shows up in:

- **Arithmetic**: models notoriously struggle with multi-digit addition partly because numbers like `12345` are tokenized inconsistently — sometimes `123` + `45`, sometimes `1` + `2` + `3` + `4` + `5`. Different tokenizations = different learned representations.
- **Spelling and character-level tasks**: asking a model "how many r's in 'strawberry'" is hard because the word may tokenize as `str` + `awberry` — the model never "sees" individual characters.
- **Code**: identifiers with underscores (`my_variable`) may split in unexpected places. Most code-focused models have tokenizers tuned for this.

### 4. Whitespace and special characters are part of tokens

In BPE (Tiktoken-style), the space before a word is _attached to the word_, not left as a separate token. `" hello"` and `"hello"` are different tokens. This matters when you're constructing prompts programmatically or doing token-level string manipulation:

```python
enc = tiktoken.encoding_for_model("gpt-4o")

print(enc.encode("hello"))    # [15339]
print(enc.encode(" hello"))   # [24748]  — different token!
print(enc.encode("Hello"))    # [9906]   — capitalization matters
```

### 5. Special/control tokens

Every tokenizer reserves special tokens that never appear in normal text. These include:

|Token|Purpose|
|---|---|
|`<|endoftext|
|`<|im_start|
|`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`|BERT-family special tokens|
|`<s>`, `</s>`|SentencePiece start/end of sequence|
|`<unk>`|Unknown token (rare in modern BPE)|

If these tokens appear in user input (e.g., prompt injection), they can disrupt model behavior. Always sanitize or escape them in production pipelines.

### 6. Context window is measured in tokens

The model's maximum context window (`128k`, `200k`, etc.) is a token budget shared between:

- System prompt
- Conversation history
- Retrieved context (RAG)
- The response itself

As an AI engineer, you need to be counting tokens _before_ making API calls, not after. Libraries like `tiktoken` (OpenAI), `transformers`' `AutoTokenizer`, or Anthropic's own token counting endpoint let you do this.

```python
import anthropic

client = anthropic.Anthropic()

# Count tokens before sending
response = client.messages.count_tokens(
    model="claude-opus-4-5",
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Explain tokenization."}]
)

print(f"Input tokens: {response.input_tokens}")
```

### 7. Tokenizer mismatch between models

When you switch models (say, from GPT-4 to Claude to LLaMA), token counts for the same string will differ. The vocabularies, merge rules, and special token sets are all different. Two important implications:

- **Cost estimates are model-specific** — you can't reuse tiktoken for Claude's pricing
- **Chunking strategies for RAG must be recalibrated** when you migrate models

### 8. Off-by-one and truncation bugs

When splitting long documents into chunks, naive character-based splitting can cut mid-token, producing malformed inputs. Always chunk at the token level:

```python
def chunk_text(text: str, model: str, max_tokens: int, overlap: int = 50) -> list[str]:
    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(enc.decode(chunk_ids))
        start += max_tokens - overlap  # sliding window with overlap
    
    return chunks
```

### 9. Vocabulary size is a design tradeoff

|Vocab size|Tradeoff|
|---|---|
|Small (~8k)|Aggressive fragmentation; slow inference; poor for rare words|
|Medium (~32k–50k)|Typical sweet spot (GPT-2: 50k, LLaMA 2: 32k)|
|Large (~100k–256k)|Better multilingual coverage; larger embedding table|

Claude uses a ~100k token vocabulary for better cross-lingual efficiency. Larger vocab = each token carries more meaning on average, but also a larger embedding matrix in memory.

---

## Practical checklist for AI engineers

When working on any LLM-powered system, keep these in mind:

1. **Measure token counts** before making API calls, especially for dynamic prompts
2. **Budget your context window** explicitly — system prompt + history + retrieval + response
3. **Use token-aware chunking** in RAG pipelines, never character-based splitting
4. **Account for language fertility** in multilingual applications for both cost and context
5. **Sanitize special tokens** from user input to prevent prompt injection
6. **Re-profile token counts** when switching models — tokenizers are not interchangeable
7. **Consider token efficiency** when designing prompts — verbose instructions cost money and context

Tokenization sits below the level most engineers interact with daily, but its effects surface constantly: in unexpected costs, weird model behaviors on edge cases, and context window overflows. Understanding it deeply gives you leverage over all of those.

## Tokenization Purpose

Tokenization serves as the **bridge between raw human-readable text and the numerical representations that neural networks can actually process**. A model cannot operate on strings — it operates on vectors of numbers — so tokenization is the necessary translation layer.

At its core, the purpose is threefold: convert text to numbers, do so in a way that preserves linguistic meaning, and do so efficiently enough that the model's finite context window isn't wasted.

**The primary benefits:**

**Vocabulary control** — rather than having millions of possible words (including misspellings, proper nouns, technical jargon, and every word in every language), tokenization with subword algorithms like BPE bounds the vocabulary to a fixed size (typically 32k–100k tokens). Every possible input can be represented without an "unknown" fallback.

**Handling unseen words gracefully** — because rare or novel words decompose into familiar subword pieces, the model can make reasonable sense of a word it never saw during training. "Transformerification" might tokenize as `Transform` + `er`+ `ification` — each piece has learned meaning.

**Computational efficiency** — processing 500 tokens is far cheaper than processing 2,500 individual characters. Subword tokenization finds a sweet spot between character-level (too many steps, too long sequences) and word-level (too large a vocabulary, can't handle unknowns).

**Shared representations across morphological variants** — words like `run`, `running`, `runner`, and `runs` may share subword tokens, allowing the model to leverage learned semantics across related forms rather than treating each as entirely independent.

**Enabling fixed-size embeddings** — once text is token IDs, each ID maps to a learned embedding vector of fixed dimension. This is what feeds into the attention mechanism. Without tokenization, there's no clean way to convert variable-length raw text into the structured numerical input a transformer expects.

**Language agnosticism** — byte-level BPE (like tiktoken) can represent any language, emoji, or even arbitrary binary sequences using the same vocabulary, because the base units are raw bytes. No language is truly "unknown."

In short: tokenization makes text mathematically tractable, keeps the vocabulary manageable, handles linguistic variability gracefully, and sets the stage for everything the model does downstream — attention, embeddings, generation, all of it starts here.