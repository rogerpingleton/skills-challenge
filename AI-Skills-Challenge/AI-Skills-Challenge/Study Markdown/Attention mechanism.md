# Attention mechanism

## 1. The Problem Attention Solves

Before attention, the dominant approach for sequential data (text, speech) was Recurrent Neural Networks (RNNs). Their fatal flaw: they processed tokens one at a time, compressing the entire history of a sentence into a single fixed-length vector — a "memory bottleneck." By the time the model reached token 512, everything from token 1 had been squeezed into one blob of numbers. Long-range dependencies (e.g., a pronoun referring to a noun 200 words earlier) were routinely lost.

**Attention breaks this bottleneck.** Instead of compressing context into a single vector, attention lets every token in the sequence look directly at every other token and decide, dynamically, how much to care about each one. There is no lossy compression — the full context is accessible at every step.

The 2017 paper _"Attention Is All You Need"_ (Vaswani et al.) formalized this idea into the **Transformer architecture**, which dispensed with recurrence entirely in favor of pure attention layers. This architecture became the backbone of every major LLM: GPT, BERT, Claude, Gemini, Llama, and beyond.

---

## 2. The Simple Analogy

### The Library Analogy

Imagine you are a researcher in a library. You have a **question** (your query) and you walk up to the card catalogue.

- The **Keys** are the index cards in the catalogue — short summaries of what each book covers.
- You compare your question to every index card to see how relevant each book is to your question.
- The **Values** are the actual books themselves — the full content.
- The most relevant books (highest match scores) get pulled off the shelf and blended together into your answer. Books with low match scores barely contribute.

This is **attention**: you dynamically weight which "books" (values) to read based on how well their "index cards" (keys) match your "question" (query).

Now scale this up: in a sentence, **every single word is simultaneously the researcher, the index card, and the book**. Each token asks its own question, compares itself to all other tokens, and builds its own context-aware meaning from the answers. This is **self-attention**.

### Why Word Meanings Are Context-Dependent

Consider the word "bank":

- "I sat by the **bank** of the river."
- "I deposited money at the **bank**."

A traditional word embedding gives "bank" one fixed vector, unable to distinguish meaning. In self-attention, the token "bank" attends to "river" in the first sentence (high attention weight) and to "deposited/money" in the second, producing a completely different contextual embedding each time. This is the core power of the mechanism.

---

## 3. Core Technical Mechanics

### 3.1 Tokens and Embeddings (the Setup)

Text is tokenized into a sequence of tokens. Each token is mapped to a dense vector of dimension `d_model` (e.g., 512 or 4096). You now have a matrix `X` of shape `(sequence_length, d_model)`.

### 3.2 Queries, Keys, and Values (Q, K, V)

From the input matrix `X`, three separate linear projections are learned during training:

|Matrix|Weight|Shape of Projection|What it Represents|
|---|---|---|---|
|Query (Q)|W_Q|(d_model, d_k)|"What am I looking for?"|
|Key (K)|W_K|(d_model, d_k)|"What do I contain / advertise?"|
|Value (V)|W_V|(d_model, d_v)|"What information do I actually provide?"|

These weights (`W_Q`, `W_K`, `W_V`) are learned parameters — the model figures out, over millions of training steps, what kinds of queries, keys, and values are useful.

```
Q = X @ W_Q   # shape: (seq_len, d_k)
K = X @ W_K   # shape: (seq_len, d_k)
V = X @ W_V   # shape: (seq_len, d_v)
```

### 3.3 Scaled Dot-Product Attention (the Formula)

```
Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
```

Breaking this down step by step:

**Step 1: Compute raw attention scores**

```python
scores = Q @ K.T   # shape: (seq_len, seq_len)
```

Each entry `scores[i, j]` measures the dot-product similarity between token `i`'s query and token `j`'s key. A high score means "token `i` should pay a lot of attention to token `j`."

**Step 2: Scale by √d_k**

```python
scores = scores / math.sqrt(d_k)
```

The dot products grow large as `d_k` increases, pushing the softmax into regions with near-zero gradients. Dividing by `√d_k`keeps the gradients healthy. This is the "scaled" part of scaled dot-product attention.

**Step 3: Apply Softmax (row-wise)**

```python
weights = softmax(scores, dim=-1)   # shape: (seq_len, seq_len)
```

This converts the raw scores into a probability distribution for each token — the values in each row sum to 1. These are the **attention weights**.

**Step 4: Weighted sum of Values**

```python
output = weights @ V   # shape: (seq_len, d_v)
```

The output for each token is a weighted blend of all value vectors, where the weights reflect how relevant each token was. Tokens with very low attention weights contribute almost nothing.

### 3.4 The Attention Matrix Visualized

For a 5-token sentence like `["The", "cat", "sat", "on", "mat"]`, the attention matrix is 5×5:

```
             The   cat   sat   on   mat
       The [ 0.5   0.3   0.1  0.05  0.05 ]
       cat [ 0.2   0.4   0.2  0.1   0.1  ]
       sat [ 0.1   0.35  0.4  0.1   0.05 ]
        on [ 0.15  0.1   0.2  0.3   0.25 ]
       mat [ 0.1   0.2   0.1  0.2   0.4  ]
```

Each row sums to 1. The diagonal often has high values (a word attends to itself), but interesting off-diagonal patterns emerge — e.g., "sat" paying strong attention to "cat" (subject-verb relationship).

---

## 4. Multi-Head Attention (MHA)

A single attention head captures one type of relationship at a time. **Multi-head attention** runs `h` independent attention heads in parallel, each with its own learned `W_Q`, `W_K`, `W_V` projections:

```python
# Pseudocode
heads = []
for i in range(num_heads):
    Q_i = X @ W_Q[i]   # shape: (seq_len, d_k)
    K_i = X @ W_K[i]
    V_i = X @ W_V[i]
    head_i = scaled_dot_product_attention(Q_i, K_i, V_i)
    heads.append(head_i)

# Concatenate all heads, then project back to d_model
MultiHead = concat(heads, dim=-1) @ W_O
```

Where `d_k = d_model / num_heads` (typically), keeping total compute roughly constant.

**Why multiple heads?** Different heads learn to capture different types of relationships:

- One head might track **syntactic** patterns (subject → verb dependencies)
- Another might track **semantic** relationships (coreference: "he" → "John")
- Another might track **positional** patterns (nearby tokens)
- Another might track **domain-specific** patterns (e.g., code tokens)

The original `"Attention Is All You Need"` paper used `h = 8` heads with `d_model = 512`, giving `d_k = 64` per head.

---

## 5. Self-Attention vs. Cross-Attention vs. Causal Attention

These are not different formulas — they are the same Q/K/V mechanism applied in different configurations:

|Type|Q comes from|K/V come from|Used in|Purpose|
|---|---|---|---|---|
|**Self-Attention**|Same sequence as K/V|Same sequence|Encoder|Every token attends to all others — builds rich contextual representations|
|**Causal (Masked) Self-Attention**|Same sequence|Same sequence (past only)|Decoder (GPT-style)|Each token can only attend to tokens that came before it — prevents peeking at the future during generation|
|**Cross-Attention**|Target sequence|Source sequence|Encoder-Decoder (T5, original translation models)|Decoder tokens query into encoder representations — links output generation to input context|

**Key implication for model selection:** GPT-style (decoder-only) models use _causal_ self-attention and are optimized for generation. BERT-style (encoder-only) models use _bidirectional_ self-attention and are optimized for understanding/embedding. T5-style (encoder-decoder) models use both, suited for seq2seq tasks.

---

## 6. The Transformer Architecture in Context

The full Transformer block wraps attention in residual connections and layer normalization:

```
Input
  │
  ▼
[Layer Norm]
  │
  ▼
[Multi-Head Self-Attention]  ← the attention mechanism
  │
  ▼ (+ residual from Input)
[Layer Norm]
  │
  ▼
[Feed-Forward Network (two linear layers + activation)]
  │
  ▼ (+ residual)
Output (to next layer)
```

A full model stacks `N` of these blocks (GPT-3 uses 96 layers; Llama 3 70B uses 80 layers). Each layer refines the contextual representations that previous layers built.

**Positional Encoding:** Unlike RNNs, the attention mechanism itself has no notion of order — `Q @ K.T` is the same regardless of whether token A came before or after token B. Position is injected explicitly via:

- **Absolute Positional Embeddings** (original Transformer, BERT) — add a fixed or learned position vector to each token embedding.
- **Rotary Positional Embeddings (RoPE)** — encode positions by rotating Q and K vectors. Used in Llama, Mistral, GPT-NeoX. Generalizes better to lengths not seen in training.
- **ALiBi** — adds a bias to attention scores based on distance. No positional encoding in the embedding layer. Used in MPT and some Falcon variants.

---

## 7. Modern Attention Variants (2023–2026)

The original Multi-Head Attention (MHA) is computationally expensive — the **KV cache** (storing computed K and V tensors for all past tokens during autoregressive generation) grows linearly with context length and is a major memory bottleneck at inference. This has driven significant innovation.

### 7.1 Multi-Query Attention (MQA)

All query heads share a **single** K/V head. Drastically reduces KV cache size. Faster inference, but noticeably degrades model quality. Used in older Falcon models and some production inference APIs where raw speed is paramount.

```
MHA:  h query heads, h key heads, h value heads
MQA:  h query heads, 1 key head,  1 value head
```

### 7.2 Grouped-Query Attention (GQA)

The practical sweet spot. Query heads are divided into `g` groups; each group shares one K/V pair.

```
GQA:  h query heads, g key heads, g value heads    (1 ≤ g ≤ h)
```

With `h=32` query heads and `g=8` KV groups, every 4 query heads share one K/V head. This gives most of the KV cache savings of MQA with modeling quality close to full MHA.

**GQA has become the de facto standard for open-weight models in 2025–2026.** Llama 2 70B, Llama 3, Mistral 7B, and Qwen2 all use GQA. Ablation studies show it performs comparably to standard MHA in terms of model quality.

### 7.3 Multi-Head Latent Attention (MLA)

Introduced in DeepSeek-V2 (2024) and used in DeepSeek-V3 and R1. Takes a fundamentally different approach: instead of reducing the _number_ of K/V heads like GQA, MLA **compresses** the K and V tensors into a low-dimensional latent space before caching.

```
Standard MHA caching:  store full K and V tensors per layer per token
MLA caching:           store a small compressed latent vector, reconstruct K/V at inference time
```

The result is a **93.3% reduction in KV cache size** versus MHA (per the DeepSeek-V2 paper), with ablation studies showing MLA can match or _outperform_ MHA quality — while GQA slightly underperformed it. This makes MLA a superior efficiency move, not just a memory hack.

The trade-off is implementation complexity. GQA remains popular for labs that want robust, simpler-to-implement attention. MLA is the frontier for large-scale inference efficiency.

### 7.4 Flash Attention

Flash Attention (Dao et al., 2022) is a hardware-aware algorithmic rewrite of standard attention. It does not change the math — the result is identical to standard attention — but it avoids writing the full N×N attention matrix to slow GPU HBM (High Bandwidth Memory).

The key insight: **attention is an I/O problem, not a compute problem.** GPU on-chip SRAM is ~10× faster than HBM but tiny (~20 MB). Flash Attention tiles the Q, K, V matrices into blocks small enough to fit in SRAM, processes them on-chip, and accumulates results directly into the output without ever materializing the full attention matrix.

- **FlashAttention-1** (2022): Foundational I/O-aware algorithm.
- **FlashAttention-2** (2023): Better work partitioning, more parallelism.
- **FlashAttention-3** (2024): Optimized for NVIDIA Hopper GPUs (H100), supports variable-length batching and RoPE.

Flash Attention is now ubiquitous — it is the default attention kernel in nearly all serious LLM training and inference frameworks (vLLM, HuggingFace, PyTorch 2.x, etc.). As an AI engineer, you should assume it is always on.

### 7.5 Sliding Window & Sparse Attention

For very long contexts, full attention is still O(N²). Sliding window attention restricts each token to attending only within a fixed local window (e.g., 4096 tokens), reducing complexity to O(N × window_size).

The downside: long-range dependencies beyond the window are not directly captured. This is why Mistral and Gemma use sliding window attention in _some_ layers while keeping full attention in others — a hybrid approach. Sparse and block-sparse attention variants (LongFormer, BigBird, SeerAttention) follow similar principles, selectively computing only the most important attention scores.

---

## 8. The Quadratic Scaling Problem

This is the central engineering tension you need to internalize:

**Standard attention scales as O(N²)** in both compute (FLOPs) and memory, where N is the sequence length.

|Context Length|Attention Matrix Size|Notes|
|---|---|---|
|1,024 tokens|~1 million entries|Trivial|
|32,768 tokens|~1 billion entries|Manageable with FlashAttn|
|128,000 tokens (GPT-4 Turbo)|~16 billion entries|Memory is the bottleneck|
|1,000,000 tokens (Gemini 1.5)|~1 trillion entries|Requires architectural tricks|

The KV cache grows proportionally to `(num_layers × num_KV_heads × seq_len × d_head × bytes_per_param)`. For a Llama 3 70B model processing 100K tokens, the KV cache alone can exceed 100 GB. This is why GQA and MLA exist, and why context length is a first-class engineering concern when selecting a model.

---

## 9. Python Code Examples

### Example 1: Minimal Scaled Dot-Product Attention from Scratch

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    mask: optional (batch, seq_len_q, seq_len_k) boolean mask (True = ignore)
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute raw scores
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    # scores shape: (batch, seq_len_q, seq_len_k)
    
    # Step 2: Apply mask (e.g., causal mask for autoregressive generation)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Step 3: Softmax over the key dimension
    weights = F.softmax(scores, dim=-1)
    # weights shape: (batch, seq_len_q, seq_len_k)
    
    # Step 4: Weighted sum of values
    output = torch.bmm(weights, V)
    # output shape: (batch, seq_len_q, d_v)
    
    return output, weights
```

### Example 2: Multi-Head Attention from Scratch

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        # Learned projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # output projection
    
    def split_heads(self, x):
        """Reshape (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.split_heads(self.W_Q(x))  # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        context = torch.matmul(weights, V)        # (batch, heads, seq_len, d_k)
        
        # Concatenate heads and project back to d_model
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, seq_len, self.d_model)
        output = self.W_O(context)
        
        return output  # (batch, seq_len, d_model)


# Quick usage check
d_model, num_heads, seq_len, batch = 512, 8, 64, 2
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
x = torch.randn(batch, seq_len, d_model)
out = mha(x)
print(out.shape)  # torch.Size([2, 64, 512])
```

### Example 3: Using Flash Attention via PyTorch (Recommended for Production)

```python
import torch
import torch.nn.functional as F

# PyTorch 2.x+ exposes F.scaled_dot_product_attention which automatically
# uses FlashAttention under the hood when available (CUDA, correct dtypes)
def efficient_attention(Q, K, V, is_causal=False):
    """
    Drop-in replacement for manual attention — uses FlashAttention kernels
    automatically when possible.
    
    Q, K, V: (batch, num_heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal  # True for autoregressive (decoder) attention
    )

# For production models, this single line gives you FlashAttention-2/3
# performance without writing any kernel code.
```

### Example 4: Inspecting Attention Weights with Hugging Face

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load a model with output_attentions=True
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
layer_0_attention = outputs.attentions[0]
print(f"Layer 0 attention shape: {layer_0_attention.shape}")
# torch.Size([1, 12, 8, 8])  -> 12 heads, 8 tokens x 8 tokens

# Average across heads for a summary view
avg_attention = layer_0_attention.squeeze(0).mean(dim=0)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)
print("Averaged attention matrix shape:", avg_attention.shape)
```

---

## 10. What This Means for Model Selection

As an AI engineer, attention mechanism choices made by a model's designers have direct consequences for your deployments.

### Context Window vs. Memory

A model's **maximum context length** is fundamentally constrained by the attention mechanism and available GPU memory. If your application requires long document processing, RAG with large retrieved chunks, or multi-turn conversations with long history:

- Check whether the model uses **GQA or MLA** — these significantly reduce the KV cache footprint, enabling longer effective context windows at the same memory budget.
- A 70B model with GQA can often serve longer contexts at lower VRAM than a smaller model using standard MHA.
- Models with MLA (DeepSeek-V3, R1) achieve a ~93% KV cache reduction, making them extremely attractive for long-context production workloads.

### Attention Type and Task Fit

|Model Type|Attention Config|Best For|
|---|---|---|
|Encoder-only (BERT, RoBERTa)|Bidirectional self-attention|Embeddings, classification, semantic search|
|Decoder-only (GPT, Llama, Mistral)|Causal self-attention|Generation, chat, code completion|
|Encoder-Decoder (T5, BART)|Self + Cross-attention|Translation, summarization, structured output|
|Long-context models (Gemini, Claude)|Hybrid/sparse attention|Document QA, long-context reasoning|

### Inference Performance

- **GQA** (Llama 3, Mistral) reduces memory bandwidth pressure during the KV cache read/write cycle, enabling higher throughput per GPU at inference time.
- **FlashAttention** is almost always enabled in modern inference stacks (vLLM, TGI, llama.cpp). If you are rolling your own inference, verify it is active — the performance difference is 2–4×.
- **MQA** models are the fastest for raw inference but sacrifice quality. If you are building a high-throughput API where quality is somewhat flexible, an MQA model may outperform a larger GQA model within the same GPU budget.

### Number of Attention Heads and Layers

Depth (number of layers) and width (number of heads, d_model) are key model quality signals. Larger models with more heads can capture more diverse relationship types simultaneously. When evaluating models:

- More heads does not always mean better — ablation studies show quality can drop with too many heads if `d_k` per head becomes too small.
- The standard in modern large models is 32–128 attention heads.

---

## 11. Practical Checklist for AI Engineers

When evaluating or deploying a foundation model, ask:

**Architecture:**

- [ ] Decoder-only, encoder-only, or encoder-decoder? (determines task fit)
- [ ] What is the attention variant? (MHA / GQA / MQA / MLA)
- [ ] What positional encoding does it use? (RoPE scales better to long contexts)
- [ ] Is FlashAttention supported or built into the inference stack?

**Context & Memory:**

- [ ] What is the maximum context length, and what is the _effective_ context length under your memory budget?
- [ ] How large will the KV cache be at your target sequence length and batch size?
- [ ] Does the model use GQA/MLA? How many KV heads?

**Inference Engineering:**

- [ ] Does the serving framework (vLLM, TGI, ollama) support the model's attention variant natively?
- [ ] Are you using continuous batching and paged KV cache? (vLLM's key optimizations)
- [ ] What quantization are you applying to the KV cache, if any? (fp16 vs int8 vs fp8)

**Quality Signals:**

- [ ] How many attention heads and layers? (quality proxy)
- [ ] Were there ablations comparing attention variants at this model's scale?

---

## 12. Summary: The Landscape at a Glance

```
ATTENTION MECHANISM EVOLUTION TIMELINE

2014  Bahdanau Attention ────────────── Original additive attention for RNN seq2seq
2017  Scaled Dot-Product Attention ──── "Attention Is All You Need" — the Transformer
2017  Multi-Head Attention (MHA) ─────── Parallel attention heads, standard for years
2019  Multi-Query Attention (MQA) ─────── Single shared K/V — fastest, lower quality
2022  Flash Attention ────────────────── IO-aware rewrite; same math, 2-4x faster
2023  Grouped-Query Attention (GQA) ──── Sweet spot between MHA and MQA; now standard
2024  Multi-Head Latent Attention (MLA)─ KV compression via latent space (DeepSeek-V2/V3)
2024  Flash Attention 3 ────────────────  H100 optimization, variable length batching
2025+ Sparse / Hybrid Attention ──────── Mixing local and global attention for long ctx
```

**The core equation never changed.** Since 2017, every major architecture uses:

```
Attention(Q, K, V) = softmax( Q @ K.T / √d_k ) @ V
```

Everything else — GQA, MLA, FlashAttention, RoPE — is an optimization of _how_ you compute or store this, not _what_you compute. Understanding the core formula deeply means you will always be able to reason about new variants, no matter how they are named.

---

_Report compiled April 2026. Sources include Vaswani et al. (2017), Ainslie et al. GQA (2023), Dao et al. FlashAttention (2022–2024), DeepSeek-V2/V3 papers (2024), and current model architecture surveys._