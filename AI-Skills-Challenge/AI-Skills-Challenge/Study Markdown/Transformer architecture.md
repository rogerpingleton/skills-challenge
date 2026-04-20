# Transformer architecture

Here's a deep, engineer-focused breakdown of the transformer architecture. I'll work through it in layers — from the big picture down to the mechanics — with diagrams to build real intuition.

---

## The transformer at a glance

The transformer (Vaswani et al., 2017 — "Attention Is All You Need") replaced recurrent architectures (LSTMs, GRUs) as the dominant paradigm for sequence modeling. Its key insight: you don't need to process tokens sequentially. Instead, every token can attend to every other token _in parallel_, making transformers both faster to train and better at capturing long-range dependencies.

As an AI Engineer, you need to understand this architecture because:

- It determines what a model can and cannot do (context length, attention patterns)
- It explains scaling behavior (why bigger models are better)
- It informs prompt engineering (why position matters, why coherence degrades at context limits)
- It's the foundation for choosing between model families

---

## 1. The input pipeline: Tokens → Vectors

Before any attention happens, raw text must be turned into numbers. This is the **embedding + positional encoding**stage.

!transformer_input_pipeline.svg|697

**Key engineer facts here:**

- `d_model` is the model's hidden dimension. GPT-2 small = 768, GPT-3 = 12288, Llama 3 70B ≈ 8192. This single number heavily influences parameter count.
- **Positional encoding** is critical because attention is inherently position-agnostic. Original transformers used sinusoidal functions. Modern LLMs use **RoPE** (Rotary Position Embeddings) or **ALiBi** — these matter for context length extrapolation.
- **Vocabulary size** is typically 32k–128k tokens (GPT-4 uses ~100k). The embedding table itself (vocab_size × d_model) can be a significant chunk of total parameters.

---

## 2. The transformer block: What actually happens

The transformer is a stack of identical blocks. Each block does two things: **multi-head self-attention**, then a **feedforward network**. Both are wrapped with residual connections and layer normalization.

!transformer_block_structure.svg|697

The **residual stream** (the skip connections) is arguably the most important structural insight. Each block _adds_ to a persistent stream of information rather than transforming it wholesale. This is why transformers are stable to train at great depth, and it's what allows early layers to be bypassed for simple patterns.

---

## 3. Self-attention: The core mechanism

This is the mechanism that makes transformers transformers. Every token can look at every other token and decide how much to "pay attention" to it.

!self_attention_mechanism.svg|697

The math behind attention is:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    Returns: output (batch, heads, seq_len, d_k), weights
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Step 2: Apply causal mask (decoder/autoregressive only)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 3: Softmax to get weights (sum to 1 per query)
    weights = F.softmax(scores, dim=-1)
    
    # Step 4: Weighted sum of values
    output = torch.matmul(weights, V)
    
    return output, weights
```

**Critical engineer intuitions:**

- `Q·Kᵀ / √d_k` — dividing by √d_k prevents dot products from growing so large that softmax saturates and gradients vanish. This is why it's called _scaled_ dot-product attention.
- The **causal mask** (decoder models like GPT) zeroes out future positions — a token at position 5 can't see position 6+. Encoder models (BERT) have no such mask — bidirectional attention.
- Complexity is **O(n²)** in sequence length — this is the fundamental bottleneck. A 4k-token context has 16M attention computations per head per layer. This is why long-context models require engineering tricks (FlashAttention, sliding window attention, etc.).

---

## 4. Multi-head attention: Why multiple heads?

A single attention head lets each token look at all others, but forces one "way of looking." Multi-head attention runs `h`attention heads in parallel, each learning a different relationship type.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Projection matrices for Q, K, V — one per head (packed together)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection
    
    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        
        # Project + split into heads: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention per head
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads → (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        return self.W_o(attn_out)
```

**What different heads tend to learn** (empirically studied): some heads track syntactic relationships (subject-verb), others track coreference (pronoun → antecedent), others capture positional patterns (attending to the previous token). This specialization emerges from training — it's not hardcoded.

Typical head counts: GPT-2 small = 12 heads, Llama 3 70B = 64 heads.

---

## 5. The feedforward network: Where "knowledge" lives

After attention mixes information across tokens, the FFN processes each token _independently_. It's a simple two-layer MLP — but with a crucial expansion ratio:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4  # Typically 4× expansion
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()  # Modern models use SiLU/GELU, not ReLU
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

The FFN is believed to act as a **key-value memory** — factual associations ("Paris is the capital of France") are stored in the weight matrices here, not in the attention heads. Research has shown you can surgically edit these weights to change specific facts the model "knows."

The FFN is also typically the **largest component** by parameter count. Doubling `d_ff` (the expansion) adds more parameters than doubling the number of attention heads.

---

## 6. Encoder vs. Decoder vs. Encoder-Decoder

This is a critical model selection decision for AI Engineers:

!transformer_variants_comparison.svg|697

---

## 7. Scaling laws & what they mean for model selection

The Chinchilla scaling laws (Hoffmann et al., 2022) are essential knowledge for AI Engineers:

**Key finding:** For a given compute budget, the optimal strategy is to scale model size and training tokens _proportionally_. Many early large models (GPT-3) were undertrained. A smaller, well-trained model (Llama 2 13B) can match a larger undertrained model (GPT-3 175B) on many benchmarks.

This informs model selection in practice:

```
# Rough mental model for token-to-parameter ratio (Chinchilla optimal):
optimal_tokens ≈ 20 × num_parameters

# Example:
# 7B model  → train on ~140B tokens
# 70B model → train on ~1.4T tokens
# Llama 3 8B was trained on 15T tokens — heavily overtrained for inference efficiency
```

**The AI Engineer takeaway:** a "smaller" model trained longer is often better for inference cost and deployment, even if a larger undertrained model exists. Always check training token counts when comparing models.

---

## 8. Key architectural variants you'll encounter

|Feature|Original transformer|Modern LLMs|
|---|---|---|
|Position encoding|Sinusoidal (fixed)|RoPE, ALiBi (learnable/relative)|
|Normalization|Post-LN|Pre-LN (more stable)|
|Activation|ReLU|SiLU / GELU / SwiGLU|
|Attention|Standard multi-head|Grouped Query Attention (GQA), MQA|
|FFN|Dense|MoE (Mixture of Experts) in some|

**Grouped Query Attention (GQA)** — used in Llama 3, Mistral, Gemma: Instead of one K/V head per Q head, multiple Q heads share a single K/V head. Dramatically reduces KV cache memory at inference, enabling longer contexts and larger batches with no significant quality loss.

```python
# Standard MHA: num_kv_heads == num_q_heads (e.g., 32 == 32)
# GQA:          num_kv_heads < num_q_heads  (e.g., 8 KV heads, 32 Q heads)
# MQA:          num_kv_heads == 1           (single shared KV — most aggressive)
```

**Mixture of Experts (MoE)** — used in Mixtral 8x7B, GPT-4 (rumored): The FFN is replaced by N "expert" FFN layers, but only 2 are activated per token via a router. This scales parameter count without proportionally scaling compute. A 47B parameter MoE model may use only ~13B active parameters per forward pass.

---

## 9. The KV cache: What you must understand for deployment

During autoregressive generation, every new token requires computing attention over all previous tokens. Recomputing K and V for every previous token on every new token would be O(n²) per token generated. The **KV cache** stores computed K and V tensors so they only need to be computed once per token.

```python
# Without KV cache (naive):
# Generating token 100 → recompute attention over tokens 1-99 → expensive

# With KV cache:
# Generating token 100 → load cached K,V for tokens 1-99 → only compute new token's K,V

# Memory cost of KV cache per token:
kv_cache_bytes_per_token = (
    2           # K and V
    * num_layers
    * num_kv_heads
    * d_head    # d_model / num_heads
    * 2         # bytes per float16
)

# For Llama 3 8B (32 layers, 8 KV heads, d_head=128):
# = 2 * 32 * 8 * 128 * 2 = 131,072 bytes ≈ 128 KB per token
# 8K context window → ~1 GB just for KV cache
```

This is why context length is a hardware constraint, not just a model capability constraint. Managing KV cache is one of the central concerns in production LLM serving.

---

## 10. What this means for practical model selection

Pulling it all together — when you're choosing a foundation model:

**Use encoder-only** (BERT, `sentence-transformers`) when you need embeddings for semantic search, classification, or retrieval. Lower cost, specialized, no generation overhead.

**Use decoder-only** (Llama 3, Mistral, Claude API) for any generative task — chat, coding, extraction, reasoning. The default choice for 95% of AI Engineering use cases today.

**Check these architectural signals in model cards:**

- `context_length` → determines max input + output size; driven by position encoding and KV cache budget
- `num_kv_heads` → GQA/MQA means better inference efficiency at long contexts
- Training tokens → more is better (up to Chinchilla optimal); overtrained models are better for inference
- `d_model` / `num_layers` → deep-and-narrow vs. wide-and-shallow affects what kinds of tasks the model excels at

**Python snippet to inspect a model's architecture:**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")

print(f"Layers:         {config.num_hidden_layers}")
print(f"d_model:        {config.hidden_size}")
print(f"Q heads:        {config.num_attention_heads}")
print(f"KV heads:       {config.num_key_value_heads}")  # GQA
print(f"FFN dimension:  {config.intermediate_size}")
print(f"Context length: {config.max_position_embeddings}")
print(f"Vocab size:     {config.vocab_size}")
# Output for Llama 3 8B:
# Layers: 32, d_model: 4096, Q heads: 32, KV heads: 8
# FFN: 14336, Context: 8192, Vocab: 128256
```

---

**The one-paragraph summary for your mental model:** A transformer is a stack of identical blocks, each of which does two things — lets every token gather information from all other tokens via attention (contextual mixing), then independently transforms each token's representation via a feedforward network (knowledge/computation). The residual stream threading through all layers means later layers can be simple corrections rather than full transformations. The architecture's power comes entirely from scale: more layers, wider dimensions, more training data, and the emergent capabilities that arise from all three together.