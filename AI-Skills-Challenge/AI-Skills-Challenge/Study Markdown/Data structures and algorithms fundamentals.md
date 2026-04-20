# Data structures and algorithms fundamentals

This is a meaty topic, so I'll cover everything that's directly relevant to your work as an AI engineer — not just the CS101 overview, but the _why_ behind each concept in the context of AI/ML systems.

---

## 1. Why DSA Matters in AI Engineering

AI engineering isn't just about calling model APIs or training neural nets. At the infrastructure level, you're constantly making decisions that hinge on algorithmic efficiency:

- **Embedding search** over millions of vectors requires understanding nearest-neighbor algorithms
- **Tokenization pipelines** depend on efficient string/trie structures
- **Graph-based reasoning** (knowledge graphs, agent DAGs) requires graph traversal
- **Data preprocessing at scale** lives or dies by time/space complexity choices

The theme throughout: **the wrong data structure can turn a real-time system into an unusable one.**

---

## 2. Complexity Analysis — The Lens for Everything

Before data structures, you need the measuring stick.

### Big-O Notation

|Notation|Name|Example|
|---|---|---|
|O(1)|Constant|Dictionary lookup|
|O(log n)|Logarithmic|Binary search|
|O(n)|Linear|Scanning a list|
|O(n log n)|Linearithmic|Merge sort|
|O(n²)|Quadratic|Naive pairwise similarity|
|O(2ⁿ)|Exponential|Brute-force search|

**AI Engineering relevance:** When your embedding database has 10M vectors, the difference between O(n) and O(log n) similarity search is the difference between 10 seconds and 3 milliseconds per query.

```python
# O(n) — linear scan, terrible at scale
def find_similar_naive(query_vec, embeddings):
    return [cosine_sim(query_vec, e) for e in embeddings]  # scans all 10M

# O(log n) — using an index (e.g., HNSW, FAISS)
import faiss
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
distances, indices = index.search(query_vec, k=10)  # approximate, but fast
```

---

## 3. Core Data Structures

### 3.1 Arrays & Lists

The foundation. In Python, `list` is a dynamic array.

- **Access:** O(1) by index
- **Append:** O(1) amortized
- **Insert/Delete at middle:** O(n) — avoid in hot paths

**AI use case:** Storing token IDs, batching inputs, holding activation tensors.

```python
# Batching tokenized inputs — arrays are ideal here
token_ids = [tokenizer.encode(text) for text in corpus]
batch = token_ids[:32]  # O(1) slice
```

**NumPy arrays** are the real workhorse — contiguous memory, SIMD-optimized, vectorized ops replace Python loops entirely.

```python
import numpy as np

# Never do this in AI work:
similarities = [np.dot(q, e) for e in embeddings]  # Python loop, slow

# Do this instead — vectorized, ~100x faster:
similarities = embeddings @ query_vec  # O(n·d) but in C, cache-friendly
```

---

### 3.2 Hash Maps (Dictionaries)

Python `dict` is a hash table. Average O(1) for get/set/delete.

**AI use cases:**

- Vocabulary lookups (token → ID)
- Caching model outputs (prompt → response)
- Counting n-grams, frequencies

```python
# Vocabulary: the backbone of every tokenizer
vocab = {"hello": 0, "world": 1, "<unk>": 2}  # O(1) lookup
token_id = vocab.get("hello", vocab["<unk>"])

# Caching LLM calls — critical for cost control
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_embedding(text: str) -> tuple:
    return tuple(model.embed(text))  # Cache hit = O(1), no API call
```

**Collision handling** is abstracted away in Python, but understand that worst-case dict lookup is O(n) — relevant if you're using mutable/unhashable objects as keys.

---

### 3.3 Trees

#### Binary Search Trees (BST)

- Search/Insert/Delete: O(log n) average, O(n) worst (unbalanced)

#### Tries (Prefix Trees)

Extremely important for NLP/AI:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search_prefix(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
```

**AI use case:** Tokenizer vocabularies (BPE, WordPiece), autocomplete systems, prefix-based token constraints in constrained decoding.

#### Decision Trees / Tree Ensembles

Not just a data structure — a model class (Random Forests, XGBoost). Understanding tree splitting (recursive partitioning) matters for feature selection and interpretability.

---

### 3.4 Heaps (Priority Queues)

A heap is a complete binary tree satisfying the heap property. Python's `heapq` is a min-heap.

- Push/pop: O(log n)
- Peek min: O(1)

**AI use cases:**

```python
import heapq

# Beam search — keeping top-k candidates at each step
# This is literally inside every seq2seq decoder
beam = []  # min-heap by negative score (simulate max-heap)

def beam_search_step(candidates, beam_width=5):
    for score, sequence in candidates:
        heapq.heappush(beam, (score, sequence))
        if len(beam) > beam_width:
            heapq.heappop(beam)  # evict worst
    return beam

# Top-k sampling from logits
import numpy as np

def top_k_sample(logits, k=50):
    top_k_indices = np.argpartition(logits, -k)[-k:]  # O(n) partial sort
    top_k_logits = logits[top_k_indices]
    probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    return np.random.choice(top_k_indices, p=probs)
```

---

### 3.5 Graphs

Graphs are everywhere in AI engineering:

|AI Concept|Graph Representation|
|---|---|
|Knowledge graphs|Nodes = entities, edges = relations|
|LLM agent pipelines|DAG of tool calls|
|Neural networks|Computation graph|
|Dependency parsing|Directed graph over tokens|
|RAG pipelines|Graph of document chunks + relations|

**Key representations:**

```python
# Adjacency list — sparse graphs (most real-world graphs)
graph = {
    "retrieve": ["rerank", "summarize"],
    "rerank": ["answer"],
    "summarize": ["answer"],
    "answer": []
}

# Adjacency matrix — dense graphs, fast edge lookup
import numpy as np
n = 4
adj_matrix = np.zeros((n, n), dtype=int)
# adj_matrix[i][j] = 1 means edge from i to j
```

**Graph traversal:**

```python
from collections import deque

# BFS — explores level by level (used in shortest-path, agent scheduling)
def bfs(graph, start):
    visited, queue = set(), deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            queue.extend(graph.get(node, []))
    return order

# DFS — used in topological sort, cycle detection in DAG pipelines
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# Topological sort — critical for DAG-based agent execution order
def topological_sort(graph):
    from collections import defaultdict
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([n for n in graph if in_degree[n] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order
```

**AI use case:** LangGraph, LlamaIndex pipelines, and any multi-agent framework uses topological sort to schedule node execution in a DAG.

---

### 3.6 Linked Lists & Deques

Less prominent in Python AI work directly, but `collections.deque` is O(1) on both ends — use it for sliding windows, rolling context buffers, and BFS queues (never use `list.pop(0)` — that's O(n)).

```python
from collections import deque

# Sliding window context buffer for a streaming agent
context_window = deque(maxlen=10)  # auto-evicts oldest on overflow
context_window.append({"role": "user", "content": "hello"})
```

---

## 4. Core Algorithms

### 4.1 Sorting

|Algorithm|Time|Space|Stable?|When to use|
|---|---|---|---|---|
|Timsort (Python default)|O(n log n)|O(n)|Yes|General purpose|
|Merge sort|O(n log n)|O(n)|Yes|External sorting|
|Quicksort|O(n log n) avg|O(log n)|No|In-place sorting|
|Counting sort|O(n+k)|O(k)|Yes|Small integer ranges|

**AI use case:** Ranking retrieval results, sorting candidates in beam search, ordering documents by relevance score.

```python
# Sorting retrieved docs by score — Timsort handles this well
docs = [("doc_3", 0.91), ("doc_1", 0.95), ("doc_2", 0.87)]
ranked = sorted(docs, key=lambda x: x[1], reverse=True)

# Partial sort with heapq when you only need top-k (faster than full sort)
import heapq
top_k = heapq.nlargest(3, docs, key=lambda x: x[1])  # O(n log k)
```

---

### 4.2 Searching

**Binary search** — O(log n), requires sorted input:

```python
import bisect

# Fast lookup in sorted token scores or thresholds
sorted_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
pos = bisect.bisect_left(sorted_scores, 0.6)  # finds insertion point in O(log n)
```

---

### 4.3 Dynamic Programming (DP)

DP solves problems by breaking them into overlapping subproblems and caching results (memoization = top-down, tabulation = bottom-up).

**AI use cases:**

```python
# Viterbi algorithm — sequence labeling (NER, POS tagging)
# Classic DP on HMMs: find most likely state sequence

# Edit distance — used in fuzzy matching, spell correction, evaluation metrics
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# Used in: evaluating OCR output, fuzzy entity matching in RAG
print(edit_distance("kitten", "sitting"))  # 3
```

---

### 4.4 Hashing & Locality-Sensitive Hashing (LSH)

Beyond standard hash maps, **LSH** is a critical AI algorithm: it hashes similar items into the same bucket with high probability — enabling approximate nearest neighbor (ANN) search.

```python
# Concept: MinHash for document similarity (Jaccard)
# Used in deduplication pipelines before training data curation

from datasketch import MinHash

m1, m2 = MinHash(), MinHash()
for word in "the cat sat on the mat".split():
    m1.update(word.encode())
for word in "the cat sat on a mat".split():
    m2.update(word.encode())

print(m1.jaccard(m2))  # ~0.8 — fast similarity without comparing all pairs
```

**HNSW (Hierarchical Navigable Small World)** — the dominant ANN algorithm in vector databases (Pinecone, Weaviate, Chroma) — is a graph-based structure built on small-world graph theory. Understanding it helps you tune `ef_construction` and `M` parameters correctly.

---

### 4.5 Sliding Window & Two-Pointer Techniques

Useful in sequence processing:

```python
# Sliding window: chunking documents for RAG with overlap
def chunk_with_overlap(tokens, chunk_size=512, overlap=64):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        start += chunk_size - overlap  # slide forward with overlap
    return chunks
```

---

## 5. Specialized AI-Relevant Structures

### 5.1 Vectors & Embedding Spaces

Embeddings _are_ your data structure in modern AI. Key operations:

- **Cosine similarity:** O(d) where d = embedding dimension
- **Dot product:** O(d), used in attention mechanisms
- **L2 distance:** O(d)

The entire attention mechanism in transformers is a structured dot-product query over a matrix:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)      # O(n² · d)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V
```

### 5.2 Inverted Index

The data structure behind search engines and keyword-based RAG (BM25):

```python
from collections import defaultdict

# Build inverted index
def build_inverted_index(docs):
    index = defaultdict(list)
    for doc_id, text in enumerate(docs):
        for token in text.lower().split():
            index[token].append(doc_id)
    return index

# Lookup: O(1) per token
index = build_inverted_index(["cat sat mat", "dog ran fast", "cat ran away"])
print(index["cat"])  # [0, 2] — docs containing "cat"
```

BM25 (used in hybrid RAG) builds on this with TF-IDF weighting.

---

## 6. Memory & Complexity Tradeoffs in Practice

|Scenario|Naive Approach|Better Approach|Why|
|---|---|---|---|
|Top-k from 1M scores|`sorted()[-k:]` — O(n log n)|`heapq.nlargest(k, ...)` — O(n log k)|Only track k items|
|Dedup 10M documents|Pairwise compare — O(n²)|MinHash LSH — O(n)|Approximate is fine|
|Token vocab lookup|List scan — O(n)|Dict — O(1)|Hash table|
|Nearest neighbor in vector DB|Brute force — O(n·d)|HNSW/FAISS — O(log n · d)|Approximate ANN|
|Sliding context window|`list.pop(0)` — O(n)|`deque(maxlen=k)` — O(1)|Right structure|

---

## 7. What to Internalize vs. What to Look Up

**Internalize deeply:**

- Big-O intuition — you should _feel_ when something is O(n²) in a loop
- Hash maps, heaps, deques — you'll use these daily
- Graph traversal (BFS/DFS/topological sort) — essential for agent frameworks
- DP conceptually — edit distance, Viterbi, memoization patterns

**Know at a high level:**

- Trie internals — important for tokenizer understanding
- LSH / HNSW — understand what vector DBs are doing under the hood
- Sorting algorithm properties — know when to use `heapq.nlargest` vs `sorted`

**You can always look up:**

- Exact BST rotation logic
- Specific graph algorithm implementations (Dijkstra, A*)
- Advanced DP formulations

---

The throughline for AI engineering is this: **the algorithmic bottlenecks in modern AI systems are rarely in the model forward pass itself — they're in the data pipelines, retrieval layers, and orchestration logic surrounding it.** That's exactly where these fundamentals pay off.