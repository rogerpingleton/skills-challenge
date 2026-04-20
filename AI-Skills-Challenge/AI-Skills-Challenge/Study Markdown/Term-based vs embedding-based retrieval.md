
# Term-based vs embedding-based retrieval

## Term-Based vs. Embedding-Based Retrieval in RAG

### The Core Idea

The **retrieval** step in RAG is where your system goes out and finds the most relevant chunks of text to feed the LLM as context. The fundamental question is: _how do you define "relevant"?_ Two very different philosophies answer this — and as an AI Engineer, you'll need to understand both deeply.

---

### Term-Based Retrieval (Sparse Retrieval)

Sparse retrieval is built on inverted indexes — the same data structure that powers traditional search engines. Each document is represented as a vector in a vocabulary-sized space where most entries are zero, and non-zero entries correspond to terms that appear in the document, weighted by a scoring function.

The dominant algorithm here is **BM25 (Best Matching 25)**. BM25 calculates relevance scores based on term frequency (TF) and inverse document frequency (IDF), while also applying document length normalization. The formula ensures that terms appearing frequently within a document and those rare across the corpus are appropriately weighted, improving search accuracy and relevance.

In plain English: if your query contains the word `"invoice"`, BM25 finds documents that literally contain the word `"invoice"`and ranks them based on how often that term appears vs. how common it is across the whole corpus.

**Strengths:**

- Operationally, sparse retrieval with BM25 is fast and cheap. Inverted indexes like Elasticsearch and OpenSearch handle billions of documents on commodity hardware, support real-time updates, and return results in single-digit milliseconds. There's no GPU requirement, no embedding model to maintain, and no approximate nearest neighbor index to rebuild when documents are updated.
- Full-text search gives you deterministic matching for identifiers, clause numbers, endpoints, and exact phrases.

**Weaknesses:**

- If a user asks "how to fix slow queries" and your document says "optimization techniques for database performance," BM25 finds no match because there is no overlapping vocabulary.

---

### Embedding-Based Retrieval (Dense Retrieval)

Dense retrieval represents both queries and documents as fixed-dimensional embedding vectors, typically 768 or 1024 dimensions, produced by a transformer encoder. At query time, the query is encoded to a vector and the system finds the documents whose embedding vectors are most similar (highest cosine similarity or dot product) to the query vector.

The key insight is that embedding models compress _meaning_ into numeric space, so semantically related concepts land near each other even if they share no words.

**Example:** A query like `"car won't start"` can retrieve a document about `"engine ignition failure"` because the embedding model has learned those concepts are related.

Unlike traditional keyword-based retrieval, dense retrieval uses vector embeddings to represent both queries and documents in a shared semantic space. This allows the system to identify relevant information even when the query and source use different terminologies — a game-changer in fields like legal research, where precise language varies across jurisdictions.

**Strengths:**

- Handles synonyms, paraphrasing, and conceptual queries naturally
- Works well for natural language questions over prose documents

**Weaknesses:**

- Embeddings average meaning across dimensions. Rare or unique tokens get diluted. This means exact identifiers like `ERROR_CODE_0427`, `SKU-9981`, or `retryCharge()` can get "washed out" in the embedding.
- Requires GPU infrastructure (at least at indexing time), an embedding model to maintain, and a vector database

---

### The Practical Consequence: Hybrid Retrieval

Because the two methods _fail in complementary ways_, the current best practice in production RAG systems is to use both.

Hybrid search runs a sparse retriever (BM25) and a dense vector retriever on the same query in parallel, then merges their result lists using a fusion algorithm like **Reciprocal Rank Fusion (RRF)** or convex combination. The combined candidate set is passed to the LLM, often after an optional reranking step. It improves recall over either method alone because dense retrieval misses exact keyword matches and sparse retrieval misses semantic synonyms.

Hybrid retrieval is the default recommended choice in 2026.

---

### Python Examples

**BM25 + Dense Retrieval with LangChain (simple hybrid):**

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Assume `docs` is a list of LangChain Document objects
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Combine: 40% BM25 weight, 60% dense/semantic weight
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

results = hybrid_retriever.invoke("how do I reset my invoice password?")
```

**Manual RRF fusion (lower-level, framework-agnostic):**

```python
def reciprocal_rank_fusion(result_lists: list[list], k: int = 60) -> list:
    """
    Fuse multiple ranked result lists using RRF.
    result_lists: each list is a ranked list of doc IDs
    """
    scores = {}
    for result_list in result_lists:
        for rank, doc_id in enumerate(result_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# Example usage
bm25_results   = ["doc_3", "doc_1", "doc_7"]   # from BM25
vector_results = ["doc_1", "doc_5", "doc_3"]   # from vector search

fused = reciprocal_rank_fusion([bm25_results, vector_results])
# Returns: ['doc_1', 'doc_3', 'doc_5', 'doc_7'] — doc_1 and doc_3 appear in both
```

---

### What You Need to Know as an AI Engineer

**1. Know when each method dominates.** If your queries are natural language questions over prose documents where paraphrasing is common, dense retrieval should dominate and sparse serves mainly as a safety net for exact-match queries. Conversely, if your corpus is full of structured content — error codes, SKUs, legal clause numbers, API names — BM25 is critical.

**2. Understand the infrastructure tradeoffs.** Term-based retrieval is operationally much simpler: Elasticsearch/OpenSearch, no GPU, real-time index updates. Dense retrieval needs a vector database (Pinecone, Qdrant, Weaviate, pgvector, FAISS), an embedding model that must stay consistent, and re-embedding when documents change.

**3. Embedding model consistency is non-negotiable.** The same embedding model that was used for the creation of the vector database must be used to translate the input query into a vector, as the similarity between query and chunks is measured using these vectors. Swapping models means re-indexing everything.

**4. Add a reranker for precision.** Hybrid retrieval maximizes recall so the relevant document is somewhere in your candidate set. A cross-encoder reranker then maximizes precision by scoring each query-document pair jointly, which is more accurate than embedding similarity but too slow for first-stage retrieval. The typical 3-stage pipeline is: hybrid retrieval → top-K candidates → reranker → top-N to LLM.

**5. There's a "learned sparse" middle ground.** A third category sits between sparse and dense: learned sparse retrieval models like **SPLADE** produce sparse vectors where the non-zero dimensions correspond to vocabulary terms, but the weights are learned by a transformer. SPLADE expands queries and documents with related terms from its learned vocabulary — a document about "gradient descent" gets non-zero weight for "optimizer," "backpropagation," and "learning rate." This is worth knowing about for advanced setups.

**6. Evaluate before adding complexity.** Start with dense-only if you have no labelled data and the corpus is prose-heavy; add BM25 hybrid if you see retrieval failures on keyword or entity queries; add a reranker if precision in the top-5 is the bottleneck. Evaluate each step with a held-out set of query-document relevance pairs before adding complexity — not every RAG system needs all three components.

---

### Quick Reference Summary

||Term-Based (BM25)|Embedding-Based (Dense)|Hybrid|
|---|---|---|---|
|**Mechanism**|Exact term matching|Semantic similarity|Both in parallel|
|**Strengths**|IDs, codes, exact phrases|Synonyms, paraphrasing|Best overall recall|
|**Weaknesses**|No semantic understanding|Dilutes rare tokens|More infra complexity|
|**Infrastructure**|Elasticsearch/OpenSearch|Vector DB + GPU|Both|
|**Best for**|Structured/technical docs|Natural language prose|Production RAG systems|
|**Python libs**|`rank_bm25`, Elasticsearch|`sentence-transformers`, FAISS, Qdrant|LangChain `EnsembleRetriever`, Haystack|

The bottom line: for any serious RAG system, hybrid retrieval is the default. Start by understanding why each method fails — that intuition is what lets you diagnose retrieval quality issues, which is one of the hardest and most impactful skills in RAG engineering.