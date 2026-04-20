# Retrieval optimization techniques

## Introduction

Retrieval-Augmented Generation (RAG) is the dominant paradigm for grounding LLM outputs in external, domain-specific, or up-to-date knowledge without costly fine-tuning. The quality of a RAG system is almost entirely gated by the quality of its retrieval step — a poorly retrieved context leads to hallucinations, irrelevant answers, or missed information, regardless of how powerful the generator model is.

This report covers every major optimization technique available as of 2025, organized from the foundational to the advanced. Each technique is explained in depth with Python code examples and practical guidance on when to apply it.

> **Core principle:** The retrieval stage must optimize for two competing objectives simultaneously — **recall**(did we find all the relevant documents?) and **precision** (are the documents we surfaced actually relevant?). Most techniques in this report are strategies for managing this trade-off.

---

## The Retrieval Pipeline

Before diving into individual techniques, it helps to have a mental model of the entire pipeline. A production-grade RAG system is not a single retrieval call — it is a multi-stage process:

```
User Query
    │
    ▼
[Query Transformation] ──── expand, rewrite, decompose
    │
    ▼
[Retrieval Stage 1]  ──── hybrid search (dense + sparse), broad recall
    │
    ▼
[Fusion & Deduplication] ─── RRF, merging results from multiple indexes
    │
    ▼
[Retrieval Stage 2: Reranking] ─── cross-encoder or LLM reranker
    │
    ▼
[Context Compression] ──── extract, summarize, distill
    │
    ▼
[LLM Generation] ──── grounded prompt with citations
    │
    ▼
[Validation / CRAG] ──── optional correctness check
```

Each stage in this pipeline is a lever you can tune. The sections below go deep on each one.

---

## 1. Chunking Strategies

### Why It Matters

Chunking is the process of splitting source documents into smaller units before embedding and indexing. The wrong chunking strategy is one of the most common causes of poor RAG performance. If chunks are too small, they lose context; if too large, they dilute the embedding signal and consume the LLM's context window with irrelevant text.

### Fixed-Size Chunking

The simplest approach: split every `N` tokens, with an optional overlap.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # tokens per chunk
    chunk_overlap=50,     # overlap to avoid cutting context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]
)

docs = splitter.split_text(raw_text)
```

**Pros:** Simple, fast, predictable.  
**Cons:** Splits mid-sentence or mid-paragraph, severing logical context. Best for uniform, plain-text corpora.

### Semantic Chunking

Instead of splitting on character count, split on _semantic boundaries_ — where the meaning changes. This is typically detected by embedding each sentence and identifying points where cosine similarity drops significantly.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95           # split at top 5% semantic shifts
)

docs = splitter.create_documents([raw_text])
```

**Pros:** Produces chunks with coherent semantic units.  
**Cons:** Slower (requires embedding every sentence) and can produce variable-length chunks.

### Document-Structure-Aware Chunking

For structured documents (PDFs, Markdown, HTML, code), use the document's own structure as chunk boundaries.

```python
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = splitter.split_text(markdown_text)
```

For PDFs, use a layout-aware parser like `unstructured` or `pymupdf` to preserve table structure, headers, and captions before chunking.

### Late Chunking (Token-Level Pooling)

A newer technique where a long document is first processed by an embedding model to produce _token-level embeddings_, and then chunks are created by mean-pooling token embeddings. This preserves global document context within each chunk's embedding, a significant improvement over independent chunk embedding.

```python
# Conceptual example using a long-context embedding model
from transformers import AutoTokenizer, AutoModel
import torch

def late_chunk_embed(text, model, tokenizer, chunk_size=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden)
    
    chunks = []
    for i in range(0, token_embeddings.shape[0], chunk_size):
        chunk_emb = token_embeddings[i:i+chunk_size].mean(dim=0)
        chunks.append(chunk_emb)
    return chunks
```

**Pros:** Each chunk's embedding is contextually aware of surrounding text.  
**Cons:** Requires models with long context windows; slower ingestion.

### Practical Guidance

|Document Type|Recommended Strategy|
|---|---|
|Plain text / articles|Recursive character splitter with overlap|
|Technical docs / manuals|Structure-aware (Markdown/HTML headers)|
|Legal / medical|Semantic chunking to preserve argument units|
|Code|Language-aware splitter (by function/class)|
|PDFs with tables|Layout-aware parser + structure splitting|

---

## 2. Embedding Model Selection & Fine-Tuning

### Why It Matters

The embedding model determines how well your vector space captures semantic similarity. A generic embedding model trained on Wikipedia-style text will perform poorly on highly specialized domains like patent law or genomics.

### Choosing a Base Model

As of 2025, strong open-source choices include:

- **`text-embedding-3-large`** (OpenAI) — best general-purpose commercial option
- **`bge-large-en-v1.5`** (BAAI/HuggingFace) — strong open-source baseline
- **`e5-mistral-7b-instruct`** (Microsoft) — excellent for instruction-following retrieval
- **`nomic-embed-text-v1.5`** — strong for long documents with Matryoshka support

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE models benefit from a query prefix
query_embedding = model.encode("Represent this sentence for retrieval: " + query)
doc_embeddings = model.encode(documents)
```

### Domain Fine-Tuning

For specialized domains, fine-tuning the embedding model on in-domain (query, positive_passage, negative_passage) triplets can dramatically improve recall.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Training data: (query, positive_doc, negative_doc) triplets
train_examples = [
    InputExample(texts=["patient hemoglobin levels", "HbA1c measures average glucose...", "weather report for Tuesday..."]),
    # ... more examples
]

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine_tuned_embeddings"
)
```

### Matryoshka Representation Learning (MRL)

MRL trains embeddings so that shorter vector prefixes (e.g., 256 dims from a 1536-dim vector) retain most of the retrieval quality. This enables a **coarse-to-fine** retrieval strategy where you first search with small vectors for speed, then rescore with the full vector for accuracy.

```python
from sentence_transformers import SentenceTransformer

# Models trained with MRL (e.g., nomic-embed, OpenAI text-embedding-3)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Use truncated dimensions for fast ANN search
small_embeddings = model.encode(docs, truncate_dim=256)
# Use full dimensions for reranking
full_embeddings = model.encode(docs, truncate_dim=768)
```

---

## 3. Hybrid Search (Dense + Sparse)

### Why It Matters

Dense vector search excels at semantic similarity — finding documents conceptually related to a query even when exact words differ. Sparse/keyword search (BM25) excels at exact-match precision — finding documents containing a specific product ID, person name, or technical term. Neither alone is sufficient for production systems.

**Dense fails on:** "What was the Q1 2026 revenue for ticker NVDA?" (exact number matching)  
**Sparse fails on:** "Show me documents about neural plasticity" when docs say "synaptic learning"

### BM25: The Sparse Retrieval Backbone

BM25 scores documents based on term frequency (TF) and inverse document frequency (IDF), normalized by document length.

```python
from rank_bm25 import BM25Okapi
import nltk

nltk.download("punkt")

def tokenize(text):
    return nltk.word_tokenize(text.lower())

tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "transformer attention mechanism"
tokenized_query = tokenize(query)
bm25_scores = bm25.get_scores(tokenized_query)

# Get top-k by BM25
top_k_bm25 = bm25_scores.argsort()[-10:][::-1]
```

### Combining Dense + Sparse with an Ensemble Retriever

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Build vector store
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Build BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

# Ensemble: equal weight by default
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]  # tune weights for your domain
)

results = hybrid_retriever.get_relevant_documents(query)
```

### Native Hybrid Search in Vector Databases

Modern vector databases like Weaviate, Qdrant, and Milvus support hybrid search natively, which is more efficient than Python-level fusion:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NamedSparseVector, NamedVector

client = QdrantClient(url="http://localhost:6333")

# Hybrid search with both dense and sparse vectors
results = client.query_points(
    collection_name="my_collection",
    prefetch=[
        {"query": sparse_vector, "using": "sparse", "limit": 20},
        {"query": dense_vector, "using": "dense", "limit": 20},
    ],
    query={"fusion": "rrf"},  # Reciprocal Rank Fusion
    limit=10,
)
```

---

## 4. Query Transformation Techniques

### Why It Matters

User queries are often vague, ambiguous, or phrased differently from the documents they're trying to match. Query transformation bridges this gap before retrieval even begins.

### Query Expansion

Add synonyms and related terms to broaden the query's coverage.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

expansion_prompt = PromptTemplate.from_template("""
Generate 3 alternative phrasings of the following query to improve document retrieval.
Return them as a JSON list.

Original query: {query}

Alternative phrasings:
""")

chain = expansion_prompt | llm

result = chain.invoke({"query": "how does attention work in transformers"})
# Returns: ["transformer self-attention mechanism", "multi-head attention architecture", 
#           "scaled dot-product attention computation"]
```

### Multi-Query Retrieval

For complex questions, decompose into multiple sub-queries and retrieve independently, then union the results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = vectorstore.as_retriever()
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# LangChain automatically decomposes the query and deduplicates results
docs = multi_query_retriever.get_relevant_documents(
    "What are the main differences between BERT and GPT architectures, "
    "and how do their training objectives differ?"
)
```

### HyDE: Hypothetical Document Embeddings

Instead of embedding the raw query, use an LLM to generate a _hypothetical answer_ — as if the perfect document existed — and then embed _that_ for retrieval. This dramatically improves semantic alignment because the hypothetical answer is in the same "language space" as actual documents.

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

base_embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

# Generates a hypothetical passage, embeds it, then queries the vector store
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    custom_prompt=PromptTemplate.from_template(
        "Write a technical passage that would answer: {question}\n\nPassage:"
    )
)

hyde_retriever = vectorstore.as_retriever(
    embedding=hyde_embedder
)
```

**When to use HyDE:** Highly effective for knowledge-heavy domains where users ask questions without using domain vocabulary. Less useful for factual lookups with exact terms.

### Step-Back Prompting

For narrow or overly-specific queries, step back to a broader, more abstract version first. Retrieve on the abstracted query, then answer the specific one.

```python
stepback_prompt = PromptTemplate.from_template("""
You are an expert. Generate a more general, higher-level question
that encompasses the specific question below. This will be used for
retrieval from a knowledge base.

Specific question: {question}
General question:
""")

stepback_chain = stepback_prompt | llm

# "What are the side effects of metformin in elderly patients?" 
# → "What are the general considerations for metformin use?"
general_q = stepback_chain.invoke({"question": specific_question})
```

---

## 5. Reranking (Two-Stage Retrieval)

### Why It Matters

The first retrieval stage (vector similarity or BM25) optimizes for speed and recall — it casts a wide net. But cosine similarity is an imperfect proxy for actual relevance. The "lost in the middle" problem is well-documented: LLMs pay less attention to documents positioned in the middle of a long context. Reranking is a second, more expensive but more accurate relevance judgment that selects the truly relevant documents from the broader candidate set.

### Cross-Encoder Reranking

A cross-encoder takes the query AND the document as a joint input and produces a relevance score. This is far more accurate than bi-encoder similarity scores because it models the interaction between query and document explicitly.

```python
from sentence_transformers import CrossEncoder

# Excellent open-source cross-encoder trained on MS MARCO
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Stage 1: Retrieve 20 candidates (high recall)
candidates = retriever.get_relevant_documents(query, k=20)

# Stage 2: Rerank with cross-encoder (high precision)
query_doc_pairs = [(query, doc.page_content) for doc in candidates]
scores = reranker.predict(query_doc_pairs)

# Sort and keep top-5 for generation
ranked_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
top_docs = [doc for _, doc in ranked_docs[:5]]
```

### ColBERT: Late Interaction Reranking

ColBERT is a more efficient alternative to cross-encoders. It encodes the query and document separately into token-level embeddings, then computes relevance via a late-interaction "MaxSim" operation. This allows precomputed document embeddings while still capturing fine-grained token matches.

```python
# Using RAGatouille — a ColBERT wrapper for RAG
from ragatouille import RAGPretrainedModel

colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index your documents
colbert.index(
    collection=documents,
    index_name="my_index",
    max_document_length=512,
)

# Retrieve and rerank in one step
results = colbert.search(query=query, k=5)
```

### API-Based Rerankers

For production use, commercial rerankers via API (no GPU required):

```python
import cohere

co = cohere.Client("YOUR_COHERE_API_KEY")

# Stage 1: Get candidates
candidates = retriever.get_relevant_documents(query, k=25)
doc_texts = [doc.page_content for doc in candidates]

# Stage 2: Cohere rerank
rerank_result = co.rerank(
    query=query,
    documents=doc_texts,
    top_n=5,
    model="rerank-english-v3.0"
)

top_docs = [candidates[r.index] for r in rerank_result.results]
```

### The "Lost in the Middle" Problem and Document Repacking

Research shows LLMs perform best when the most relevant context appears at the **beginning or end** of the prompt, not in the middle. After reranking, use a "sides" repacking strategy:

```python
def repack_sides(docs):
    """Place most relevant docs at head and tail, less relevant in the middle."""
    if len(docs) <= 2:
        return docs
    even_indexed = docs[::2]   # positions 0, 2, 4...  (high relevance → front)
    odd_indexed = docs[1::2]   # positions 1, 3, 5...  (lower relevance → back)
    return even_indexed + odd_indexed[::-1]

repacked = repack_sides(top_docs)
```

---

## 6. Metadata Filtering

### Why It Matters

Vector similarity search is semantic, but many queries have hard constraints that semantics cannot express: "only show me documents from 2024," "only legal opinions from the 9th Circuit," or "only documents tagged as internal policy." Metadata filtering applies these hard constraints _before or during_ vector search, drastically reducing the search space and eliminating irrelevant results.

### Attaching Metadata at Ingestion

```python
from langchain.schema import Document

documents = [
    Document(
        page_content="The Federal Reserve held rates steady in March 2025...",
        metadata={
            "source": "reuters",
            "date": "2025-03-20",
            "category": "finance",
            "region": "US",
            "doc_id": "reuters-20250320-fed"
        }
    ),
    # ...
]
```

### Filtered Vector Search

```python
# Chroma example: filter by metadata
results = vectorstore.similarity_search(
    query=query,
    k=10,
    filter={"category": "finance", "region": "US"}
)

# Pinecone example: structured metadata filter
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "date": {"$gte": "2025-01-01"},
        "category": {"$in": ["finance", "economics"]}
    }
)
```

### Self-Query Retriever: LLM-Powered Filter Extraction

Instead of requiring users to specify filters manually, use an LLM to extract them from the natural language query:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="The news source", type="string"),
    AttributeInfo(name="date", description="Publication date in YYYY-MM-DD format", type="string"),
    AttributeInfo(name="category", description="Topic category", type="string"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Financial and economic news articles",
    metadata_field_info=metadata_field_info,
    verbose=True
)

# The query "recent US finance news from 2025" automatically extracts:
# {"date": {"$gte": "2025-01-01"}, "category": "finance", "region": "US"}
docs = self_query_retriever.get_relevant_documents(
    "recent US finance news from 2025"
)
```

---

## 7. Parent-Document & Hierarchical Retrieval

### Why It Matters

There is a fundamental tension in chunking: **small chunks** produce high-quality, focused embeddings but lose surrounding context; **large chunks** preserve context but produce diluted embeddings that are harder to match precisely. The parent-document pattern resolves this by splitting at two granularities.

### The Pattern

1. **Index small "child" chunks** for high-precision embedding and retrieval.
2. **Store their full "parent" chunks** (or entire documents) separately.
3. At query time, retrieve by small chunk similarity, but return the corresponding parent chunk for generation.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent splitter: large chunks for context preservation
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Child splitter: small chunks for precise retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# In-memory or persistent doc store for parent docs
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(raw_documents)

# Retrieves by child chunk similarity, returns parent context
docs = retriever.get_relevant_documents(query)
```

### Hierarchical Summarization Trees (RAPTOR)

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) extends this concept into a full tree. Documents are clustered by semantic similarity, each cluster is summarized, and this process repeats recursively upward, forming a tree from leaves (individual chunks) to the root (global summary). At query time, the system retrieves from multiple levels of the tree.

```python
# Pseudocode for RAPTOR indexing
def build_raptor_tree(chunks, embedding_model, llm, n_levels=3):
    current_level = chunks
    all_nodes = list(chunks)
    
    for level in range(n_levels):
        # Embed current level
        embeddings = embedding_model.encode([c.text for c in current_level])
        
        # Cluster using GMM or k-means
        clusters = gaussian_mixture_cluster(embeddings)
        
        # Summarize each cluster
        summaries = []
        for cluster in clusters:
            cluster_text = "\n".join([c.text for c in cluster])
            summary = llm.summarize(cluster_text)
            summaries.append(Node(text=summary, level=level+1, children=cluster))
        
        all_nodes.extend(summaries)
        current_level = summaries
    
    return all_nodes
```

---

## 8. Contextual Retrieval

### Why It Matters

Standard chunking strips chunks of their surrounding context. A chunk like "The policy was amended in 1987" is meaningless without knowing _which policy_. Anthropic's Contextual Retrieval technique prepends each chunk with a concise, LLM-generated contextual summary that situates it within its source document.

### Implementation

```python
import anthropic
from langchain.schema import Document

client = anthropic.Anthropic()

CONTEXT_PROMPT = """
<document>
{full_document}
</document>

Here is a chunk from this document:
<chunk>
{chunk_text}
</chunk>

Write a short (2-3 sentence) contextual description that situates this chunk
within the document. Focus on making the chunk more findable via retrieval.
Respond only with the context, no preamble.
"""

def add_chunk_context(full_doc: str, chunk: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast, cheap model for this step
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": CONTEXT_PROMPT.format(
                full_document=full_doc[:4000],  # truncate for context window
                chunk_text=chunk
            )
        }]
    )
    context = response.content[0].text
    return f"{context}\n\n{chunk}"

# Apply at ingestion time
contextualized_docs = []
for doc in raw_documents:
    for chunk in splitter.split_text(doc.page_content):
        contextualized_chunk = add_chunk_context(doc.page_content, chunk)
        contextualized_docs.append(Document(
            page_content=contextualized_chunk,
            metadata=doc.metadata
        ))
```

This technique is especially effective when combined with hybrid search and reranking, as the richer chunk text improves both BM25 and semantic matching.

---

## 9. Graph-Based Retrieval (GraphRAG)

### Why It Matters

Standard RAG treats documents as isolated units. But real knowledge is relational — understanding a legal case requires knowing related precedents; understanding a company requires knowing its organizational hierarchy. Graph-based retrieval captures these relationships explicitly.

### Microsoft's GraphRAG Approach

GraphRAG (introduced by Microsoft Research, 2024) builds a knowledge graph from the document corpus, then uses it for both local (entity-specific) and global (theme-level) retrieval.

```python
# Using the official graphrag library
# pip install graphrag

# 1. Index: builds a knowledge graph with entity extraction
# graphrag index --root ./my_project

# 2. Query: uses the graph for retrieval
# graphrag query --root ./my_project --method global --query "What are the main themes?"

# Programmatic usage
import asyncio
from graphrag.query.structured_search.global_search.search import GlobalSearch

async def graph_query(query: str):
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=tiktoken.get_encoding("cl100k_base"),
        max_data_tokens=12_000,
        map_llm_params={"max_tokens": 1000, "temperature": 0.0},
        reduce_llm_params={"max_tokens": 2000, "temperature": 0.0},
    )
    result = await search_engine.asearch(query)
    return result.response
```

### Lightweight Entity-Aware Retrieval with Neo4j

For custom graph RAG without the full GraphRAG framework:

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Extract entities from query and traverse graph
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    return_direct=False
)

result = chain.invoke({
    "query": "What companies are associated with Sam Altman and what are their relationships?"
})
```

---

## 10. Adaptive & Corrective RAG

### Adaptive RAG

Not every query needs retrieval. Simple greetings, well-known facts, or follow-up questions in an ongoing conversation may be answerable from the LLM's parametric memory. Adaptive RAG uses a classifier to decide _whether_ to retrieve and _from where_.

```python
from langchain_openai import ChatOpenAI
from enum import Enum

class RetrievalDecision(str, Enum):
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    WEB_SEARCH = "web_search"

ROUTING_PROMPT = """
Classify the following user query. Should we:
- 'retrieve': query a local knowledge base (for specific domain knowledge)
- 'no_retrieve': answer directly from LLM knowledge (for general knowledge)
- 'web_search': search the live web (for current events)

Query: {query}

Classification (one of: retrieve, no_retrieve, web_search):
"""

def route_query(query: str) -> RetrievalDecision:
    response = llm.invoke(ROUTING_PROMPT.format(query=query))
    return RetrievalDecision(response.content.strip().lower())

decision = route_query("What is the capital of France?")  # → no_retrieve
decision = route_query("What is our Q3 refund policy?")   # → retrieve
decision = route_query("What happened in the news today?") # → web_search
```

### Corrective RAG (CRAG)

CRAG adds a retrieval confidence evaluator. If retrieved documents are judged as irrelevant or low quality, it triggers a fallback action (web search, query rewriting, etc.) before generating.

```python
RELEVANCE_EVALUATOR_PROMPT = """
Given the query and retrieved document, assess whether the document is relevant.

Query: {query}
Document: {document}

Answer with: 'relevant', 'partially_relevant', or 'irrelevant'.
"""

def evaluate_relevance(query: str, doc: str) -> str:
    response = llm.invoke(
        RELEVANCE_EVALUATOR_PROMPT.format(query=query, document=doc[:1000])
    )
    return response.content.strip().lower()

def corrective_rag_pipeline(query: str) -> str:
    # Stage 1: Retrieve
    docs = retriever.get_relevant_documents(query)
    
    # Stage 2: Evaluate each doc
    relevant_docs = []
    for doc in docs:
        score = evaluate_relevance(query, doc.page_content)
        if score == "relevant":
            relevant_docs.append(doc)
        elif score == "partially_relevant":
            # Decompose and recompose: extract key facts only
            relevant_docs.append(extract_key_facts(doc))
    
    # Stage 3: Fallback if no relevant docs found
    if not relevant_docs:
        # Trigger web search or query rewriting
        relevant_docs = web_search_retriever.get_relevant_documents(query)
    
    # Stage 4: Generate
    return generate_response(query, relevant_docs)
```

### Self-RAG

Self-RAG trains the LLM itself to decide when to retrieve and to critique its own outputs using special "reflection tokens." While the full Self-RAG system requires a specifically fine-tuned model, its reflection principles can be approximated with prompt engineering:

```python
SELF_RAG_GENERATION_PROMPT = """
Answer the question using the provided context.

Context:
{context}

Question: {question}

Instructions:
1. First, determine if the context is sufficient to answer the question.
2. If yes, answer the question and provide a citation.
3. If no, state what information is missing.
4. After answering, rate your confidence (High/Medium/Low) and explain why.

Response:
"""
```

---

## 11. Context Compression & Distillation

### Why It Matters

Even after reranking, retrieved chunks often contain noise — sentences unrelated to the specific query. Feeding a 2,000-token chunk to an LLM when only 200 tokens are actually relevant wastes the context window, increases latency, and can confuse the generator. Context compression extracts only the relevant portions.

### LLM-Based Compressor

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Returns only the relevant sentences from each chunk
compressed_docs = compression_retriever.get_relevant_documents(query)
```

### Embeddings Filter Compressor (Faster, No LLM Call)

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76  # drop sentences below this relevance threshold
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)
```

### Context Distillation with RECOMP

RECOMP (Retrieve, Compress, Prepend) trains a small summarization model specifically for compressing retrieved documents into concise, query-conditioned summaries. Research from the best practices paper found RECOMP outperformed naive truncation significantly on downstream generation quality.

```python
# Abstractive compression using a fine-tuned summarizer
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"  # or a RECOMP-fine-tuned model
)

def compress_docs(query: str, docs: list, max_length: int = 150) -> str:
    combined = " ".join([d.page_content for d in docs])
    prompt = f"Query: {query}\n\nDocument: {combined}\n\nRelevant summary:"
    summary = summarizer(prompt, max_length=max_length, min_length=40)[0]["summary_text"]
    return summary
```

---

## 12. Reciprocal Rank Fusion (RRF)

### Why It Matters

When combining results from multiple retrievers (BM25 + dense vectors, or multiple embedding models), you need a principled way to merge the ranked lists. Simple score averaging fails because BM25 and cosine similarity scores are not on the same scale. RRF is a rank-based fusion method that is robust to this problem and empirically outperforms most weighting heuristics.

### RRF Formula

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

Where `k` is a constant (typically 60) and `rank_i` is the document's rank in list `i`.

### Implementation

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int = 60,
    id_fn=lambda x: x.metadata.get("doc_id", x.page_content[:100])
) -> list:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of ranked document lists (each from a different retriever)
        k: Constant to avoid high influence of top-ranked items (default 60)
    
    Returns:
        Fused and re-ranked document list
    """
    scores = defaultdict(float)
    doc_map = {}
    
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = id_fn(doc)
            scores[doc_id] += 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc  # store the actual document
    
    # Sort by fused score descending
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# Usage
bm25_results = bm25_retriever.get_relevant_documents(query)
dense_results = dense_retriever.get_relevant_documents(query)
colbert_results = colbert_retriever.get_relevant_documents(query)

fused_results = reciprocal_rank_fusion([
    bm25_results,
    dense_results,
    colbert_results
])
```

---

## 13. Approximate Nearest Neighbor (ANN) Indexing

### Why It Matters

Exact k-NN search over millions of vectors is prohibitively slow (O(n) per query). ANN indexes trade a small amount of accuracy for massive speed improvements (O(log n) or near-constant), making real-time retrieval feasible at scale.

### HNSW (Hierarchical Navigable Small World)

HNSW is the most widely used ANN index in production RAG systems. It builds a multi-layer graph where edges connect nodes to their approximate nearest neighbors.

```python
import hnswlib
import numpy as np

# Build index
dim = 1536  # embedding dimension
num_elements = len(embeddings)

index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(
    max_elements=num_elements,
    ef_construction=400,  # accuracy vs. build time tradeoff
    M=64                  # number of connections per node (higher = more accurate but more memory)
)
index.add_items(embeddings, ids=list(range(num_elements)))

# Search
index.set_ef(200)  # higher ef = more accurate search, higher latency
labels, distances = index.knn_query(query_embedding, k=10)
```

Most vector databases (Chroma, Qdrant, Weaviate, Pinecone, Milvus) use HNSW internally.

### Key Tuning Parameters

|Parameter|Effect|
|---|---|
|`M`|Graph connectivity. Higher = better recall but more memory/build time. Range: 16–64.|
|`ef_construction`|Build-time search scope. Higher = better index quality but slower build. Range: 100–500.|
|`ef` (query time)|Query-time search scope. Higher = better recall but more latency. Tune for your SLA.|

### Product Quantization (PQ) for Memory Efficiency

For very large corpora (100M+ vectors), compress vectors with PQ to fit in memory:

```python
import faiss

# Build IVF-PQ index for billion-scale retrieval
nlist = 4096   # number of inverted list clusters
m = 8          # number of PQ subspaces (embedding_dim must be divisible by m)
bits = 8       # bits per subspace

quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)

index.train(embeddings[:100_000])  # needs training
index.add(embeddings)
index.nprobe = 64  # number of clusters to search (accuracy vs. speed tradeoff)

D, I = index.search(query_embedding.reshape(1, -1), k=10)
```

---

## 14. Feedback Loops & Continuous Improvement

### Why It Matters

A RAG system deployed in production will drift over time as the document corpus changes, user query patterns evolve, and edge cases are discovered. A robust system builds in mechanisms to detect and fix degradation automatically.

### Query Telemetry & Performance Logging

```python
import time
import json
from dataclasses import dataclass, asdict

@dataclass
class RetrievalEvent:
    query: str
    retrieved_doc_ids: list[str]
    retrieval_latency_ms: float
    reranker_scores: list[float]
    llm_response: str
    user_feedback: str | None = None  # "positive" / "negative" / None
    timestamp: float = None

    def __post_init__(self):
        self.timestamp = time.time()

def log_retrieval_event(event: RetrievalEvent):
    with open("retrieval_telemetry.jsonl", "a") as f:
        f.write(json.dumps(asdict(event)) + "\n")
```

### Automated Evaluation with RAGAS

RAGAS is the standard framework for evaluating RAG pipelines, measuring faithfulness, answer relevancy, context precision, and context recall.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Build evaluation dataset from your logs
eval_data = {
    "question": [...],
    "answer": [...],       # LLM's response
    "contexts": [...],     # retrieved chunks
    "ground_truth": [...], # known correct answers
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
])

print(results)
# Output:
# {
#   "faithfulness": 0.87,
#   "answer_relevancy": 0.91,
#   "context_precision": 0.78,
#   "context_recall": 0.84
# }
```

### Embedding Model Retraining from Feedback

Positive and negative user feedback can generate new fine-tuning pairs for your embedding model:

```python
# Extract hard negatives from failed retrievals
def generate_finetuning_pairs_from_logs(telemetry_log: str):
    with open(telemetry_log) as f:
        events = [json.loads(line) for line in f]
    
    training_pairs = []
    for event in events:
        if event["user_feedback"] == "negative":
            # The retrieved docs were wrong → hard negative pairs
            training_pairs.append({
                "query": event["query"],
                "positive": event.get("correct_doc_id"),  # from human review
                "negative": event["retrieved_doc_ids"][0] # top retrieved (wrong)
            })
    return training_pairs
```

---

## Putting It All Together

Here is a complete production-grade RAG pipeline that incorporates the key techniques:

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain.schema import Document
import cohere

# ── Configuration ─────────────────────────────────────────
EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(model="gpt-4o", temperature=0)
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
INITIAL_K = 20
FINAL_K = 5

# ── Retrieval Pipeline ─────────────────────────────────────
def production_rag_pipeline(query: str, vectorstore, bm25_retriever) -> str:
    
    # Step 1: Route (adaptive)
    if is_general_knowledge(query):
        return LLM.invoke(query).content
    
    # Step 2: Query transformation (HyDE for semantic-heavy queries)
    hyde_query = generate_hypothetical_document(query)
    
    # Step 3: Hybrid retrieval
    dense_results = vectorstore.similarity_search(hyde_query, k=INITIAL_K)
    sparse_results = bm25_retriever.get_relevant_documents(query)  # use original
    
    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion([dense_results, sparse_results])[:INITIAL_K]
    
    # Step 5: Cross-encoder reranking
    pairs = [(query, doc.page_content) for doc in fused]
    scores = RERANKER.predict(pairs)
    ranked = sorted(zip(scores, fused), reverse=True)
    top_docs = [doc for _, doc in ranked[:FINAL_K]]
    
    # Step 6: Context compression
    compressed = compress_to_relevant_sentences(query, top_docs)
    
    # Step 7: Repacking (sides strategy)
    final_context = repack_sides(compressed)
    
    # Step 8: Generate with grounded prompt
    context_str = "\n\n---\n\n".join([d.page_content for d in final_context])
    response = LLM.invoke(
        f"Answer the question using ONLY the context below. Cite sources.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {query}"
    )
    
    # Step 9: Log for evaluation
    log_retrieval_event(RetrievalEvent(
        query=query,
        retrieved_doc_ids=[d.metadata.get("doc_id") for d in top_docs],
        retrieval_latency_ms=...,
        reranker_scores=[s for s, _ in ranked[:FINAL_K]],
        llm_response=response.content
    ))
    
    return response.content
```

---

## Evaluation Metrics

|Metric|Measures|Tool|
|---|---|---|
|**Context Precision**|What fraction of retrieved docs are actually relevant|RAGAS|
|**Context Recall**|What fraction of relevant docs were retrieved|RAGAS|
|**Faithfulness**|Does the answer stay grounded in context (no hallucination)?|RAGAS|
|**Answer Relevancy**|Does the answer actually answer the question?|RAGAS|
|**MRR (Mean Reciprocal Rank)**|How high does the first relevant doc appear?|Custom|
|**NDCG@k**|Normalized ranked retrieval quality|`sklearn` / custom|
|**Latency (p50/p99)**|End-to-end and per-stage timing|Observability layer|

---

## Summary Table

|Technique|Primary Benefit|Complexity|Cost Impact|
|---|---|---|---|
|Semantic Chunking|Better embedding coherence|Medium|Low|
|Contextual Retrieval|Situates orphaned chunks|Medium|Medium (LLM calls at ingestion)|
|Hybrid Search (BM25 + Dense)|Combines exact + semantic recall|Medium|Low|
|HyDE|Bridges query-doc vocabulary gap|Low|Low (fast LLM)|
|Multi-Query|Better coverage on complex queries|Low|Low|
|Cross-Encoder Reranking|High-precision final context|Medium|Medium (GPU or API)|
|ColBERT|Efficient high-accuracy reranking|High|Medium|
|Metadata Filtering|Hard constraint enforcement|Low|Very Low|
|Parent-Document Retrieval|Context preservation at generation|Medium|Low|
|RRF Fusion|Principled multi-retriever merging|Low|Very Low|
|GraphRAG|Multi-hop relational reasoning|High|High|
|Adaptive RAG|Avoids unnecessary retrieval|Medium|Low|
|Corrective RAG|Reduces hallucination from bad retrieval|Medium|Medium|
|Context Compression|Reduced noise in LLM context|Medium|Low–Medium|
|ANN Tuning (HNSW)|Sub-millisecond retrieval at scale|Low|Very Low|
|Feedback Loops + RAGAS|Continuous quality monitoring|High|Medium|

---

_Report compiled April 2025 based on current research and production best practices._