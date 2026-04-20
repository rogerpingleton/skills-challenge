# Vector database implementation

## 1. The RAG Architecture at a Glance

RAG gives an LLM access to external, up-to-date, or private knowledge without retraining. The basic pipeline has two phases:

### Indexing Phase (offline / batch)

```
Raw Documents
    │
    ▼
 Chunking
    │
    ▼
 Embedding Model  ──►  Dense Vector (float32[], e.g. 1536-dim)
    │
    ▼
 Vector Database  (store vector + original text + metadata)
```

### Retrieval Phase (online / per query)

```
User Query
    │
    ▼
 Embedding Model  ──►  Query Vector
    │
    ▼
 Vector Database ANN Search  ──►  Top-K Chunks
    │
    ▼
 (Optional) Reranker / Metadata Filter
    │
    ▼
 Prompt Assembly  ──►  LLM  ──►  Response
```

The key insight: instead of relying solely on training data, the LLM is grounded in retrieved context, reducing hallucinations and enabling access to domain-specific or recent information.

---

## 2. What Is a Vector Database?

A vector database is a storage and retrieval system optimized for **high-dimensional float arrays** (vectors). Unlike a relational database that searches by equality or range, a vector DB searches by **similarity** — finding vectors geometrically close to a query vector.

Core capabilities you need from a vector DB:

|Capability|Why It Matters|
|---|---|
|Approximate Nearest Neighbor (ANN) search|Exact search is O(n·d); ANN makes it sub-linear|
|Metadata filtering|Pre- or post-filter results by document type, date, source, etc.|
|Hybrid search (dense + sparse)|Combines semantic similarity with keyword precision|
|CRUD for vectors|Add, update, delete as knowledge base evolves|
|Persistence & durability|Survive restarts; replicate across nodes|
|Horizontal scalability|Handle hundreds of millions to billions of vectors|

---

## 3. Embeddings: The Foundation

An **embedding model** maps text (or images, audio) into a dense vector such that semantically similar inputs are geometrically close. Choosing the right model is as important as choosing the right DB.

### Popular Embedding Models (2025–2026)

|Model|Dims|Notes|
|---|---|---|
|`text-embedding-3-small` (OpenAI)|1536|Good general-purpose; very cost-effective|
|`text-embedding-3-large` (OpenAI)|3072|Higher accuracy; heavier|
|`text-embedding-ada-002` (OpenAI)|1536|Older but still widely deployed|
|`voyage-3` (Voyage AI)|1024|Excellent for RAG; strong MTEB scores|
|`nomic-embed-text-v1.5` (Nomic)|768|Open-source; competitive with commercial|
|`all-MiniLM-L6-v2` (sentence-transformers)|384|Tiny, fast, great for local/edge|
|`BAAI/bge-large-en-v1.5`|1024|Strong open-source alternative|
|`intfloat/e5-large-v2`|1024|Instruction-tuned; handles query/passage asymmetry|

> **Practical rule:** Check the MTEB Leaderboard for your task type (retrieval, semantic similarity, clustering). Don't default to OpenAI embeddings without evaluating OSS alternatives — especially for privacy-sensitive or cost-sensitive deployments.

### Generating Embeddings in Python

```python
# Option 1: sentence-transformers (local, no API key)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
texts = ["The capital of France is Paris.", "How do neural networks learn?"]
embeddings = model.encode(texts, normalize_embeddings=True)
print(embeddings.shape)  # (2, 1024)


# Option 2: OpenAI API
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

def embed(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

vecs = embed(["Explain transformers in ML."])
print(len(vecs[0]))  # 1536


# Option 3: Voyage AI (strong for RAG)
import voyageai

vo = voyageai.Client()  # reads VOYAGE_API_KEY from env
result = vo.embed(["Paris is in France."], model="voyage-3", input_type="document")
print(len(result.embeddings[0]))  # 1024
```

### Asymmetric Embedding (Query vs. Document)

Many production models use different prompts for query and document encoding. This is critical for retrieval quality:

```python
# BGE-style instruction prompting
query_instruction = "Represent this sentence for searching relevant passages: "
doc_instruction = ""  # BGE large doesn't need one for docs

query_vec = model.encode(query_instruction + "What is RAG?", normalize_embeddings=True)
doc_vec = model.encode("Retrieval-Augmented Generation is a technique...", normalize_embeddings=True)
```

---

## 4. Chunking Strategies

Chunking is the step that determines **what** gets embedded and stored. Poor chunking is often the root cause of poor RAG performance — not the model or DB.

### The Core Tradeoff

- **Too small (< 100 tokens):** Retrieves with high precision but lacks context for the LLM to generate a useful answer.
- **Too large (> 2,500 tokens):** The "context cliff" — LLM response quality degrades as context grows; also dilutes relevance.
- **Sweet spot:** 256–512 tokens with 10–25% overlap (i.e., 50–128 tokens).

### Strategy Comparison

|Strategy|When to Use|Library|
|---|---|---|
|**Fixed-size / Character**|Never for production; use as a last resort|Manual|
|**RecursiveCharacterTextSplitter**|Default starting point — fast, reliable|LangChain|
|**Token-based**|When you have hard token-budget constraints|LangChain, tiktoken|
|**MarkdownHeaderTextSplitter**|Docs, wikis, README files — highest easy win|LangChain|
|**Semantic chunking**|Thematically dense content (papers, books)|LlamaIndex SemanticSplitterNodeParser|
|**Late chunking**|Content with heavy cross-references (pronouns, headings)|Jina AI late-chunking|
|**Hierarchical / Parent-child**|When you need recall at different granularities|LlamaIndex|

> **2026 benchmark finding:** A Vectara NAACL 2025 peer-reviewed study found that **recursive/fixed chunking at 512 tokens consistently matched or beat semantic chunking** on end-to-end answer accuracy. Semantic chunking scored higher in retrieval recall but produced 43-token fragments that gave the LLM too little context to answer correctly. Start recursive, and only upgrade to semantic if your metrics justify it.

### Code: The Three Most Common Strategies

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# --- Strategy 1: Recursive (your default) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # in characters (tune to your embedding model's token limit)
    chunk_overlap=64,     # ~10-15% overlap to preserve context across boundaries
    separators=["\n\n", "\n", ". ", " ", ""],  # tries each in order
)
chunks = splitter.split_text(document_text)


# --- Strategy 2: Markdown-aware (best for structured docs) ---
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)
md_chunks = md_splitter.split_text(markdown_text)
# Each chunk carries header metadata for filtering


# --- Strategy 3: Token-based (hard budget) ---
import tiktoken
from langchain_text_splitters import TokenTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")  # matches text-embedding-3-small
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=400,   # tokens
    chunk_overlap=40,
)
token_chunks = token_splitter.split_text(document_text)


# --- Chunk Inspection Utility ---
def inspect_chunks(chunks: list[str]) -> None:
    token_counts = [len(tokenizer.encode(c)) for c in chunks]
    print(f"Total chunks: {len(chunks)}")
    print(f"Token stats: min={min(token_counts)}, max={max(token_counts)}, "
          f"avg={sum(token_counts)/len(token_counts):.0f}")
```

---

## 5. Indexing Algorithms

Once you have vectors, the vector DB uses an **index structure** to support efficient approximate nearest neighbor (ANN) search.

### HNSW — Hierarchical Navigable Small World

The dominant algorithm for most production RAG workloads.

**How it works:** Builds a multi-layered graph. Top layers have few nodes with long-range connections (fast navigation); lower layers have denser local connections (fine search). Queries start at the top layer and "zoom in" layer by layer.

```
Layer 2 (sparse, long-range):  A ─────────────── Z
Layer 1 (medium):              A ── D ── H ── Z
Layer 0 (dense, local):        A─B─C─D─E─F─G─H─I─J─K─Z
```

**Key parameters:**

|Parameter|Effect|Typical Value|
|---|---|---|
|`M`|Connections per node per layer. Higher = more accurate but more RAM.|16–64|
|`ef_construction`|Search width during index build. Higher = better quality, slower build.|100–400|
|`ef_search` (or `ef`)|Search width at query time. Tune recall vs. latency.|50–200|

**Strengths:** High recall, dynamic inserts (no retraining), handles arbitrary distance metrics, fast at moderate scale.  
**Weaknesses:** High RAM usage `((d×4 + M×2×4) bytes/vector)`; no native vector deletion (mark + rebuild).

### IVF — Inverted File Index

Clusters vectors into `nlist` buckets using k-means. At query time, only searches `nprobe` nearest clusters.

**Key parameters:**

|Parameter|Effect|
|---|---|
|`nlist`|Number of clusters. Rule of thumb: `sqrt(N)` to `4*sqrt(N)`.|
|`nprobe`|Clusters to search at query time. Higher = better recall, slower.|

**Strengths:** Very fast for huge datasets (billions of vectors); GPU-acceleratable via FAISS.  
**Weaknesses:** Requires training (k-means); fixed cluster structure doesn't adapt well to dynamic inserts.

### IVF-PQ — IVF with Product Quantization

Adds vector compression: each vector is split into `M` sub-vectors, each quantized to 8 bits. Reduces storage by ~8–32×.

```
Full vector (1536 float32 = 6,144 bytes)
  ──► IVF-PQ (M=96, 8-bit codes = 96 bytes)  →  64× compression
```

**When to use:** You have millions+ of vectors and RAM is the bottleneck.

### DiskANN

Stores most of the index on disk; keeps only hot data in RAM. Enables billion-scale search with commodity hardware.

**Where it's available:** Milvus, Azure AI Search, Qdrant (experimental).

### Quick Selection Guide

```
Dataset Size          RAM Available    Recommended Index
─────────────────     ─────────────    ──────────────────
< 1M vectors          Plentiful        HNSW (flat or in DB)
1M – 100M vectors     Moderate         HNSW or IVF-HNSW
100M – 1B vectors     Limited          IVF-PQ (FAISS / Milvus)
> 1B vectors          Any              DiskANN or distributed IVF-PQ
```

---

## 6. Distance / Similarity Metrics

The choice of metric must match your embedding model's training objective.

|Metric|Formula (intuition)|Use When|
|---|---|---|
|**Cosine similarity**|Angle between vectors|Normalized embeddings (most text models)|
|**Dot product (IP)**|Magnitude × alignment|Un-normalized; used by some instruction-tuned models|
|**L2 (Euclidean)**|Straight-line distance|Models not trained for cosine; image embeddings|

> Most sentence-transformer and OpenAI models produce unit-normalized vectors by default, making cosine similarity and dot product equivalent. Always check your model's card.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# For normalized vectors, these are identical
a = np.array([0.6, 0.8])  # unit norm
b = np.array([0.8, 0.6])  # unit norm
print(cosine_similarity(a, b))  # 0.96
print(dot_product(a, b))        # 0.96
```

---

## 7. Choosing a Vector Database

### Feature Matrix

|Database|Type|Scale|Hybrid Search|Metadata Filter|Managed Option|Python SDK|
|---|---|---|---|---|---|---|
|**ChromaDB**|OSS|Small–Med|Basic|Yes|No (self-host)|`chromadb`|
|**Qdrant**|OSS|Large|Yes (BM42+dense)|Yes (payload index)|Qdrant Cloud|`qdrant-client`|
|**Milvus**|OSS|Massive|Yes|Yes|Zilliz Cloud|`pymilvus`|
|**Pinecone**|Managed SaaS|Large|Yes|Yes|Yes (fully managed)|`pinecone`|
|**Weaviate**|OSS|Large|Yes|Yes|Weaviate Cloud|`weaviate-client`|
|**pgvector**|Postgres ext.|Med|With pg_trgm|Full SQL|RDS / Supabase|`psycopg2`/`asyncpg`|
|**FAISS**|Library|Massive|No (library only)|No (DIY)|No|`faiss-cpu` / `faiss-gpu`|
|**Redis (VSS)**|OSS + Cloud|Med|Yes|Yes|Redis Cloud|`redis`|

### Decision Framework Summary

```
Are you prototyping or building a local demo?
  → ChromaDB (zero setup, in-process)

Do you need full SQL + vector in one system?
  → pgvector (especially if you're already on Postgres)

Do you need production-grade OSS with no vendor lock-in?
  → Qdrant (best-in-class hybrid search + filtering)

Do you need massive scale (100M+ vectors) OSS?
  → Milvus (most horizontally scalable OSS option)

Do you need zero-infra, hands-off managed service?
  → Pinecone (best managed DX; higher cost at scale)

Do you need billion-scale with fine-grained GPU control?
  → FAISS (pure library; wrap in your own service)
```

---

## 8. Concrete Implementation Examples

### 8.1 ChromaDB — Local / Prototype

Best for: local development, small knowledge bases, Jupyter notebooks.

```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import Settings

# --- Setup ---
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

# Persistent client (survives restarts)
client = chromadb.PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # metric choice
)

# --- Ingest Documents ---
documents = [
    "RAG combines retrieval with generation to reduce hallucinations.",
    "HNSW is a graph-based ANN algorithm used in most vector databases.",
    "Chunking splits documents into smaller pieces before embedding.",
    "Product Quantization compresses vectors by encoding sub-vectors.",
]

metadatas = [
    {"source": "rag_intro.md", "topic": "rag"},
    {"source": "indexing.md", "topic": "indexing"},
    {"source": "chunking.md", "topic": "chunking"},
    {"source": "compression.md", "topic": "indexing"},
]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))],
)

print(f"Collection size: {collection.count()}")  # 4

# --- Query ---
results = collection.query(
    query_texts=["How do I make vector search faster?"],
    n_results=2,
    where={"topic": "indexing"},  # metadata filter
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"[{dist:.4f}] ({meta['source']}) {doc[:80]}")

# --- Update & Delete ---
collection.update(
    ids=["doc_0"],
    documents=["RAG grounds LLM responses in retrieved external knowledge."],
)

collection.delete(ids=["doc_3"])
print(f"After delete: {collection.count()}")  # 3
```

### 8.2 Qdrant — Production-Grade OSS

Best for: production RAG, hybrid search, advanced metadata filtering.

```bash
pip install qdrant-client sentence-transformers
# Run Qdrant locally: docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, ScoredPoint,
)
from sentence_transformers import SentenceTransformer
from uuid import uuid4

# --- Setup ---
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
COLLECTION = "rag_docs"
VECTOR_SIZE = 1024  # BGE large output dim

# Create collection (idempotent with recreate=False logic)
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            # HNSW params (optional tuning)
            hnsw_config={"m": 16, "ef_construct": 200},
        ),
    )
    # Create payload index for fast metadata filtering
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="topic",
        field_schema="keyword",
    )

# --- Ingest ---
docs = [
    {"text": "RAG reduces hallucinations by grounding responses in retrieved data.", 
     "source": "rag.md", "topic": "rag", "year": 2024},
    {"text": "HNSW builds a hierarchical graph for ANN search.", 
     "source": "hnsw.md", "topic": "indexing", "year": 2023},
    {"text": "Recursive chunking is the recommended default strategy.", 
     "source": "chunking.md", "topic": "chunking", "year": 2025},
    {"text": "Hybrid search combines BM25 keyword search with dense embeddings.", 
     "source": "hybrid.md", "topic": "retrieval", "year": 2025},
    {"text": "Reranking with cross-encoders improves top-k precision after retrieval.", 
     "source": "rerank.md", "topic": "retrieval", "year": 2025},
]

texts = [d["text"] for d in docs]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

points = [
    PointStruct(
        id=str(uuid4()),
        vector=vec,
        payload={k: v for k, v in doc.items() if k != "text"} | {"text": doc["text"]},
    )
    for doc, vec in zip(docs, embeddings)
]

client.upsert(collection_name=COLLECTION, points=points)
print(f"Inserted {len(points)} points")

# --- Basic Similarity Search ---
query = "What algorithm is used for fast vector search?"
query_vec = model.encode(query, normalize_embeddings=True).tolist()

results: list[ScoredPoint] = client.search(
    collection_name=COLLECTION,
    query_vector=query_vec,
    limit=3,
    with_payload=True,
)

print("\n=== Top-3 Results ===")
for r in results:
    print(f"  score={r.score:.4f} | {r.payload['text'][:70]}")

# --- Filtered Search ---
filtered = client.search(
    collection_name=COLLECTION,
    query_vector=query_vec,
    query_filter=Filter(
        must=[FieldCondition(key="topic", match=MatchValue(value="retrieval"))]
    ),
    limit=3,
    with_payload=True,
)

print("\n=== Filtered (topic=retrieval) ===")
for r in filtered:
    print(f"  score={r.score:.4f} | {r.payload['text'][:70]}")

# --- Scroll (list all docs) ---
all_points, _ = client.scroll(
    collection_name=COLLECTION,
    limit=100,
    with_payload=True,
)
print(f"\nTotal docs in collection: {len(all_points)}")

# --- Delete by filter ---
client.delete(
    collection_name=COLLECTION,
    points_selector=Filter(
        must=[FieldCondition(key="year", match=MatchValue(value=2023))]
    ),
)
```

### 8.3 Pinecone — Managed Cloud

Best for: teams that want zero infrastructure overhead.

```bash
pip install pinecone openai
```

```python
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os

# --- Setup ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
oai = OpenAI()

INDEX_NAME = "rag-knowledge-base"

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# --- Helper: batch embed ---
def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(
        input=texts, model="text-embedding-3-small"
    )
    return [item.embedding for item in resp.data]

# --- Ingest with metadata ---
documents = [
    {"id": "chunk_001", "text": "RAG reduces hallucinations.", "source": "intro.md"},
    {"id": "chunk_002", "text": "HNSW is fast for ANN search.", "source": "index.md"},
    {"id": "chunk_003", "text": "Use 512 tokens as default chunk size.", "source": "chunking.md"},
]

vectors = embed_batch([d["text"] for d in documents])

index.upsert(
    vectors=[
        {
            "id": doc["id"],
            "values": vec,
            "metadata": {"text": doc["text"], "source": doc["source"]},
        }
        for doc, vec in zip(documents, vectors)
    ]
)

print(index.describe_index_stats())

# --- Query ---
query_vec = embed_batch(["How do I pick a chunk size?"])[0]

response = index.query(
    vector=query_vec,
    top_k=3,
    include_metadata=True,
    filter={"source": {"$eq": "chunking.md"}},  # metadata filter
)

for match in response.matches:
    print(f"  score={match.score:.4f} | {match.metadata['text']}")
```

### 8.4 pgvector — Postgres Extension

Best for: teams already using Postgres who want to avoid another database.

```bash
pip install psycopg2-binary pgvector
# In Postgres: CREATE EXTENSION vector;
```

```python
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Setup ---
conn = psycopg2.connect("postgresql://user:pass@localhost:5432/ragdb")
register_vector(conn)
cur = conn.cursor()

# Create table with vector column
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        source VARCHAR(255),
        topic VARCHAR(100),
        embedding vector(1024)  -- match your model's dim
    );
""")

# Create HNSW index (pgvector >= 0.5.0)
cur.execute("""
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
""")
conn.commit()

# --- Ingest ---
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

docs = [
    ("RAG retrieves relevant context before generation.", "rag.md", "rag"),
    ("HNSW provides fast approximate nearest neighbor search.", "hnsw.md", "indexing"),
    ("Semantic search finds documents by meaning, not just keywords.", "search.md", "retrieval"),
]

for text, source, topic in docs:
    vec = model.encode(text, normalize_embeddings=True)
    cur.execute(
        "INSERT INTO documents (text, source, topic, embedding) VALUES (%s, %s, %s, %s)",
        (text, source, topic, vec.tolist()),
    )
conn.commit()

# --- Similarity Search with SQL ---
query_vec = model.encode("How does fast vector search work?", normalize_embeddings=True)

cur.execute("""
    SELECT text, source, 1 - (embedding <=> %s) AS similarity
    FROM documents
    WHERE topic = 'indexing'
    ORDER BY embedding <=> %s
    LIMIT 3;
""", (query_vec.tolist(), query_vec.tolist()))

for text, source, sim in cur.fetchall():
    print(f"  sim={sim:.4f} | ({source}) {text[:70]}")

# pgvector operators:
#   <=>  cosine distance  (use 1 - distance for similarity)
#   <->  L2 distance
#   <#>  negative dot product (use -1 * for similarity)
```

### 8.5 FAISS — Library-Level Control

Best for: maximum control, GPU acceleration, billion-scale offline pipelines.

```bash
pip install faiss-cpu  # or faiss-gpu for CUDA
```

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
DIM = 1024

# --- Build a corpus ---
corpus = [
    "Transformers use self-attention to model sequences.",
    "FAISS was developed by Meta AI for billion-scale search.",
    "Product quantization compresses vectors for memory efficiency.",
    "Cosine similarity measures the angle between two vectors.",
    "Chunking splits documents for more precise retrieval.",
]

embeddings = model.encode(corpus, normalize_embeddings=True).astype(np.float32)

# --- Index 1: Flat (exact, small datasets) ---
flat_index = faiss.IndexFlatIP(DIM)  # Inner product = cosine on normalized vecs
flat_index.add(embeddings)
print(f"Flat index size: {flat_index.ntotal}")

# --- Index 2: HNSW (fast, moderate memory) ---
hnsw_index = faiss.IndexHNSWFlat(DIM, 32)  # M=32
hnsw_index.hnsw.efConstruction = 200
hnsw_index.hnsw.efSearch = 64
hnsw_index.add(embeddings)

# --- Index 3: IVF-PQ (large datasets, compressed) ---
nlist = 4  # small dataset; use sqrt(N) for real data
quantizer = faiss.IndexFlatIP(DIM)
ivfpq_index = faiss.IndexIVFPQ(quantizer, DIM, nlist, 64, 8)  # 64 sub-vectors, 8-bit
ivfpq_index.train(embeddings)
ivfpq_index.add(embeddings)
ivfpq_index.nprobe = 2  # search 2 of 4 clusters

# --- Search ---
query = "How does memory compression work in vector databases?"
query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

for name, index in [("Flat", flat_index), ("HNSW", hnsw_index), ("IVF-PQ", ivfpq_index)]:
    D, I = index.search(query_vec, k=2)
    print(f"\n=== {name} ===")
    for score, idx in zip(D[0], I[0]):
        if idx >= 0:
            print(f"  score={score:.4f} | {corpus[idx]}")

# --- Persist and reload ---
faiss.write_index(hnsw_index, "hnsw.index")
loaded_index = faiss.read_index("hnsw.index")
```

---

## 9. Hybrid Search (Dense + Sparse)

Pure dense (semantic) search misses exact keyword matches. Hybrid search combines:

- **Dense:** Semantic embedding similarity (what things _mean_)
- **Sparse (BM25/BM42/SPLADE):** Keyword frequency weighting (exact _words_)

Combination via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))   where k ≈ 60
```

### Hybrid Search with Qdrant + BM25

```python
from qdrant_client.models import (
    SparseVector, SparseVectorParams,
    NamedVector, NamedSparseVector,
    Prefetch, FusionQuery, Fusion,
)
from rank_bm25 import BM25Okapi  # pip install rank_bm25
import numpy as np

# Note: Qdrant's native hybrid uses its own BM42 (neural sparse).
# This example uses external BM25 for the sparse component.

corpus_texts = [
    "RAG grounds LLM outputs in retrieved facts.",
    "HNSW enables fast graph-based ANN search.",
    "BM25 is a classic sparse retrieval algorithm.",
    "Hybrid search combines keyword and semantic retrieval.",
]

# Build BM25 index
tokenized_corpus = [t.lower().split() for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_sparse_vector(query: str) -> dict:
    """Returns {token_idx: score} for non-zero BM25 scores."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    non_zero = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
    if not non_zero:
        return {"indices": [], "values": []}
    indices, values = zip(*non_zero)
    return {"indices": list(indices), "values": list(values)}

def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = 60,
) -> list:
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for rank, (score, idx, text) in enumerate(dense_results):
        doc_id = str(idx)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    for rank, (score, idx, text) in enumerate(sparse_results):
        doc_id = str(idx)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(rrf_score, texts[doc_id]) for doc_id, rrf_score in ranked]


# Dense search
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
vecs = model.encode(corpus_texts, normalize_embeddings=True).astype(np.float32)
flat = faiss.IndexFlatIP(1024)
flat.add(vecs)

query = "BM25 keyword retrieval"
q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
D, I = flat.search(q_vec, 4)
dense_results = [(D[0][i], I[0][i], corpus_texts[I[0][i]]) for i in range(4)]

# Sparse BM25 search
bm25_scores = bm25.get_scores(query.lower().split())
sparse_ranked = sorted(
    [(bm25_scores[i], i, corpus_texts[i]) for i in range(len(corpus_texts))],
    reverse=True,
)

# Fuse
fused = reciprocal_rank_fusion(dense_results, sparse_ranked)
print("=== Hybrid RRF Results ===")
for score, text in fused:
    print(f"  rrf={score:.4f} | {text}")
```

---

## 10. Metadata Filtering

Metadata filters are critical for multi-tenant RAG, access control, time-bounded retrieval, and source attribution.

### Filter Design Principles

1. **Index your filter fields** — without a payload/column index, filters degrade to full scans.
2. **Pre-filter vs. post-filter:** Pre-filtering restricts the ANN search space (can hurt recall if filter is very selective). Post-filtering searches broadly then discards (can waste compute). Most modern DBs use intelligent planners.
3. **Use structured metadata:** Store dates as integers (Unix timestamps), use enums for category fields.

```python
# Qdrant: rich filter conditions
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range,
    MatchAny, IsEmptyCondition, IsNullCondition,
)

# Example: documents from 2024–2025, topic in [rag, retrieval]
my_filter = Filter(
    must=[
        FieldCondition(
            key="topic",
            match=MatchAny(any=["rag", "retrieval"]),
        ),
        FieldCondition(
            key="year",
            range=Range(gte=2024, lte=2025),
        ),
    ],
    must_not=[
        FieldCondition(key="draft", match=MatchValue(value=True)),
    ],
)

# Use in search:
# client.search(collection_name="...", query_vector=q_vec,
#               query_filter=my_filter, limit=5)
```

```python
# pgvector: full SQL power
cur.execute("""
    SELECT text, source, 1 - (embedding <=> %s) AS sim
    FROM documents
    WHERE topic = ANY(%s)
      AND created_at >= NOW() - INTERVAL '1 year'
      AND is_draft = FALSE
    ORDER BY embedding <=> %s
    LIMIT 5;
""", (q_vec.tolist(), ["rag", "retrieval"], q_vec.tolist()))
```

---

## 11. Reranking

After ANN retrieval, a **reranker** (cross-encoder) scores each candidate pair `(query, chunk)` jointly — much more accurate than bi-encoder similarity but too slow to run over the entire corpus.

**Pipeline:** ANN retrieval (top-50) → reranker → return top-5 to LLM.

```python
# pip install sentence-transformers cohere
from sentence_transformers import CrossEncoder

# Cross-encoder reranker (local)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How does HNSW achieve fast search?"
candidates = [
    "HNSW builds hierarchical graphs for approximate search.",
    "Product quantization compresses vectors for efficiency.",
    "HNSW uses multiple layers — top layers for coarse search, bottom for fine.",
    "Reranking improves precision after initial retrieval.",
]

# Score all (query, candidate) pairs
pairs = [(query, c) for c in candidates]
scores = reranker.predict(pairs)

# Sort by score
ranked = sorted(zip(scores, candidates), reverse=True)

print("=== Reranked Results ===")
for score, text in ranked:
    print(f"  {score:.4f} | {text[:70]}")
```

```python
# Cohere Rerank API (cloud, production-grade)
import cohere

co = cohere.Client(os.environ["COHERE_API_KEY"])

results = co.rerank(
    model="rerank-english-v3.0",
    query="How does HNSW achieve fast search?",
    documents=candidates,
    top_n=3,
)

for r in results.results:
    print(f"  score={r.relevance_score:.4f} | {candidates[r.index][:70]}")
```

---

## 12. Advanced RAG Patterns

### Multi-Query Retrieval

Generate multiple query variations to improve recall:

```python
from openai import OpenAI

client = OpenAI()

def expand_query(query: str, n: int = 3) -> list[str]:
    """Use an LLM to generate query variations."""
    prompt = f"""Generate {n} different ways to ask the following question.
Return one question per line, no numbering.
Question: {query}"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return [q.strip() for q in resp.choices[0].message.content.strip().split("\n") if q.strip()]

queries = expand_query("How do vector databases store embeddings?")
# Embed each, retrieve, deduplicate results
```

### Contextual Retrieval (Anthropic)

Before embedding each chunk, prepend a context summary to make chunks self-contained:

```python
import anthropic

def contextualize_chunk(document: str, chunk: str) -> str:
    """Prepend document-level context to a chunk before embedding."""
    client = anthropic.Anthropic()
    
    prompt = f"""<document>
{document[:3000]}  # Use first 3000 chars for context
</document>

<chunk>
{chunk}
</chunk>

Write a very brief (1-2 sentence) context that situates this chunk
within the document. This will be prepended to the chunk for retrieval purposes.
Respond with only the context, no preamble."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    context = msg.content[0].text.strip()
    return f"{context}\n\n{chunk}"

# contextualized = contextualize_chunk(full_document, chunk_text)
# Then embed `contextualized` instead of raw `chunk_text`
```

### Parent-Document Retrieval (Small-to-Large)

Store small chunks for precise retrieval, but return larger parent chunks for richer LLM context:

```python
from dataclasses import dataclass, field
from uuid import uuid4

@dataclass
class ParentChunk:
    id: str
    text: str
    children: list[str] = field(default_factory=list)

@dataclass  
class ChildChunk:
    id: str
    text: str
    parent_id: str

def build_parent_child_chunks(
    document: str,
    parent_size: int = 1024,
    child_size: int = 256,
    overlap: int = 32,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Split into large parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
    parent_texts = parent_splitter.split_text(document)

    parents, children = [], []
    for p_text in parent_texts:
        parent = ParentChunk(id=str(uuid4()), text=p_text)
        
        # Split each parent into small children
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=overlap
        )
        child_texts = child_splitter.split_text(p_text)
        
        for c_text in child_texts:
            child = ChildChunk(id=str(uuid4()), text=c_text, parent_id=parent.id)
            parent.children.append(child.id)
            children.append(child)
        
        parents.append(parent)

    return parents, children

# Embed and store child chunks in vector DB (with parent_id in metadata)
# At query time: retrieve child chunks, look up their parents, return parent text to LLM
```

---

## 13. Production Considerations

### Performance Tuning

|Area|Action|
|---|---|
|**Index parameters**|Profile `ef_search`/`nprobe` vs. recall@k using your actual queries|
|**Batch embedding**|Embed 32–256 documents per API call; avoid one-by-one|
|**Embedding cache**|Cache embeddings for repeated or near-duplicate queries|
|**Async clients**|Use `AsyncQdrantClient` / async OpenAI for concurrent retrieval|
|**Quantization**|Enable scalar quantization in Qdrant to halve memory with minimal recall loss|

### Monitoring & Evaluation

```python
# Key metrics to track:
# - Recall@k: fraction of relevant docs in top-k results
# - MRR (Mean Reciprocal Rank): how high is the first relevant result?
# - Latency P50/P95/P99: ANN query time
# - Answer faithfulness: do LLM answers match the retrieved context?
# - Context precision: how much of retrieved context is actually used?

# Tools: RAGAS (pip install ragas), TruLens, LangSmith
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
```

### Security Checklist

- **Tenant isolation:** Use separate Qdrant collections or Pinecone namespaces per tenant.
- **PII in embeddings:** Scrub or hash PII fields before embedding — vectors cannot be easily reversed, but metadata can be exposed.
- **Audit logs:** Log every retrieval call (query vector hash, filter, n_results, latency).
- **Access tokens:** Qdrant supports API key authentication; use it even internally.

### Data Freshness

```python
# Pattern: store ingestion timestamp, query with recency filter
import time

metadata = {
    "text": chunk_text,
    "source": source_path,
    "ingested_at": int(time.time()),  # Unix timestamp
    "doc_date": "2025-03",
}

# Query: only retrieve docs ingested in the last 30 days
from qdrant_client.models import Range, FieldCondition, Filter

recency_filter = Filter(
    must=[
        FieldCondition(
            key="ingested_at",
            range=Range(gte=int(time.time()) - 30 * 86400),
        )
    ]
)
```

---

## 14. Decision Framework

Work through these questions in order:

```
1. SCALE
   How many vectors? (< 1M / 1M–100M / > 100M)
   Growth rate? (static / slow / fast)

2. DEPLOYMENT CONSTRAINTS  
   Can you run docker/k8s? → OSS is fine
   No infra team? → Managed (Pinecone, Qdrant Cloud, Zilliz)
   Air-gapped / on-prem required? → Milvus or Qdrant self-hosted
   Already on Postgres? → Try pgvector first

3. RETRIEVAL REQUIREMENTS
   Semantic only? → Any DB with HNSW
   Semantic + keyword? → Qdrant (BM42), Weaviate, Milvus, Pinecone
   Full SQL + vector? → pgvector, SingleStore

4. OPERATIONAL MATURITY
   Prototyping? → ChromaDB
   Small team, production? → Qdrant (best OSS DX)
   Large team, needs SLA? → Pinecone or Weaviate Cloud

5. COST MODEL
   Free tier sufficient? → Pinecone Starter, Qdrant free cluster
   Optimize for compute? → Self-hosted Qdrant or Milvus
   Optimize for storage? → IVF-PQ compression, Qdrant scalar quantization
```

---

## 15. Key Libraries & References

### Core Libraries

```bash
# Vector DBs
pip install chromadb                    # local prototype
pip install qdrant-client               # production OSS
pip install pymilvus                    # massive scale OSS
pip install pinecone                    # managed cloud
pip install weaviate-client             # OSS + cloud
pip install pgvector psycopg2-binary    # postgres extension
pip install faiss-cpu                   # library (or faiss-gpu)

# Embeddings
pip install sentence-transformers       # local OSS models
pip install openai                      # OpenAI embeddings
pip install voyageai                    # Voyage AI embeddings

# RAG Frameworks
pip install langchain langchain-community langchain-text-splitters
pip install llama-index-core llama-index-vector-stores-qdrant
pip install haystack-ai                 # production pipelines

# Chunking
pip install tiktoken                    # token counting
pip install chonkie                     # fast chunking library

# Reranking
pip install cohere                      # Cohere Rerank API
# cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers

# Evaluation
pip install ragas                       # RAG evaluation metrics
pip install trulens-eval                # LLM app observability
```

### Authoritative References

- MTEB Leaderboard — benchmark for choosing embedding models
- FAISS Index Selection Guide — Meta's official guidance
- Qdrant Documentation — best-in-class hybrid search docs
- ANN Benchmarks — independent recall/throughput benchmarks
- Vectara NAACL 2025 Chunking Study — peer-reviewed chunking evaluation
- Chroma Chunking Evaluation — recall benchmarks across strategies
- RAGAs — evaluation metrics (faithfulness, context precision, answer relevancy)

---

_Report generated April 2026. Libraries and benchmarks evolve rapidly — verify versions before production deployment._