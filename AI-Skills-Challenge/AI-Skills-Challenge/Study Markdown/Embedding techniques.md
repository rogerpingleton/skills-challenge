
# Embedding techniques

## What Is an Embedding?

An embedding is a mathematical (vector) representation of an object like text. When a neural network is trained, it creates many representations that have connections to other objects in the network — capturing semantic meaning so you can compare objects mathematically.

The classic illustration of this:

```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```

In a RAG pipeline, both your **documents** and the **user's query** get embedded using the same model. Retrieval is then just a nearest-neighbor search in that vector space.

---

## Why Embeddings Are the Core of RAG

Unlike traditional keyword-based retrieval, dense retrieval uses vector embeddings to represent both queries and documents in a shared semantic space. This allows the system to identify relevant information even when the query and source use different terminology — a game-changer in fields like legal research, where precise language varies across jurisdictions.

The embedding model converts both the query and stored data into numerical representations (vectors), making them far quicker and easier to match.

---

## The Two Phases You'll Work In

**Ingestion (offline):** Chunk documents → embed chunks → store in a vector DB.

**Query (online):** Embed the user query → similarity search → retrieve top-k chunks → pass to LLM.

```python
# Ingestion
from openai import OpenAI
client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# Storing (pseudocode with any vector DB)
for chunk in document_chunks:
    vector = embed(chunk.text)
    vector_db.upsert(id=chunk.id, vector=vector, metadata={"text": chunk.text})

# Query time
query_vector = embed("What is the refund policy?")
results = vector_db.query(vector=query_vector, top_k=5)
context = "\n".join([r.metadata["text"] for r in results])
```

---

## Choosing an Embedding Model

The embedding model decision cascades through the entire system — changing models later requires re-embedding the complete document corpus, a process that costs time, compute, and potentially service disruption. Choose carefully upfront.

The current landscape (as of early 2026):

**Commercial:**

- **Voyage AI** (`voyage-3-large`) leads MTEB benchmarks, outperforming OpenAI's `text-embedding-3-large` by ~10% and supporting 32K-token context windows. Its 1024-dimensional embeddings cost $0.06/million tokens — about 2.2x cheaper than OpenAI.
- **Cohere embed-v4** achieved the highest MTEB score (65.2) as of November 2025, optimized for search/retrieval with strong multilingual support across 100+ languages.
- **OpenAI `text-embedding-3-large`** is the most battle-tested for production.

**Open-source (self-hosted):**

- **BGE, E5, and GTE** models enable self-hosted embedding at scale. Organizations processing billions of documents often deploy these on internal GPU infrastructure to eliminate per-token costs.

**Rule of thumb:** Start with a commercial API. Switch to self-hosted only when you're processing at very high volume.

---

## Similarity Metrics

Once you have vectors, you need a distance function:

|Metric|Formula|Best For|
|---|---|---|
|**Cosine similarity**|`cos(θ) = A·B / (‖A‖ ‖B‖)`|Most text/RAG use cases|
|**Dot product**|`A·B`|When vectors are normalized|
|**Euclidean (L2)**|`‖A - B‖`|Some image/multimodal cases|

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Most vector databases (Pinecone, Weaviate, Milvus, Qdrant) handle this for you — just declare the metric at index creation time.

---

## Hybrid Retrieval: Dense + Sparse

Traditional semantic search alone is no longer enough. Leading enterprise implementations now use hybrid retrieval, combining semantic and keyword-based approaches, which consistently outperforms single-method pipelines for accuracy, especially in noisy datasets.

Sparse retrieval (BM25/keyword) excels at exact matches. Dense embeddings excel at semantic similarity. Combining both covers your bases:

```python
# Using LangChain's EnsembleRetriever pattern
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma

# Dense retriever
dense_retriever = Chroma(...).as_retriever(search_kwargs={"k": 5})

# Sparse retriever
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 5

# Ensemble (weights: 60% dense, 40% sparse)
hybrid = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4]
)
```

---

## Chunking Strategy Matters

Semantic chunking improves recall up to 9% over fixed-size approaches. You're not just picking a model — you're deciding _what_ gets embedded.

Common strategies:

```python
# Fixed-size chunking (simple, fast)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Semantic chunking (better, slower)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(OpenAIEmbeddings())
chunks = semantic_splitter.split_documents(docs)
```

Overlap between chunks ensures context isn't lost at boundaries.

---

## Fine-Tuning Embeddings for Your Domain

Domain-adaptive pretraining fine-tunes retrieval embeddings on domain-specific corpora such as legal or scientific texts, enhancing precision in specialized fields.

Fine-tuning adjusts the model's weights to better represent your vocabulary and semantics. It can improve retrieval accuracy, especially for specialized domains like code search or legal documents — but it requires careful evaluation and can sometimes degrade performance if the training data is poor.

Use the `sentence-transformers` library for this in Python:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Training pairs: (query, relevant_doc)
train_examples = [
    InputExample(texts=["What is our return policy?", "Returns accepted within 30 days..."]),
    # ...
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
model.save("my-domain-embedding-model")
```

---

## Dimensionality Reduction

Embedding vectors can be high-dimensional, increasing storage and compute costs. Dimensionality reduction techniques like PCA or t-SNE help make embeddings more manageable. These tools are available in PyTorch and scikit-learn, and can improve semantic clarity and eliminate noisy features.

```python
from sklearn.decomposition import PCA
import numpy as np

# Reduce from 1536-dim (OpenAI) to 256-dim
pca = PCA(n_components=256)
reduced_vectors = pca.fit_transform(np.array(all_embeddings))
```

Use this when storage cost or query latency is a concern at scale.

---

## Multimodal Embeddings

Multimodal RAG integrates image, audio, tabular, and video embeddings to create more holistic reasoning. Models like CLIP, combined with speech-to-text and frame analysis, extend RAG capabilities to video — allowing queries like "Find scenes with emotional conflict."

For image+text use cases, CLIP is the standard starting point:

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("diagram.png")
inputs = processor(images=image, return_tensors="pt")
image_embedding = model.get_image_features(**inputs)
```

---

## Evaluating Your Embeddings

Retrieval performance is the most practical way to evaluate embedding quality. Use real-world queries and content to test your models. The Hugging Face MTEB leaderboard provides up-to-date rankings.

Key metrics to track:

- **Recall@k** — did the right document appear in the top k results?
- **MRR (Mean Reciprocal Rank)** — how high does the correct doc rank on average?
- **Latency** — embedding + search round-trip time (production target: sub-50ms)

---

## Quick Reference: Production Checklist

|Decision|Recommendation|
|---|---|
|Starting model|`text-embedding-3-large` or `voyage-3-large`|
|Chunking|Semantic chunking; 256–512 tokens with ~10% overlap|
|Retrieval|Hybrid (dense + BM25)|
|Vector DB|Pinecone / Weaviate (managed) or Milvus / Qdrant (self-hosted)|
|Similarity metric|Cosine similarity|
|Evaluation|MTEB + your own domain query set|
|Fine-tuning|Only when general models miss domain-specific terminology|

The single most important principle: **your RAG system is only as good as what it can retrieve**, and retrieval quality is 80% a function of your embedding model + chunking strategy.