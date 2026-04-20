# Document chunking strategies

Here's a thorough breakdown of document chunking for RAG — one of the most impactful engineering decisions you'll make in any RAG system.

---

## What is document chunking and why does it matter?

When you build a RAG pipeline, you can't stuff entire documents into a vector store and retrieve them wholesale. You need to break them into smaller pieces — **chunks** — that can be embedded, indexed, and retrieved individually. The quality of your chunking strategy directly determines retrieval quality, and retrieval quality is the ceiling for your LLM's answer quality.

Poor chunking leads to two failure modes: chunks that are too large (irrelevant noise gets retrieved alongside relevant content) and chunks that are too small (useful context gets split across multiple chunks, none of which are retrieved together).

Let's look at the main strategies you'll encounter.

---

## The strategies in depth

### 1. Fixed-size chunking

The simplest approach: split every N tokens (or characters), regardless of content.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # characters of overlap
    separator="\n"
)
chunks = splitter.split_text(document_text)
```

**When to use it:** Prototyping, homogeneous documents (logs, data dumps), when you need predictable chunk sizes for cost estimation.

**Watch out for:** Sentences and ideas getting cut mid-thought. A chunk ending with "The treatment was effective because" and the next starting mid-explanation is a retrieval disaster.

---

### 2. Sentence-based chunking

Split on sentence boundaries (periods, question marks), then group N sentences per chunk. Preserves grammatical units.

```python
import nltk
from langchain.text_splitter import NLTKTextSplitter

splitter = NLTKTextSplitter(chunk_size=300)
chunks = splitter.split_text(document_text)
```

**When to use it:** News articles, legal text, conversational documents — anything where individual sentences carry meaning.

---

### 3. Recursive character splitting

The practical default for most production RAG systems. You provide a priority list of separators and the splitter tries them in order — first double newlines, then single newlines, then spaces, then characters — only going finer if the chunk is still too large.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document_text)
```

This is the **go-to starting point** for most engineers. It respects document structure when possible, falls back gracefully, and the parameters are easy to tune.

---

### 4. Structure-aware (markdown - HTML) chunking

For structured documents, split on logical units — headings, sections, HTML tags — rather than raw characters.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
chunks = splitter.split_text(markdown_document)
# Each chunk carries metadata: {"h1": "Introduction", "h2": "Setup"}
```

The metadata is valuable — you can filter by section during retrieval, or prepend section context to each chunk automatically.

---

### 5. Semantic chunking

Instead of splitting on characters or structure, you split on _meaning shifts_. You embed consecutive sentences, compute cosine similarity between adjacent embeddings, and cut where similarity drops sharply (a topic shift).

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95
)
chunks = splitter.split_text(document_text)
```

**When to use it:** Long, heterogeneous documents where topics shift within a section — research papers, transcripts, long-form content.

**Watch out for:** Cost (every sentence gets embedded during chunking) and non-determinism (the threshold is a hyperparameter you'll need to tune per corpus).

---

### 6. Hierarchical / parent-child chunking

This is a more architectural pattern than a pure chunking strategy. You store _small_ child chunks for high-precision retrieval, but you return the _parent_ (larger) chunk to the LLM for richer context. The index contains children; the context window gets parents.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent: large chunks the LLM will read
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# Child: small chunks the retriever will match against
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(embedding_function=embeddings)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
```

This handles a common failure mode: small chunks retrieve precisely but the LLM lacks context; large chunks provide context but embed poorly. Parent-child gives you both.

---

### 7. Agentic - contextual chunking

The highest-quality and most expensive approach: use an LLM to decide boundaries or to augment each chunk with context before indexing.

Anthropic's contextual retrieval technique involves prepending a chunk-specific summary to each chunk before embedding, dramatically reducing the "lost context" problem:

```python
import anthropic

client = anthropic.Anthropic()

def add_context_to_chunk(document: str, chunk: str) -> str:
    prompt = f"""<document>
{document}
</document>

Here is a chunk from this document:
<chunk>
{chunk}
</chunk>

Write a short sentence (1-2 sentences) situating this chunk within the document.
Only output the context sentence, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    context = response.content[0].text
    return f"{context}\n\n{chunk}"

# Apply to all chunks before embedding
contextualized_chunks = [add_context_to_chunk(full_doc, c) for c in raw_chunks]
```

This can meaningfully reduce retrieval failure rates, especially for chunks that contain references like "As mentioned above..." or "This approach..." without the referent.

---

## The overlap parameter

Regardless of which strategy you use, **overlap** is almost always worth adding. A sliding window ensures that content near a chunk boundary appears in two adjacent chunks, so retrieval doesn't miss a passage just because it straddles a cut.

A good rule of thumb: overlap = 10–20% of chunk size. For a 1000-token chunk, use 100–200 tokens of overlap. More overlap means more storage and more retrieved noise; less means more risk of boundary misses.

---

## Choosing a strategy: practical decision guide

|Situation|Recommended approach|
|---|---|
|Prototyping / unknown corpus|Recursive character splitting, `chunk_size=1000`, `overlap=200`|
|Markdown docs, structured knowledge bases|`MarkdownHeaderTextSplitter` + recursive fallback|
|Long heterogeneous documents (papers, transcripts)|Semantic chunking|
|High-precision Q&A over dense documents|Parent-child retrieval|
|Maximum quality, budget available|Contextual/agentic chunking with parent-child|
|Code files|Language-specific splitters (LangChain has these: `PythonCodeTextSplitter`, etc.)|

---

## What to measure

Chunking is an empirical problem. Always evaluate with your actual queries:

```python
# Minimal retrieval eval loop
from sklearn.metrics.pairwise import cosine_similarity

def retrieval_hit_rate(retriever, eval_pairs):
    """eval_pairs: list of (query, expected_text_fragment)"""
    hits = 0
    for query, expected in eval_pairs:
        results = retriever.get_relevant_documents(query)
        retrieved_text = " ".join([r.page_content for r in results])
        if expected.lower() in retrieved_text.lower():
            hits += 1
    return hits / len(eval_pairs)
```

The key metrics to track are **recall** (did the right chunks get retrieved?) and **precision** (were the retrieved chunks relevant?). Tools like `ragas` and `LangSmith` can automate this at scale.

---

The practical path: start with recursive character splitting, add overlap, and evaluate retrieval quality on a sample of real queries. Only move to semantic or agentic chunking if your eval metrics show you need them — they add meaningful complexity and cost.

## Parent - Child Chunking

Every other chunking strategy forces a single tradeoff between retrieval precision and context richness. Small chunks embed well and retrieve precisely but lack context. Large chunks provide rich context but embed poorly — the vector averages over too many topics and matches nothing specifically.

Parent-child chunking resolves this by maintaining **two separate representations of the same content**: small child chunks for retrieval, large parent chunks for generation. The vector index only contains children. When a child is retrieved, the system fetches its parent and sends _that_ to the LLM instead.