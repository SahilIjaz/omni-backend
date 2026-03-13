# test_hybrid.py
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import get_vector_store
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker

# ── Setup ────────────────────────────────────────────
embedder = EmbeddingModel()
vector_store = get_vector_store(embedder, store_type="chroma")
bm25 = BM25Retriever()
reranker = Reranker()

# ── Ingest sample docs ───────────────────────────────
print("=" * 60)
print("STEP 1: Ingesting documents")
print("=" * 60)

loader = DocumentLoader()
chunker = TextChunker(chunk_size=400, chunk_overlap=50)

sources = [{"type": "web",
            "url": "https://docs.anthropic.com/en/docs/about-claude/models/overview"}]

docs = loader.load_many(sources)
all_chunks = []
for doc in docs:
    chunks = chunker.chunk(doc)
    all_chunks.extend(chunks)

# Build BM25 index from same chunks
bm25.build_index(all_chunks)
bm25.save()

# Embed + store in vector DB
from app.retrieval.embedder import EmbeddingModel as EM
em = EmbeddingModel()
vectors = em.embed_chunks(all_chunks)
vector_store.upsert(vectors, namespace="test")

# ── Test hybrid retrieval ────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Hybrid Retrieval Comparison")
print("=" * 60)

hybrid = HybridRetriever(embedder, vector_store, bm25, reranker)

test_queries = [
    "What is the latest Claude model available?",
    "claude-sonnet token limit context window",
    "Which model should I use for complex reasoning tasks?",
]

for query in test_queries:
    print(f"\n{'─'*60}")
    print(f"QUERY: {query}")
    print(f"{'─'*60}")

    results = hybrid.retrieve(
        query,
        top_k=3,
        use_reranker=True,
        namespace="test"
    )

    print(f"\nTOP RESULTS:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Type: {r.get('retrieval_type')} | "
              f"Rerank: {r.get('rerank_score', 'N/A')}")
        print(f"     {r['content'][:200]}...")