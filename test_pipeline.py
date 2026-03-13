# test_pipeline.py
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import get_vector_store
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.ingestion.pipeline import IngestionPipeline
from app.rag.pipeline import RAGPipeline

# ── Initialize all components ────────────────────────
print("🚀 Initializing RAG System...")
embedder = EmbeddingModel()
vector_store = get_vector_store(embedder, store_type="chroma")
bm25 = BM25Retriever()
reranker = Reranker()

# ── Ingest documents ─────────────────────────────────
ingest = IngestionPipeline(vector_store)
result = ingest.ingest(
    sources=[
        {"type": "web",
         "url": "https://docs.anthropic.com/en/docs/about-claude/models/overview"},
    ],
    namespace="demo"
)

# Rebuild BM25 from same chunks
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker

loader = DocumentLoader()
chunker = TextChunker()
docs = loader.load_many([
    {"type": "web",
     "url": "https://docs.anthropic.com/en/docs/about-claude/models/overview"}
])
all_chunks = []
for doc in docs:
    all_chunks.extend(chunker.chunk(doc))
bm25.build_index(all_chunks)

# ── Build hybrid retriever + RAG pipeline ────────────
retriever = HybridRetriever(embedder, vector_store, bm25, reranker)
rag = RAGPipeline(retriever)

# ── Test 1: Standard query ───────────────────────────
print("\n" + "="*60)
print("TEST 1: Standard Query")
print("="*60)

result = rag.query(
    "What Claude models are currently available?",
    top_k=5,
    namespace="demo"
)

print(f"\n📝 ANSWER:\n{result['answer']}")
print(f"\n📚 Sources: {result['sources']}")
print(f"🔢 Chunks used: {result['chunks_used']}")
print(f"🔍 Method: {result['retrieval_method']}")

# ── Test 2: Multi-turn conversation ──────────────────
print("\n" + "="*60)
print("TEST 2: Multi-Turn Conversation")
print("="*60)

questions = [
    "What is Claude Sonnet best used for?",
    "How does it compare to Opus?",       # Tests chat history
    "Which one is more cost effective?",  # Tests continued context
]

for q in questions:
    print(f"\n👤 User: {q}")
    result = rag.query(q, namespace="demo")
    print(f"🤖 Claude: {result['answer'][:300]}...")

# ── Test 3: Streaming ────────────────────────────────
print("\n" + "="*60)
print("TEST 3: Streaming Response")
print("="*60)

rag.clear_history()
print("👤 User: What are the key differences between Claude models?")
print("🤖 Claude: ", end="", flush=True)

for event in rag.query_stream(
    "What are the key differences between Claude models?",
    namespace="demo"
):
    if event["type"] == "metadata":
        print(f"\n[Sources: {event['sources']}]")
        print("Streaming answer: ", end="", flush=True)
    elif event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "done":
        print("\n✅ Stream complete")