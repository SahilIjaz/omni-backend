# test_pipeline.py
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import get_vector_store
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.retrieval.embedder import EmbeddingModel
from app.rag.pipeline import RAGPipeline

# ── Initialize components (only once) ───────────────
print("🚀 Initializing RAG System...")
embedder    = EmbeddingModel()
vector_store = get_vector_store(embedder, store_type="chroma")
bm25        = BM25Retriever()
reranker    = Reranker()

# ── Load + chunk documents ───────────────────────────
print("\n📄 Loading and chunking documents...")
loader  = DocumentLoader()
chunker = TextChunker(chunk_size=300, chunk_overlap=50)

sources = [
    {"type": "text", "path": "docs/sample.txt"},
]

docs = loader.load_many(sources)
all_chunks = []
for doc in docs:
    chunks = chunker.chunk(doc, strategy="recursive")
    all_chunks.extend(chunks)
    print(f"  {doc.metadata['filename']} → {len(chunks)} chunks")

print(f"✅ Total chunks: {len(all_chunks)}")

# Print chunks so you can see what they look like
print("\n--- Sample Chunks ---")
for i, c in enumerate(all_chunks[:3]):
    print(f"\n[Chunk {i}] ({len(c)} chars)")
    print(c.content)

# ── Embed + store in vector DB ───────────────────────
print("\n🔢 Embedding and storing...")
vectors = embedder.embed_chunks(all_chunks)
vector_store.upsert(vectors, namespace="demo")

# ── Build BM25 index ─────────────────────────────────
print("\n🔧 Building BM25 index...")
bm25.build_index(all_chunks)
bm25.save()

# ── Build retriever + RAG pipeline ───────────────────
retriever = HybridRetriever(embedder, vector_store, bm25, reranker)
rag = RAGPipeline(retriever)

# ── TEST 1: Standard Query ───────────────────────────
print("\n" + "="*60)
print("TEST 1: Standard RAG Query")
print("="*60)

result = rag.query(
    "What Claude models are available and what are they best for?",
    top_k=4,
    namespace="demo"
)

print(f"\n📝 ANSWER:\n{result['answer']}")
print(f"\n📚 Sources:  {result['sources']}")
print(f"🔢 Chunks:   {result['chunks_used']}")
print(f"🔍 Method:   {result['retrieval_method']}")

# ── TEST 2: Multi-turn conversation ──────────────────
print("\n" + "="*60)
print("TEST 2: Multi-Turn Conversation")
print("="*60)

turns = [
    "What is RAG and why is it useful?",
    "What vector databases does it use?",       # relies on chat history
    "Which one is better for production?",      # relies on previous answer
]

for question in turns:
    print(f"\n👤 You: {question}")
    result = rag.query(question, namespace="demo")
    print(f"🤖 Claude: {result['answer'][:400]}...")

# ── TEST 3: Streaming ────────────────────────────────
print("\n" + "="*60)
print("TEST 3: Streaming Response")
print("="*60)

rag.clear_history()
question = "Explain the difference between Opus, Sonnet and Haiku"
print(f"👤 You: {question}")
print("🤖 Claude: ", end="", flush=True)

for event in rag.query_stream(question, namespace="demo"):
    if event["type"] == "metadata":
        print(f"\n[📚 Sources: {event['sources']} | "
              f"Chunks: {event['chunks_used']} | "
              f"Method: {event['retrieval_method']}]")
        print("Streaming → ", end="", flush=True)
    elif event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "done":
        print("\n\n✅ Stream complete!")

# ── TEST 4: Keyword-specific query ──────────────────
print("\n" + "="*60)
print("TEST 4: Exact Keyword Query (BM25 strength)")
print("="*60)

result = rag.query(
    "claude-sonnet-4-6 model identifier",   # exact keyword — BM25 shines here
    namespace="demo"
)
print(f"📝 Answer: {result['answer']}")
print(f"🔍 Method: {result['retrieval_method']}")