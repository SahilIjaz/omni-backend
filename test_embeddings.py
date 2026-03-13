# test_embeddings.py
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import get_vector_store
from app.ingestion.pipeline import IngestionPipeline

# ── Test 1: Embedding similarity ──────────────────────
print("=" * 60)
print("TEST 1: Semantic Similarity")
print("=" * 60)

embedder = EmbeddingModel()

queries = [
    "How do I reset my password?",
    "Steps to change my login credentials",
    "What is photosynthesis?",
]

vectors = [embedder.embed_text(q) for q in queries]

def cosine_similarity(a, b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b)

print(f"\n'{queries[0]}' vs '{queries[1]}'")
print(f"Similarity: {cosine_similarity(vectors[0], vectors[1]):.4f} (should be HIGH)")

print(f"\n'{queries[0]}' vs '{queries[2]}'")
print(f"Similarity: {cosine_similarity(vectors[0], vectors[2]):.4f} (should be LOW)")

# ── Test 2: Full ingestion + search pipeline ──────────
print("\n" + "=" * 60)
print("TEST 2: Ingest → Search Round Trip")
print("=" * 60)

vector_store = get_vector_store(embedder, store_type="chroma")
pipeline = IngestionPipeline(vector_store)

# Ingest a webpage
pipeline.ingest(
    sources=[{"type": "web",
              "url": "https://docs.anthropic.com/en/docs/about-claude/models/overview"}],
    namespace="test"
)

# Search it
print("\n🔍 Searching: 'What is the latest Claude model?'")
query_vec = embedder.embed_text("What is the latest Claude model?")
results = vector_store.search(query_vec, top_k=3, namespace="test")

for i, r in enumerate(results):
    print(f"\n[Result {i+1}] Score: {r['score']}")
    print(f"Source: {r['source']}")
    print(f"Content: {r['content'][:200]}...")