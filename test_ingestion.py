# test_ingestion.py
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker

loader = DocumentLoader()
chunker = TextChunker(chunk_size=300, chunk_overlap=50)

# ── Test 1: Load a webpage ──────────────────────────
print("=" * 60)
print("TEST 1: Web Loading + Recursive Chunking")
print("=" * 60)

doc = loader.load_web("https://docs.anthropic.com/en/docs/about-claude/models/overview")
print(f"📄 Loaded: {doc.metadata['title']}")
print(f"📊 Total chars: {len(doc.content)}")

chunks = chunker.chunk(doc, strategy="recursive")
print(f"🔪 Chunks created: {len(chunks)}")
print(f"\n--- First 3 chunks ---")
for c in chunks[:3]:
    print(f"\n[Chunk {c.chunk_index}] ({len(c)} chars)")
    print(c.content[:200] + "...")

# ── Test 2: Compare strategies on same text ─────────
print("\n" + "=" * 60)
print("TEST 2: Comparing Chunking Strategies")
print("=" * 60)

sample_text = """
Introduction to RAG Systems

RAG stands for Retrieval Augmented Generation. It combines 
the power of large language models with external knowledge retrieval.

How Retrieval Works

When a user asks a question, the system first converts it into 
an embedding vector. This vector is then used to search a vector 
database for semantically similar content.

The Generation Step

Retrieved chunks are injected into the LLM prompt as context.
The model then generates a grounded, factual response based 
only on the provided context, reducing hallucination significantly.
"""

from app.ingestion.document_loader import Document

test_doc = Document(
    content=sample_text,
    source="test",
    doc_type="text",
    metadata={}
)

for strategy in ["fixed", "recursive", "document_aware"]:
    chunks = chunker.chunk(test_doc, strategy=strategy)
    print(f"\n📌 {strategy.upper()}: {len(chunks)} chunks")
    for c in chunks:
        print(f"   [{c.chunk_index}] {len(c)} chars: {c.content[:80]}...")