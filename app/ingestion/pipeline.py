# app/ingestion/pipeline.py
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import BaseVectorStore


class IngestionPipeline:
    """
    Orchestrates the full document → vector DB pipeline.
    
    FLOW:
    Source → Load → Chunk → Embed → Store
    
    This runs ONCE (or when docs change).
    The retrieval pipeline runs on EVERY user query.
    """

    def __init__(self,
                 vector_store: BaseVectorStore,
                 chunk_size: int = 512,
                 chunk_overlap: int = 64):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingModel()
        self.vector_store = vector_store

    def ingest(self, sources: list[dict],
               strategy: str = "auto",
               namespace: str = "default") -> dict:
        """
        Full ingestion: load → chunk → embed → store.
        Returns summary stats.
        """
        print("\n📥 INGESTION PIPELINE STARTING")
        print("=" * 50)

        # Step 1: Load documents
        print("\n[1/4] Loading documents...")
        documents = self.loader.load_many(sources)
        print(f"✅ Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        print("\n[2/4] Chunking documents...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc, strategy=strategy)
            all_chunks.extend(chunks)
            print(f"  {doc.metadata.get('filename') or doc.source[:50]}"
                  f" → {len(chunks)} chunks")
        print(f"✅ Total chunks: {len(all_chunks)}")

        # Step 3: Generate embeddings
        print("\n[3/4] Generating embeddings...")
        vectors = self.embedder.embed_chunks(all_chunks)
        print(f"✅ Embedded {len(vectors)} chunks")

        # Step 4: Store in vector DB
        print("\n[4/4] Storing in vector database...")
        self.vector_store.upsert(vectors, namespace=namespace)

        stats = self.vector_store.get_stats()
        print(f"\n✅ INGESTION COMPLETE")
        print(f"   Documents: {len(documents)}")
        print(f"   Chunks:    {len(all_chunks)}")
        print(f"   DB Stats:  {stats}")

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "vectors": len(vectors),
            "stats": stats
        }