# app/retrieval/vector_store.py
from abc import ABC, abstractmethod
from app.retrieval.embedder import EmbeddingModel
from app.ingestion.chunker import Chunk
from app.core.config import settings


# ─────────────────────────────────────────────
# Abstract base — lets us swap vector DBs easily
# ─────────────────────────────────────────────
class BaseVectorStore(ABC):
    """
    WHY AN ABSTRACT CLASS?
    We want to switch between Pinecone (prod) and 
    Chroma (local dev) without changing any RAG pipeline code.
    This is the Strategy design pattern.
    """

    @abstractmethod
    def upsert(self, vectors: list[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: list[float],
               top_k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        pass


# ─────────────────────────────────────────────
# Pinecone — Production vector store
# ─────────────────────────────────────────────
class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone is a managed vector database.
    
    KEY CONCEPTS:
    - Index:     Like a database table, holds all your vectors
    - Namespace: Like a schema — partition vectors by source/topic
    - Upsert:    Insert or update (idempotent — safe to re-run)
    - Top-K:     Return K most similar vectors to query
    """

    def __init__(self, embedder: EmbeddingModel):
        from pinecone import Pinecone, ServerlessSpec

        self.embedder = embedder
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # Create index if it doesn't exist
        index_name = settings.pinecone_index_name
        existing = [i.name for i in pc.list_indexes()]

        if index_name not in existing:
            print(f"🔧 Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=embedder.dimension,    # Must match embedding model!
                metric="cosine",                 # Cosine similarity
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"           # Free tier region
                )
            )
            print(f"✅ Index created: {index_name}")
        else:
            print(f"✅ Using existing index: {index_name}")

        self.index = pc.Index(index_name)

    def upsert(self, vectors: list[dict],
               namespace: str = "default",
               batch_size: int = 100) -> None:
        """
        Store vectors in Pinecone.
        
        WHY BATCH UPSERTS?
        Pinecone recommends max 100 vectors per request.
        Larger batches get rejected or slow down significantly.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"  Upserted {min(i+batch_size, len(vectors))}"
                  f"/{len(vectors)} vectors")

    def search(self, query_vector: list[float],
               top_k: int = 5,
               namespace: str = "default",
               filter: dict = None) -> list[dict]:
        """
        Find top-K most similar chunks to the query vector.
        
        FILTER example (metadata filtering):
        filter={"doc_type": {"$eq": "pdf"}}
        → Only search PDF chunks, ignore web chunks
        
        Returns list of matches with scores and metadata.
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,    # We need the text content back!
            filter=filter
        )

        return [
            {
                "id": match.id,
                "score": round(match.score, 4),
                "content": match.metadata.get("content", ""),
                "source": match.metadata.get("source", ""),
                "doc_type": match.metadata.get("doc_type", ""),
                "metadata": match.metadata
            }
            for match in results.matches
        ]

    def delete(self, ids: list[str],
               namespace: str = "default") -> None:
        self.index.delete(ids=ids, namespace=namespace)

    def get_stats(self) -> dict:
        return self.index.describe_index_stats()


# ─────────────────────────────────────────────
# Chroma — Local dev vector store (no API key)
# ─────────────────────────────────────────────
class ChromaVectorStore(BaseVectorStore):
    """
    Chroma runs fully locally — perfect for development
    and testing without needing Pinecone credentials.
    
    Data persists to disk at ./chroma_db/
    """

    def __init__(self, embedder: EmbeddingModel,
                 persist_dir: str = "./chroma_db"):
        import chromadb
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Chroma ready | Collection: rag_knowledge")

    def upsert(self, vectors: list[dict],
               namespace: str = "default") -> None:
        ids = [v["id"] for v in vectors]
        embeddings = [v["values"] for v in vectors]
        metadatas = [v["metadata"] for v in vectors]
        documents = [v["metadata"].get("content", "") for v in vectors]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"✅ Upserted {len(vectors)} vectors to Chroma")

    def search(self, query_vector: list[float],
               top_k: int = 5,
               namespace: str = "default",
               filter: dict = None) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter
        )

        matches = []
        for i, doc_id in enumerate(results["ids"][0]):
            matches.append({
                "id": doc_id,
                "score": round(1 - results["distances"][0][i], 4),
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "doc_type": results["metadatas"][0][i].get("doc_type", ""),
                "metadata": results["metadatas"][0][i]
            })

        return matches

    def delete(self, ids: list[str],
               namespace: str = "default") -> None:
        self.collection.delete(ids=ids)

    def get_stats(self) -> dict:
        return {"total_vectors": self.collection.count()}


# ─────────────────────────────────────────────
# Factory — picks the right store automatically
# ─────────────────────────────────────────────
def get_vector_store(embedder: EmbeddingModel,
                     store_type: str = "auto") -> BaseVectorStore:
    """
    Factory function — returns correct vector store.
    
    "auto" logic:
    - Pinecone key exists → use Pinecone (production)
    - No key → use Chroma (local dev)
    
    This means your RAG pipeline code NEVER needs to know
    which database it's talking to. Clean separation.
    """
    if store_type == "auto":
        store_type = "pinecone" if settings.pinecone_api_key else "chroma"

    print(f"🗄️  Vector store: {store_type.upper()}")

    if store_type == "pinecone":
        return PineconeVectorStore(embedder)
    elif store_type == "chroma":
        return ChromaVectorStore(embedder)
    else:
        raise ValueError(f"Unknown store type: {store_type}")