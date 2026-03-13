# app/retrieval/embedder.py
import os
import time
from sentence_transformers import SentenceTransformer
from app.ingestion.chunker import Chunk


class EmbeddingModel:
    """
    Converts text into dense vector representations.
    
    WHY sentence-transformers over OpenAI embeddings?
    - Free & runs locally — no API cost per embedding
    - 'all-MiniLM-L6-v2' is fast and production-proven
    - Same model MUST be used for both indexing AND querying
      (critical — mixing models breaks similarity search)
    
    DIMENSION: This model outputs 384-dim vectors.
    Pinecone index must be created with dimension=384.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded | Dimension: {self.dimension}")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string → vector.
        Used for: embedding user queries at search time.
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_chunks(self,
                     chunks: list[Chunk],
                     batch_size: int = 32) -> list[dict]:
        """
        Embed a list of chunks in batches.
        
        WHY BATCHING?
        Embedding models are optimized for batch processing.
        Batches of 32 are ~10x faster than one-by-one.
        
        Returns list of dicts ready for vector DB upsert:
        {
            "id":     chunk_id,
            "values": [0.23, -0.81, ...],   ← the vector
            "metadata": { source, content, ... }
        }
        """
        results = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]

            print(f"  Embedding batch {i//batch_size + 1}"
                  f"/{-(-len(chunks)//batch_size)}...")

            vectors = self.model.encode(texts, convert_to_numpy=True)

            for chunk, vector in zip(batch, vectors):
                results.append({
                    "id": chunk.chunk_id,
                    "values": vector.tolist(),
                    "metadata": {
                        "content": chunk.content,      # Store text in metadata!
                        "source": chunk.source,        # For citations
                        "doc_type": chunk.doc_type,
                        "chunk_index": chunk.chunk_index,
                        **{k: str(v) for k, v in chunk.metadata.items()
                           if isinstance(v, (str, int, float, bool))}
                    }
                })

        return results