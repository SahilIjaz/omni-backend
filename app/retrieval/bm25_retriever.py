# app/retrieval/bm25_retriever.py
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.ingestion.chunker import Chunk


class BM25Retriever:
    """
    Keyword-based retrieval using BM25 algorithm.
    
    WHY BM25 alongside vectors?
    - Exact term matching for: model names, version numbers,
      technical terms, proper nouns, IDs
    - Zero hallucination risk (only matches real text)
    - Computationally cheap once index is built
    - Catches what semantic search misses
    
    LIFECYCLE:
    1. Build index from chunks (one-time)
    2. Save to disk (so you don't rebuild every run)
    3. Load and search at query time
    """

    def __init__(self):
        self.bm25 = None
        self.chunks = []          # Keep chunks to return content from
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple but effective tokenization.
        
        Steps:
        1. Lowercase everything
        2. Split on non-alphanumeric characters
        3. Remove stopwords
        4. Keep tokens > 2 chars
        
        WHY REMOVE STOPWORDS?
        Words like 'the', 'is', 'at' appear everywhere
        and hurt BM25's ability to discriminate relevant docs.
        """
        stopwords = {
            "the", "is", "at", "which", "on", "a", "an",
            "and", "or", "but", "in", "to", "of", "for",
            "with", "as", "by", "from", "this", "that",
            "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will",
            "would", "can", "could", "should", "may"
        }

        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from chunks.
        Call this after ingestion, before searching.
        """
        print(f"🔧 Building BM25 index for {len(chunks)} chunks...")

        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.content) for chunk in chunks
        ]

        # BM25Okapi is the standard variant — best general performance
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ BM25 index built")

    def save(self, path: str = "./bm25_index.pkl") -> None:
        """Persist index to disk — avoid rebuilding every restart."""
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        print(f"💾 BM25 index saved to {path}")

    def load(self, path: str = "./bm25_index.pkl") -> bool:
        """Load index from disk. Returns True if successful."""
        if not Path(path).exists():
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
            self.tokenized_corpus = data["tokenized_corpus"]

        print(f"✅ BM25 index loaded from {path}")
        return True

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search using BM25 keyword matching.
        Returns results in same format as vector store
        for easy merging.
        """
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:           # Only return actual matches
                chunk = self.chunks[idx]
                results.append({
                    "id": chunk.chunk_id,
                    "score": round(float(scores[idx]), 4),
                    "content": chunk.content,
                    "source": chunk.source,
                    "doc_type": chunk.doc_type,
                    "metadata": chunk.metadata,
                    "retrieval_type": "bm25"
                })

        return results