# app/retrieval/reranker.py
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker for precise relevance scoring.
    
    DIFFERENCE: Bi-encoder vs Cross-encoder
    
    Bi-encoder (what we used for embeddings):
        query  → encode → vector A  ┐
        chunk  → encode → vector B  ┘ → cosine similarity
        Fast! But approximate. Encodes query & chunk separately.
    
    Cross-encoder:
        [query + chunk together] → single relevance score
        Slower. But MUCH more accurate. Sees full interaction
        between query and chunk text — understands nuance.
    
    WHY use both?
    Bi-encoder is fast enough to scan 100,000 chunks.
    Cross-encoder is too slow for that but perfect for 
    re-scoring the top 20 candidates bi-encoder returns.
    
    This two-stage approach gives you both speed AND accuracy.
    """

    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🔄 Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✅ Reranker loaded")

    def rerank(self,
               query: str,
               candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        Re-score candidate chunks against query.
        
        INPUT:  ~15-20 merged candidates from hybrid retrieval
        OUTPUT: top_k truly relevant chunks, precisely scored
        
        HOW IT WORKS:
        CrossEncoder takes (query, chunk_text) pairs and outputs
        a single relevance score — it reads them TOGETHER so it
        understands context, negation, specificity much better.
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, c["content"]) for c in candidates]

        # Score all pairs in one batch pass
        scores = self.model.predict(pairs)

        # Attach new scores to candidates
        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append({
                **candidate,
                "rerank_score": round(float(score), 4),
                "original_score": candidate.get("score", 0)
            })

        # Sort by rerank score, return top_k
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]