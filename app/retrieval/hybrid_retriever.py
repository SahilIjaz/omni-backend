# app/retrieval/hybrid_retriever.py
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import BaseVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker


class HybridRetriever:
    """
    Combines vector search + BM25 + reranking into one pipeline.
    
    FULL RETRIEVAL FLOW:
    
    Query
      ├── Vector Search  (semantic)  → top 10 results
      ├── BM25 Search    (keyword)   → top 10 results
      │         ↓
      │    Reciprocal Rank Fusion    ← merge strategy
      │         ↓
      │    ~15-20 unique candidates
      │         ↓
      └── Reranker                   ← precision pass
               ↓
          Final top-K chunks
    """

    def __init__(self,
                 embedder: EmbeddingModel,
                 vector_store: BaseVectorStore,
                 bm25_retriever: BM25Retriever,
                 reranker: Reranker):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.reranker = reranker

    def _reciprocal_rank_fusion(self,
                                 vector_results: list[dict],
                                 bm25_results: list[dict],
                                 k: int = 60) -> list[dict]:
        """
        Merge two ranked lists into one using RRF algorithm.
        
        WHY RRF (Reciprocal Rank Fusion)?
        Vector scores and BM25 scores are on different scales:
        - Vector: 0.0 to 1.0 (cosine similarity)
        - BM25:   0.0 to 20+ (unbounded relevance score)
        
        You CANNOT just average them — they're incomparable.
        
        RRF formula: score(d) = Σ 1/(k + rank(d))
        
        Instead of using raw scores, it uses RANK POSITION.
        Being #1 in either list is worth the same regardless
        of the actual score value. Elegant and robust.
        
        k=60 is the standard constant that dampens outliers.
        """
        fused_scores = {}
        all_results = {}

        # Score from vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1/(k + rank + 1)
            result["retrieval_type"] = "vector"
            all_results[doc_id] = result

        # Score from BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1/(k + rank + 1)
            # If same doc found by both — mark it (strong signal!)
            if doc_id in all_results:
                all_results[doc_id]["retrieval_type"] = "hybrid"
            else:
                result["retrieval_type"] = "bm25"
                all_results[doc_id] = result

        # Sort by fused RRF score
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        merged = []
        for doc_id in sorted_ids:
            result = all_results[doc_id].copy()
            result["rrf_score"] = round(fused_scores[doc_id], 6)
            merged.append(result)

        return merged

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 vector_top_k: int = 10,
                 bm25_top_k: int = 10,
                 namespace: str = "default",
                 use_reranker: bool = True,
                 filter: dict = None) -> list[dict]:
        """
        Full hybrid retrieval pipeline.
        
        PARAMS:
        - query:        user's question
        - top_k:        final chunks to return to LLM
        - vector_top_k: how many vector results to fetch first
        - bm25_top_k:   how many BM25 results to fetch first
        - use_reranker: toggle reranking (slower but better)
        - filter:       metadata filter e.g. {"doc_type": "pdf"}
        """
        print(f"\n🔍 Hybrid Retrieval: '{query[:60]}...'")

        # ── Step 1: Vector search ────────────────────────────
        query_vector = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_vector,
            top_k=vector_top_k,
            namespace=namespace,
            filter=filter
        )
        print(f"  Vector results: {len(vector_results)}")

        # ── Step 2: BM25 search ──────────────────────────────
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)
        print(f"  BM25 results:   {len(bm25_results)}")

        # ── Step 3: Merge with RRF ───────────────────────────
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
        print(f"  Merged unique:  {len(merged)}")

        # ── Step 4: Rerank ───────────────────────────────────
        if use_reranker and len(merged) > 0:
            final = self.reranker.rerank(query, merged, top_k=top_k)
            print(f"  After rerank:   {len(final)}")
        else:
            final = merged[:top_k]

        # ── Step 5: Log retrieval sources ───────────────────
        for i, r in enumerate(final):
            rtype = r.get("retrieval_type", "unknown")
            score = r.get("rerank_score", r.get("rrf_score", 0))
            print(f"  [{i+1}] {rtype:8} | score: {score:.4f} | "
                  f"{r['content'][:60]}...")

        return final