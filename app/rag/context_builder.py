# app/rag/context_builder.py


class ContextBuilder:
    """
    Formats retrieved chunks into a clean context block for Claude.
    
    WHY A SEPARATE CLASS?
    How you format context dramatically affects answer quality.
    Bad formatting = Claude loses track of which info came from where.
    Good formatting = Claude cites correctly and reasons clearly.
    
    We also handle:
    - Deduplication (same chunk retrieved twice)
    - Token budget management (don't overflow context window)
    - Source tracking (for citations)
    """

    def __init__(self, max_context_chars: int = 8000):
        # ~8000 chars ≈ 2000 tokens — leaves room for prompt + response
        self.max_context_chars = max_context_chars

    def build(self, retrieved_chunks: list[dict]) -> dict:
        """
        Convert raw retrieved chunks into formatted context.
        
        Returns dict with everything the prompt needs:
        {
            context:          formatted text block for Claude
            sources:          unique source list for citations  
            chunk_count:      number of chunks used
            retrieval_method: what combination was used
            was_truncated:    whether we hit token limit
        }
        """
        if not retrieved_chunks:
            return {
                "context": "No relevant information found.",
                "sources": [],
                "chunk_count": 0,
                "retrieval_method": "none",
                "was_truncated": False
            }

        # Step 1: Deduplicate by content hash
        seen_content = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            content_hash = hash(chunk["content"][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)

        # Step 2: Build context respecting token budget
        context_parts = []
        total_chars = 0
        used_chunks = []
        was_truncated = False

        for i, chunk in enumerate(unique_chunks):
            chunk_text = self._format_chunk(chunk, i + 1)

            if total_chars + len(chunk_text) > self.max_context_chars:
                was_truncated = True
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            used_chunks.append(chunk)

        # Step 3: Track sources for citations
        sources = list(dict.fromkeys([
            c["source"] for c in used_chunks
        ]))

        # Step 4: Determine retrieval method mix
        retrieval_types = set(
            c.get("retrieval_type", "vector") for c in used_chunks
        )
        if "hybrid" in retrieval_types or (
            "vector" in retrieval_types and "bm25" in retrieval_types
        ):
            method = "hybrid (vector + BM25 + reranking)"
        elif "bm25" in retrieval_types:
            method = "BM25 keyword search"
        else:
            method = "vector semantic search"

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "chunk_count": len(used_chunks),
            "retrieval_method": method,
            "was_truncated": was_truncated,
            "used_chunks": used_chunks
        }

    def _format_chunk(self, chunk: dict, index: int) -> str:
        """
        Format a single chunk with clear boundaries and metadata.
        
        WHY THIS FORMAT?
        XML-style tags help Claude clearly distinguish between
        chunks and understand metadata. Claude is trained to
        respect these boundaries in its reasoning.
        """
        source = chunk.get("source", "unknown")
        doc_type = chunk.get("doc_type", "unknown")
        score = chunk.get("rerank_score",
                chunk.get("rrf_score",
                chunk.get("score", 0)))

        return f"""<chunk id="{index}" source="{source}" \
type="{doc_type}" relevance="{score:.3f}">
{chunk['content']}
</chunk>"""