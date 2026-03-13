# app/rag/pipeline.py
from typing import Generator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from app.rag.prompts import RAG_PROMPT, NO_CONTEXT_PROMPT
from app.rag.context_builder import ContextBuilder
from app.rag.query_analyzer import QueryAnalyzer
from app.retrieval.hybrid_retriever import HybridRetriever
from app.core.config import settings


class RAGPipeline:
    """
    The brain of the system — orchestrates every component.
    
    DESIGN PRINCIPLES:
    1. Query first, retrieve second  (analysis improves retrieval)
    2. Hybrid always beats single    (vector + BM25 > either alone)
    3. Rerank before generate        (quality > speed for final answer)
    4. Stream the response           (UX > latency)
    5. Always cite sources           (trust > raw answers)
    
    LANGCHAIN EXPRESSION LANGUAGE (LCEL):
    We use the pipe operator | to compose chains:
    prompt | llm | output_parser
    
    This is lazy — nothing runs until .invoke() or .stream()
    is called. It's also composable — swap any component easily.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.context_builder = ContextBuilder()
        self.query_analyzer = QueryAnalyzer()

        # LangChain Claude wrapper (for LCEL chains)
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        # LCEL chains using pipe operator
        self.rag_chain = RAG_PROMPT | self.llm
        self.fallback_chain = NO_CONTEXT_PROMPT | self.llm

        # Conversation memory (in-memory for now)
        # Phase 6 will add persistent memory
        self.chat_history: list = []

    def _build_prompt_inputs(self,
                              question: str,
                              context_data: dict) -> dict:
        """Assemble all inputs the prompt template needs."""
        return {
            "question": question,
            "context": context_data["context"],
            "chunk_count": context_data["chunk_count"],
            "sources": ", ".join(context_data["sources"]) or "none",
            "retrieval_method": context_data["retrieval_method"],
            "chat_history": self.chat_history
        }

    def query(self,
              question: str,
              top_k: int = 5,
              namespace: str = "default",
              use_reranker: bool = True,
              doc_type_filter: str = None) -> dict:
        """
        Standard (non-streaming) RAG query.
        Returns complete response with sources.
        """
        print(f"\n{'='*60}")
        print(f"RAG QUERY: {question}")
        print(f"{'='*60}")

        # ── Step 1: Analyze query ────────────────────────────
        analysis = self.query_analyzer.analyze(question)
        search_query = analysis.get("expanded_query", question)

        # ── Step 2: Build metadata filter ───────────────────
        filter_dict = None
        if doc_type_filter:
            filter_dict = {"doc_type": {"$eq": doc_type_filter}}

        # ── Step 3: Hybrid retrieval ─────────────────────────
        chunks = self.retriever.retrieve(
            query=search_query,
            top_k=top_k,
            use_reranker=use_reranker,
            namespace=namespace,
            filter=filter_dict
        )

        # ── Step 4: Build context ────────────────────────────
        context_data = self.context_builder.build(chunks)

        # ── Step 5: Choose prompt + generate ─────────────────
        if context_data["chunk_count"] == 0:
            response = self.fallback_chain.invoke({
                "question": question,
                "sources": "No documents ingested yet"
            })
        else:
            inputs = self._build_prompt_inputs(question, context_data)
            response = self.rag_chain.invoke(inputs)

        answer = response.content

        # ── Step 6: Update chat history ──────────────────────
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # Keep history bounded — last 10 exchanges (20 messages)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return {
            "answer": answer,
            "sources": context_data["sources"],
            "chunks_used": context_data["chunk_count"],
            "retrieval_method": context_data["retrieval_method"],
            "was_truncated": context_data["was_truncated"],
            "query_analysis": analysis
        }

    def query_stream(self,
                     question: str,
                     top_k: int = 5,
                     namespace: str = "default",
                     use_reranker: bool = True) -> Generator:
        """
        Streaming RAG query — yields tokens as they arrive.
        
        YIELDS:
        - {"type": "metadata", ...}   first — sends retrieval info
        - {"type": "token",   ...}    per token — the answer stream
        - {"type": "done",    ...}    last — signals completion
        
        This structure lets the frontend show sources immediately
        while the answer streams in — great UX pattern.
        """
        # Steps 1-4 same as non-streaming
        analysis = self.query_analyzer.analyze(question)
        search_query = analysis.get("expanded_query", question)

        chunks = self.retriever.retrieve(
            query=search_query,
            top_k=top_k,
            use_reranker=use_reranker,
            namespace=namespace
        )

        context_data = self.context_builder.build(chunks)

        # Yield metadata first — frontend can show sources
        # before generation even starts
        yield {
            "type": "metadata",
            "sources": context_data["sources"],
            "chunks_used": context_data["chunk_count"],
            "retrieval_method": context_data["retrieval_method"],
            "query_analysis": analysis
        }

        # Stream generation token by token
        full_answer = ""
        inputs = self._build_prompt_inputs(question, context_data)

        for chunk in self.rag_chain.stream(inputs):
            token = chunk.content
            full_answer += token
            yield {"type": "token", "token": token}

        # Update history with complete answer
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_answer))

        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        yield {"type": "done", "full_answer": full_answer}

    def clear_history(self):
        """Reset conversation — start fresh session."""
        self.chat_history = []
        print("🗑️  Chat history cleared")