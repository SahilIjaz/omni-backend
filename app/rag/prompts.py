# app/rag/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ─────────────────────────────────────────────────────
# Main RAG Prompt
# This is the most critical prompt in the system.
# Every word matters — it defines HOW Claude uses context.
# ─────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert knowledge assistant with access \
to a curated knowledge base.

STRICT RULES:
1. Answer ONLY from the provided context chunks below
2. If context is insufficient, say exactly what's missing
3. Always cite your sources using [Source: X] format
4. Never hallucinate or add outside knowledge
5. Be precise — prefer specific over general answers
6. If multiple chunks conflict, acknowledge the discrepancy

CONTEXT CHUNKS:
{context}

METADATA:
- Total chunks retrieved: {chunk_count}
- Sources: {sources}
- Retrieval method: {retrieval_method}"""),

    # Chat history placeholder — enables multi-turn conversation
    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{question}")
])


# ─────────────────────────────────────────────────────
# Query Analysis Prompt
# Before retrieval, analyze what the user actually wants
# ─────────────────────────────────────────────────────
QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's question and return a JSON response.

Return ONLY valid JSON, no explanation:
{{
    "query_type": "factual|conceptual|procedural|comparative",
    "key_terms": ["term1", "term2"],
    "expanded_query": "rewritten query optimized for search",
    "requires_context": true/false,
    "complexity": "simple|medium|complex"
}}

Query types:
- factual:     specific fact lookup ("what is X", "when was Y")
- conceptual:  understanding a concept ("how does X work")
- procedural:  step-by-step ("how to do X")  
- comparative: comparison ("X vs Y", "difference between")"""),

    ("human", "Analyze this query: {question}")
])


# ─────────────────────────────────────────────────────
# Fallback Prompt (when no relevant context found)
# ─────────────────────────────────────────────────────
NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. The knowledge base 
did not return relevant results for this query.

Tell the user:
1. No relevant information was found in the knowledge base
2. What topics ARE covered (based on available sources: {sources})
3. Suggest how they might rephrase their question"""),

    ("human", "{question}")
])