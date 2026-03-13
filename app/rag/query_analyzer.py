# app/rag/query_analyzer.py
import json
from langchain_anthropic import ChatAnthropic
from app.rag.prompts import QUERY_ANALYSIS_PROMPT
from app.core.config import settings


class QueryAnalyzer:
    """
    Analyzes incoming queries BEFORE retrieval.
    
    WHY ANALYZE FIRST?
    The raw user query is often not optimal for search.
    
    User types:    "whats the diff between claude opus and sonnet"
    Optimized:     "Claude Opus vs Sonnet comparison capabilities
                    performance use cases differences"
    
    The expanded query retrieves dramatically better chunks.
    This pre-retrieval step is called "Query Rewriting" or
    "HyDE" (Hypothetical Document Embeddings) in RAG literature.
    """

    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0      # Deterministic — we want consistent JSON
        )
        self.chain = QUERY_ANALYSIS_PROMPT | self.llm

    def analyze(self, question: str) -> dict:
        """
        Returns structured analysis of the user's question.
        Falls back gracefully if JSON parsing fails.
        """
        try:
            response = self.chain.invoke({"question": question})
            text = response.content.strip()

            # Strip markdown code fences if present
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            analysis = json.loads(text)

            print(f"\n📊 Query Analysis:")
            print(f"   Type:     {analysis.get('query_type')}")
            print(f"   Terms:    {analysis.get('key_terms')}")
            print(f"   Expanded: {analysis.get('expanded_query')[:80]}...")

            return analysis

        except Exception as e:
            # Graceful fallback — never block the pipeline
            print(f"⚠️  Query analysis failed: {e}, using raw query")
            return {
                "query_type": "factual",
                "key_terms": question.split()[:5],
                "expanded_query": question,
                "requires_context": True,
                "complexity": "simple"
            }