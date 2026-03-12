# app/core/claude_client.py
import anthropic
from typing import Generator
from app.core.config import settings

class ClaudeClient:
    """
    Wrapper around Anthropic API.
    
    Why a wrapper? 
    - Centralizes all Claude configuration
    - Makes it easy to swap models later
    - Adds retry logic, logging in one place
    - Separates concerns (RAG logic vs API calls)
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        
    def generate(self, 
                 user_message: str, 
                 system_prompt: str,
                 context_chunks: list[str] = None) -> str:
        """
        Standard (non-streaming) generation.
        Used for: testing, simple queries
        """
        # Build the full prompt with retrieved context
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            full_message = f"""Here is the relevant context retrieved from the knowledge base:

<context>
{context_text}
</context>

Using ONLY the context above, answer this question:
{user_message}

If the context doesn't contain enough information, say so clearly."""
        else:
            full_message = user_message
            
        response = self.client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": full_message}]
        )
        
        return response.content[0].text
    
    def generate_stream(self,
                        user_message: str,
                        system_prompt: str, 
                        context_chunks: list[str] = None) -> Generator[str, None, None]:
        """
        Streaming generation — yields text tokens as they arrive.
        Used for: real-time UI responses (much better UX)
        
        KEY CONCEPT: Streaming means Claude sends tokens one by one
        instead of waiting for the full response. Critical for RAG
        where responses can be long.
        """
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            full_message = f"""Here is the relevant context retrieved from the knowledge base:

<context>
{context_text}
</context>

Using ONLY the context above, answer this question:
{user_message}

If the context doesn't contain enough information, say so clearly."""
        else:
            full_message = user_message

        # stream=True returns a context manager
        with self.client.messages.stream(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": full_message}]
        ) as stream:
            for text in stream.text_stream:
                yield text  # Each yield = one token chunk

# System prompt for RAG — this is critical
RAG_SYSTEM_PROMPT = """You are an intelligent knowledge assistant with access to a curated knowledge base.

Your behavior rules:
1. Answer questions ONLY based on the provided context chunks
2. If context is insufficient, explicitly state what's missing
3. Always be precise — cite which part of the context supports your answer
4. Never hallucinate or add information beyond the provided context
5. If asked something outside your knowledge base, say: "This topic isn't covered in my knowledge base"

Your tone: Professional, clear, and concise."""