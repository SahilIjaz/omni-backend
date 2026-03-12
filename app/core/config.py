# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-knowledge-agent"
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.1

    class Config:
        env_file = ".env"

settings = Settings()