# test_claude.py
from app.core.claude_client import ClaudeClient, RAG_SYSTEM_PROMPT

client = ClaudeClient()

# Simulate what RAG will do — pass fake "retrieved" context
fake_context = [
    "FastAPI is a modern Python web framework for building APIs. It uses Python type hints and is based on Starlette.",
    "FastAPI supports async/await natively and generates OpenAPI docs automatically at /docs endpoint."
]

question = "What is FastAPI and does it support async?"

print("🤖 Standard response:")
response = client.generate(question, RAG_SYSTEM_PROMPT, fake_context)
print(response)

print("\n🤖 Streaming response:")
for token in client.generate_stream(question, RAG_SYSTEM_PROMPT, fake_context):
    print(token, end="", flush=True)
print()