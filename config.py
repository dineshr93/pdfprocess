"""Single source of truth for endpoints, model and chunk settings."""
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

EMBED_BASE = "http://localhost:8001/v1"
LLM_BASE = "http://localhost:8888/v1"
CHROMA_PATH = "./chroma_db"
COLLECTION = "complex_pdfs"
PDF_DIR = "./pdfs"

# The local embedding server ignores the model name; it serves mxbai-embed-large-v1 (1024 dim).
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_base=EMBED_BASE,
    api_key="not-needed",
    embed_batch_size=32,
)
Settings.llm = OpenAILike(
    model="local-model",
    api_base=LLM_BASE,
    api_key="not-needed",
    is_chat_model=True,
    temperature=0.1,
    max_tokens=4096,
    context_window=8192,
    timeout=180,
)
Settings.chunk_size = 400
Settings.chunk_overlap = 60
