import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings:

    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- LLM GATEWAY (LITELLM) ---
    LITELLM_PROXY_URL = os.getenv(
    "LITELLM_PROXY_URL",
    "http://localhost:4000"
    )
    LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")


    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")


    # --- OBSERVABILITY (LANGSMITH) ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com"
    )


# Apply LangChain environment variables
os.environ["LANGCHAIN_TRACING_V2"] = Settings.LANGSMITH_TRACING
os.environ["LANGCHAIN_API_KEY"] = Settings.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = Settings.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = Settings.LANGSMITH_ENDPOINT


settings = Settings()