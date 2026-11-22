from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.config import settings


# LiteLLM model alias
# Actual provider models and fallback are configured in litellm_config.yaml.
PRIMARY_MODEL = "rag-primary"


def get_langchain_llm(
    feature: str = "rag",
    model: str = PRIMARY_MODEL,
    temperature: float = 0.0
) -> ChatOpenAI:
    """
    Returns a LangChain-compatible LLM routed through LiteLLM.

    LiteLLM manages provider selection, retries and fallback.
    """

    return ChatOpenAI(
        model=model,
        api_key=settings.LITELLM_MASTER_KEY,
        base_url=settings.LITELLM_PROXY_URL,
        temperature=temperature,
        extra_body={
            "metadata": {
                "tags": [feature, "rag-system", "dev"]
            }
        }
    )


# Direct OpenAI-compatible client for non-LangChain calls.
litellm_client = OpenAI(
    api_key=settings.LITELLM_MASTER_KEY,
    base_url=settings.LITELLM_PROXY_URL
)


def extract_cache_status(raw_response) -> str:
    """
    Extracts LiteLLM cache status from response headers.
    """

    return (
        "hit"
        if raw_response.headers.get("x-litellm-cache-key")
        else "miss"
    )