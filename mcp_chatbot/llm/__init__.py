from typing import Literal

from ..config.configuration import Configuration
from .ollama import OllamaClient

__all__ = ["OllamaClient", "create_llm_client"]


LLMProvider = Literal["ollama"]


def create_llm_client(provider: LLMProvider, config: Configuration) -> OllamaClient:
    """Create appropriate LLM client based on provider.

    Args:
        provider: LLM provider type ("ollama" or any provided llm)
        config: Configuration object containing LLM model name, API key, and base URL

    Returns:
        Initialized LLM client instance
    """

    if provider == "ollama":
        return OllamaClient(
            model_name=config.ollama_model_name,
            api_base=config.ollama_base_url,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
