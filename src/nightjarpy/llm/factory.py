from typing import TYPE_CHECKING

from nightjarpy.configs import LLMConfig

if TYPE_CHECKING:
    from nightjarpy.llm.base import LLM


def create_llm(config: LLMConfig) -> "LLM":
    """
    Factory function to create the appropriate LLM instance based on model name.

    Args:
        config: LLM configuration containing model name

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If model provider is not supported
    """
    model = config.model.lower()

    if model.startswith("openai/"):
        from nightjarpy.llm.clients.openai import OpenAI

        return OpenAI(config)
    elif model.startswith("anthropic/"):
        from nightjarpy.llm.clients.anthropic import Anthropic

        return Anthropic(config)
    else:
        raise ValueError(f"Unsupported model provider: {model}. Supported providers: openai/, anthropic/")
