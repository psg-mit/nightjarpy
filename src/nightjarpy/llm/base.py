import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from optparse import Option
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
)

from nightjarpy.configs import LLMConfig
from nightjarpy.effects import Effect
from nightjarpy.types import (
    ChatMessage,
    JsonType,
    ResponseFormat,
    ResponseType,
    ToolCall,
)
from nightjarpy.utils.utils import LLMUsage

logger = logging.getLogger(__name__)


class LLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        if self.config.cache:
            logger.info("CACHING IS ENABLED")

    @abstractmethod
    def gen_tool_calls(
        self,
        messages: List[ChatMessage],
        tools: Sequence[Effect],
        tool_choice: Optional[Literal["auto", "required"]] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> List[ToolCall]:
        """
        Send a chat completion request to the LLM and return tool calls.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions
            tool_choice: How to handle tool calls
            parallel_tool_calls: Whether to allow parallel tool calls
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt to use for the conversation

        Returns:
            List of ToolCall objects representing the tool calls made by the LLM
        """
        ...

    @abstractmethod
    def gen(
        self,
        messages: List[ChatMessage],
        schema: Optional[ResponseFormat[ResponseType]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> str | ResponseType:
        """
        Generate a response from the LLM, optionally returning a structured output.

        Args:
            messages: List of ChatMessage objects representing the conversation history.
            schema: Optional ResponseFormat specifying the expected structured output. If None, returns a string.
            temperature: Optional float for sampling temperature.
            max_tokens: Optional int for the maximum number of tokens to generate.
            system: Optional string for the system prompt.

        Returns:
            Either a string response or a structured object of type ResponseType, depending on the schema.
        """
        ...

    @abstractmethod
    def gen_structured_output(
        self,
        messages: List[ChatMessage],
        schema: ResponseFormat[ResponseType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ResponseType:
        """
        Send a chat completion request to the LLM and return structured output.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            schema: Schema defining the expected structured output format
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt to use for the conversation

        Returns:
            The structured output parsed and returned according to the provided schema.
        """
        ...

    @abstractmethod
    def gen_code_interpreter(
        self,
        message: str,
        system: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
        schema: Optional[ResponseFormat[ResponseType]] = None,
    ) -> Optional[str | ResponseType]: ...

    """
    Usage the official code interpreter tool to execute instructions

    Args:
        message: User message of instruction
        system: System prompt
        max_tool_calls: Optional maximum number of tool calls to make

    Returns:
        The string output of the code interpreter execution, or None if no output is produced.
    """

    @abstractmethod
    def get_usage(self) -> Optional[LLMUsage]:
        """Get token usage information from the last request."""
        ...
