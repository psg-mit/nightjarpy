import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

import anthropic
import dotenv
from anthropic.types.beta import (
    BetaJSONOutputFormatParam,
    BetaServerToolUseBlock,
    BetaTextBlock,
)

from nightjarpy.effects import Effect
from nightjarpy.llm.base import (
    LLM,
    ChatMessage,
    LLMUsage,
    ResponseFormat,
    ResponseType,
    ToolCall,
)
from nightjarpy.types import Argument, AssistantMessage, JsonType, UserMessage
from nightjarpy.utils import with_cache
from nightjarpy.utils.utils import NJ_TELEMETRY, to_strict_json_schema

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class Anthropic(LLM):
    """Anthropic LLM implementation."""

    def __init__(self, config):
        super().__init__(config)
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self._last_usage: Optional[LLMUsage] = None

    def get_usage(self) -> Optional[LLMUsage]:
        """Get token usage information from the last request."""
        return self._last_usage

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
        if len(messages) == 0:
            raise RuntimeError("No messages")

        tool_choice = tool_choice or self.config.tool_choice
        parallel_tool_calls = parallel_tool_calls or self.config.parallel_tool_calls

        if tool_choice == "required":
            tool_choice_anthropic = {"type": "any", "disable_parallel_tool_use": not parallel_tool_calls}
        elif tool_choice == "auto":
            tool_choice_anthropic = {"type": "auto", "disable_parallel_tool_use": not parallel_tool_calls}
        else:
            # Note: Not an option right now
            assert isinstance(tool_choice, dict)
            # forced tool choice
            tool_choice_anthropic = {
                "type": "tool",
                "name": tool_choice["function"]["name"],
            }

        anthropic_messages = [message.to_anthropic() for message in messages]

        # TODO: when parallel calls, need to merge the tool results into one "user" content chunk

        # Add cache control
        if anthropic_messages[-1]["role"] == "user":
            if isinstance(anthropic_messages[-1]["content"], str):
                anthropic_messages[-1]["content"] = [
                    {
                        "type": "text",
                        "text": anthropic_messages[-1]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        else:
            # if isinstance(anthropic_messages[-1]["content"], str):
            #     anthropic_messages[-1]["content"] = [
            #         {
            #             "type": "text",
            #             "text": anthropic_messages[-1]["content"],
            #             "cache_control": {"type": "ephemeral"},
            #         }
            #     ]
            # else:
            #     anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Get tool schema and add cache control
        tool_schema = [tool.to_anthropic_function() for tool in tools]
        if len(tools) > 0:
            tool_schema[-1]["cache_control"] = {"type": "ephemeral"}

        # Prepare the request parameters
        request_params = {
            "model": self.config.model.replace("anthropic/", ""),
            "messages": anthropic_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "tools": tool_schema,
            "tool_choice": tool_choice_anthropic,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        }

        logger.info("Anthropic Query: " + str(messages[-1]))

        try:
            # Make the request
            if self.config.cache:
                response_json = with_cache(
                    lambda **p: self.client.messages.create(**p).model_dump_json(), request_params
                )
                response = anthropic.types.Message.model_validate_json(response_json)
            else:
                response = self.client.messages.create(**request_params)
            response: anthropic.types.Message

            logger.info(f"Anthropic Response: {response}")

            if len(response.content) == 0:
                raise RuntimeError("LLM returned nothing")

            # Store usage information
            if hasattr(response, "usage") and response.usage:
                self._last_usage = LLMUsage.from_anthropic_usage(response.usage)

            if response.content[0].type == "tool_use":
                wrapped_tool_calls: List[ToolCall] = []
                for call in response.content:
                    if call.type == "tool_use":
                        assert isinstance(call, anthropic.types.ToolUseBlock)
                        wrapped_tool_calls.append(
                            ToolCall(
                                name=call.name,
                                args=cast(Dict[str, Any], call.input),
                                id=call.id,
                            )
                        )
            else:
                raise RuntimeError("LLM did not return tool calls")

            return wrapped_tool_calls

        except Exception as e:
            raise RuntimeError(f"Anthropic API request failed: {str(e)}")

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
        if len(messages) == 0:
            raise RuntimeError("No messages")

        anthropic_messages = [message.to_anthropic() for message in messages]

        # Add cache control
        if anthropic_messages[-1]["role"] == "user":
            if isinstance(anthropic_messages[-1]["content"], str):
                anthropic_messages[-1]["content"] = [
                    {
                        "type": "text",
                        "text": anthropic_messages[-1]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        else:
            anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Prepare the request parameters
        request_params = {
            "model": self.config.model.replace("anthropic/", ""),
            "messages": anthropic_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        if system is not None:
            request_params["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        logger.info("Anthropic Query: " + str(messages[-1]))

        if schema is not None:
            request_params["tools"] = [schema.to_anthropic_schema()]
            request_params["tool_choice"] = {"type": "tool", "name": schema.name, "disable_parallel_tool_use": True}

        try:
            # Make the request
            if self.config.cache:
                response_json = with_cache(
                    lambda **p: self.client.messages.create(**p).model_dump_json(), request_params
                )
                response = anthropic.types.Message.model_validate_json(response_json)
            else:
                response = self.client.messages.create(**request_params)
            response: anthropic.types.Message

            logger.info(f"Anthropic Response: {response}")

            if len(response.content) == 0:
                raise RuntimeError("LLM returned nothing")

            # Store usage information
            if hasattr(response, "usage") and response.usage:
                self._last_usage = LLMUsage.from_anthropic_usage(response.usage)

            if response.content[0].type == "tool_use":
                wrapped_tool_calls: List[ToolCall] = []
                for call in response.content:
                    if call.type == "tool_use":
                        assert isinstance(call, anthropic.types.ToolUseBlock)
                        wrapped_tool_calls.append(
                            ToolCall(
                                name=call.name,
                                args=cast(Dict[str, Any], call.input),
                                id=call.id,
                            )
                        )
            elif response.content[0].type == "text" and schema is None:
                res_content = response.content[0]
                assert isinstance(res_content, anthropic.types.TextBlock)
                return res_content.text
            else:
                raise RuntimeError("LLM did not return tool calls")

            # using tool to enforce schema, now just return the result
            if schema is not None:
                return schema.parse(json.dumps(wrapped_tool_calls[0].args))

            res = []
            for x in response.content:
                assert isinstance(x, anthropic.types.TextBlock)
                res.append(x)
            return "".join(res)

        except Exception as e:
            raise RuntimeError(f"Anthropic API request failed: {str(e)}")

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

        res = self.gen(
            messages=messages,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        assert not isinstance(res, str)
        return res

    def gen_code_interpreter(
        self,
        message: str,
        system: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
        schema: Optional[ResponseFormat[ResponseType]] = None,
    ) -> Optional[str | ResponseType]:
        assert isinstance(self.client, anthropic.Anthropic)

        tool_trace: List[ChatMessage] = [UserMessage(content=message)]

        output_schema_str = ""
        if schema is not None:
            if not isinstance(schema.res_schema, Dict):
                # BaseModel
                pydantic_dict = schema.res_schema.model_json_schema()  # type: ignore
                pydantic_dict = to_strict_json_schema(pydantic_dict)
                schema_dict = {
                    "type": "object",
                    "properties": pydantic_dict["properties"],
                    "required": list(pydantic_dict["properties"].keys()),  # type: ignore
                    "additionalProperties": False,
                }
                if "$defs" in pydantic_dict:
                    schema_dict["$defs"] = pydantic_dict["$defs"]
                schema_dict = {
                    "type": "json_schema",
                    "name": pydantic_dict["title"],
                    "schema": schema_dict,
                    "strict": True,
                }
            else:
                schema_dict = {
                    "type": "json_schema",
                    "name": schema.res_schema["json_schema"]["name"],  # type: ignore
                    "schema": schema.res_schema["json_schema"]["schema"],  # type: ignore
                    "strict": True,
                }

            output_schema_str += f"\n\nFormat output as JSON:\n{schema_dict}"

        # NOTE: Anthropic's code execution currently does not expose container/session control
        # in the same way as OpenAI's code interpreter, so `max_tool_calls` and `container_id`
        # are not used here.
        request_params = {
            "model": self.config.model.replace("anthropic/", ""),
            "betas": ["code-execution-2025-08-25"],
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "tools": [
                {
                    "type": "code_execution_20250825",
                    "name": "code_execution",
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "system": (system or "") + output_schema_str,
        }

        res = self.client.beta.messages.create(**request_params)

        logger.info(f"Anthropic Response: {res}")

        if res.usage:
            self._last_usage = LLMUsage.from_anthropic_usage(res.usage)
            NJ_TELEMETRY.log_llm_usage(filename=__name__, funcname="main", usage=self._last_usage)
        else:
            raise ValueError(f"No usage data in response: {res}")

        # Count how many tool calls were used and collect the final text output
        n_tool_calls = 0
        output_text: Optional[str] = None
        for out in res.content:
            if isinstance(out, BetaServerToolUseBlock):
                n_tool_calls += 1
                # Log the actual tool call as an AssistantMessage with a ToolCall payload
                tool_trace.append(
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(
                                name=out.name,
                                args=cast(Dict[str, Any], out.input),
                                id=out.id,
                            )
                        ]
                    )
                )
            elif isinstance(out, BetaTextBlock):
                output_text = out.text

        self.n_tool_calls = n_tool_calls  # type: ignore[attr-defined]

        NJ_TELEMETRY.n_tool_calls += n_tool_calls
        NJ_TELEMETRY.log_messages(tool_trace)

        if output_text is None:
            return None

        if schema is not None:
            output_text = "{" + output_text.split("{", 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
            return schema.parse(output_text)

        return output_text
