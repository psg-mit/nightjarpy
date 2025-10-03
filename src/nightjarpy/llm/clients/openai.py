import json
import logging
import os
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import dotenv
import openai
from openai.types.chat import ParsedChatCompletion
from openai.types.responses import ResponseCodeInterpreterToolCall

from nightjarpy.effects import Effect
from nightjarpy.llm.base import (
    LLM,
    ChatMessage,
    LLMConfig,
    LLMUsage,
    ResponseFormat,
    ResponseType,
    ToolCall,
)
from nightjarpy.types import (
    Argument,
    AssistantMessage,
    JsonType,
    ToolMessage,
    UserMessage,
)
from nightjarpy.utils import with_cache
from nightjarpy.utils.utils import (
    NJ_TELEMETRY,
    openai_schema_to_function_schema,
    to_strict_json_schema,
)

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class OpenAI(LLM):
    """OpenAI LLM implementation."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.config = config
        self._last_usage: Optional[LLMUsage] = None

        if config.container:
            container = self.client.containers.create(name="test-container")
            self.container_id = container.id
        else:
            self.container_id = None

        available_models = self.client.models.list().data
        available_models = [model.id for model in available_models]

        if config.model.replace("openai/", "") not in available_models:
            raise ValueError(f"Unknown model: {config.model}")

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
        Send a chat completion request to OpenAI and return tool calls.

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

        # Prepare the request parameters
        request_params = {
            "model": self.config.model.replace("openai/", ""),
            "messages": [message.to_openai() for message in messages],
            "temperature": temperature or self.config.temperature,
            "max_completion_tokens": max_tokens or self.config.max_tokens,
            "tools": [tool.to_openai_function() for tool in tools],
            "tool_choice": tool_choice or self.config.tool_choice,
            "parallel_tool_calls": parallel_tool_calls or self.config.parallel_tool_calls,
        }

        if self.config.model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            request_params["reasoning_effor"] = self.config.reasoning_effort
            request_params["verbosity"] = self.config.verbosity

        # Add system prompt
        if system is not None:
            request_params["messages"] = [
                {
                    "role": "system",
                    "content": system,
                }
            ] + request_params["messages"]

        logger.info("OpenAI Query: " + str(messages[-1]))

        try:
            # Make the request
            if self.config.cache:
                response_json = with_cache(
                    lambda **p: self.client.beta.chat.completions.parse(**p).model_dump_json(), request_params
                )
                response = ParsedChatCompletion.model_validate_json(response_json)
            else:
                response = self.client.beta.chat.completions.parse(**request_params)

            logger.info(f"OpenAI Response: {response}")

            if len(response.choices) == 0:
                raise RuntimeError("LLM returned nothing")

            # Store usage information
            if hasattr(response, "usage") and response.usage:
                self._last_usage = LLMUsage.from_openai_usage(response.usage)

            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls or []
                wrapped_tool_calls: List[ToolCall] = []
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"Failed to decode function call arguments: {call.function.arguments} of function {call.function.name} in tool call {call.id}"
                        )
                    wrapped_tool_calls.append(
                        ToolCall(
                            name=call.function.name,
                            args=args,
                            id=call.id,
                        )
                    )
            else:
                raise RuntimeError("LLM did not return tool calls")

            return wrapped_tool_calls

        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {str(e)}")

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

        # Prepare the request parameters
        request_params = {
            "model": self.config.model.replace("openai/", ""),
            "messages": [message.to_openai() for message in messages],
            "temperature": temperature or self.config.temperature,
            "max_completion_tokens": max_tokens or self.config.max_tokens,
        }

        if schema is not None:
            request_params["response_format"] = schema.to_openai_schema()
            # request_params["tools"] = [openai_schema_to_function_schema(schema)]
            # request_params["tool_choice"] = {"type": "function", "function": {"name": schema.name}}

        if self.config.model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            request_params["reasoning_effort"] = self.config.reasoning_effort
            request_params["verbosity"] = self.config.verbosity

        # Add system prompt
        if system is not None:
            request_params["messages"] = [
                {
                    "role": "system",
                    "content": system,
                }
            ] + request_params["messages"]

        logger.info("OpenAI Query: " + str(messages[-1]))

        try:
            # Make the request
            if self.config.cache:
                response_json = with_cache(
                    lambda **p: self.client.beta.chat.completions.parse(**p).model_dump_json(), request_params
                )
                response = ParsedChatCompletion.model_validate_json(response_json)
            else:
                response = self.client.beta.chat.completions.parse(**request_params)

            logger.info(f"OpenAI Response: {response}")

            if len(response.choices) == 0:
                raise RuntimeError("LLM returned nothing")

            # Store usage information
            if hasattr(response, "usage") and response.usage:
                self._last_usage = LLMUsage.from_openai_usage(response.usage)

            # if response.choices[0].finish_reason == "stop":
            #     if response.choices[0].message.content is None:
            #         raise RuntimeError(f"LLM did not respond")
            #     if schema is not None:
            #         return schema.parse(response.choices[0].message.content)
            #     else:
            #         return response.choices[0].message.content
            # else:
            #     raise RuntimeError(f"LLM finished for reason: {response.choices[0].finish_reason}")

            choice = response.choices[0]

            if schema is not None:
                return schema.parse(choice.message.content or "")
                # # Handle structured output via function calling
                # if choice.finish_reason in ["stop", "tool_calls"] and choice.message.tool_calls:
                #     tool_call = choice.message.tool_calls[0]
                #     if tool_call.function.name == schema.name:
                #         try:
                #             return schema.parse(tool_call.function.arguments)
                #         except (json.JSONDecodeError, ValueError) as e:
                #             raise RuntimeError(f"Failed to parse structured output: {str(e)}")
                #     else:
                #         raise RuntimeError(f"Unexpected function call: {tool_call.function.name}")
                # else:
                #     raise RuntimeError(f"Expected function call but got finish_reason: {choice.finish_reason}")
            else:
                # Handle regular text output
                if choice.finish_reason == "stop":
                    if choice.message.content is None:
                        raise RuntimeError("LLM did not respond")
                    return choice.message.content
                else:
                    raise RuntimeError(f"LLM finished for reason: {choice.finish_reason}")

        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {str(e)}")

    def gen_structured_output(
        self,
        messages: List[ChatMessage],
        schema: ResponseFormat[ResponseType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ResponseType:
        """
        Send a chat completion request to OpenAI and return structured output.

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
        tool_trace: List[ChatMessage] = [UserMessage(content=message)]

        # container = {"type": "auto"} if not self.container_id else self.container_id

        request_params = {
            "model": self.config.model.replace("openai/", ""),
            "instructions": system,
            "input": message,
            "tools": [
                {
                    "type": "code_interpreter",
                    "container": self.container_id,
                }  # type:ignore
            ],
            "tool_choice": "required",
            "temperature": self.config.temperature,
            "parallel_tool_calls": False,
            "max_output_tokens": self.config.max_tokens,
        }
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
            request_params["text"] = {"format": schema_dict}

        res = self.client.responses.create(**request_params)

        logger.info(f"OpenAI Response: {res}")

        if res.usage:
            self._last_usage = LLMUsage.from_openai_response_usage(res.usage)
            NJ_TELEMETRY.log_llm_usage(filename=__name__, funcname="main", usage=self._last_usage)
        else:
            raise ValueError(f"No usage data in response: {res}")

        # count how many tool calls used
        n_tool_calls = 0
        for out in res.output:
            if isinstance(out, ResponseCodeInterpreterToolCall):
                n_tool_calls += 1
                if out.outputs:
                    tool_output = json.dumps([x.model_dump() for x in out.outputs])
                else:
                    tool_output = ""
                tool_trace.extend(
                    [
                        AssistantMessage(
                            tool_calls=[ToolCall(name="code_interpreter_tool", args={"code": out.code}, id=out.id)]
                        ),
                        ToolMessage(content=tool_output, tool_call_id=out.id),
                    ]
                )

        self.n_tool_calls = n_tool_calls

        NJ_TELEMETRY.n_tool_calls += n_tool_calls
        NJ_TELEMETRY.log_messages(tool_trace)

        if schema is not None:
            output = schema.parse(res.output_text)
        else:
            output = res.output_text

        return output

    def get_usage(self) -> Optional[LLMUsage]:
        """Get token usage information from the last request."""
        return self._last_usage
