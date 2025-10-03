import json
import logging
import re
from contextlib import redirect_stdout
from io import StringIO

from nightjarpy import NJ_TELEMETRY
from nightjarpy.effects import Effect, Parameter
from nightjarpy.llm import LLM, LLMConfig
from nightjarpy.llm.factory import create_llm
from nightjarpy.types import (
    AssistantMessage,
    ChatMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

effects = [
    Effect(
        name="exec",
        description="Execute a Python code block. Returns what's printed to standard out.",
        parameters=(Parameter("code", str),),
        handler=None,
    ),
    Effect(
        name="answer",
        description="Answer the problem",
        parameters=(Parameter("ans", float),),
        handler=None,
    ),
]


def main(problem: str, llm: LLM, max_tool_calls: int) -> float:
    messages: List[ChatMessage] = [UserMessage(content=problem)]
    for i in range(max_tool_calls):
        res = llm.gen_tool_calls(
            system="Solve the math problem using tools. Return only the answer as a float with the format: #### {answer}",
            messages=messages,
            tool_choice="required",
            tools=effects,
        )
        messages.append(AssistantMessage(tool_calls=res))
        for tool_call in res:
            if tool_call.name == "exec":
                f = StringIO()
                with redirect_stdout(f):
                    # exec a piece of python code
                    exec(tool_call.args["code"], {})
                s = f.getvalue()
                messages.append(
                    ToolMessage(
                        content=s,
                        tool_call_id=tool_call.id,
                    )
                )
            elif tool_call.name == "answer":
                NJ_TELEMETRY.log_messages(messages)
                NJ_TELEMETRY.n_tool_calls = i + 1
                return float(tool_call.args["ans"])

    raise ValueError("Max tool calls")


#### Tests ####
from typing import Any, Dict, List, Tuple

if __name__ == "__main__":
    llm = create_llm(LLMConfig(model="openai/gpt-4.1"))
    print(
        main(
            problem="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            llm=llm,
            max_tool_calls=100,
        )
    )
