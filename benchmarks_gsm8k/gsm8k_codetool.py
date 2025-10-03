import logging
import re
from typing import Optional

from pydantic import BaseModel

from nightjarpy.llm import LLM, LLMConfig
from nightjarpy.llm.factory import create_llm
from nightjarpy.types import ResponseFormat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


class Res(BaseModel):
    ans: float


def main(problem: str, llm: LLM, max_tool_calls: int, container_id: Optional[str] = None) -> float:
    res = llm.gen_code_interpreter(
        system="Solve the math problem using the python code tool.",
        message=problem,
        max_tool_calls=max_tool_calls,
        schema=ResponseFormat(Res),
    )

    if res is None or isinstance(res, str):
        raise ValueError("Did not get an answer")
    ans = res.ans

    return ans


#### Tests ####
from typing import Any, Dict, List, Tuple

if __name__ == "__main__":
    llm = create_llm(LLMConfig(model="openai/gpt-4.1-2025-04-14", container=True))
    print(
        main(
            problem="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            llm=llm,
            max_tool_calls=100,
        )
    )
