from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class FactCheck(BaseModel):
    claim: str
    is_true: bool
    explanation: str


class FactCheckingLLMResult(BaseModel):
    checks: List[FactCheck] = Field(default_factory=list)


def main(passage: str, nj_llm):

    result: FactCheckingLLMResult = nj_llm(
        "Identify claims in the <passage> that warrant fact-checking. "
        "For each claim, decide if you believe it is true (best guess) and provide a brief reasoning. "
        f"Return results in the structured 'checks' field.\n<passage>{passage}</passage>",
        output_format=FactCheckingLLMResult,
    )

    checks: List[Tuple[str, bool, str]] = [(item.claim, item.is_true, item.explanation) for item in result.checks]
    return checks


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    hard_results = {}

    tests = [
        (
            "The Eiffel Tower is located in Paris. The Great Wall of China is visible from space. Bananas grow on trees.",
            {
                "The Eiffel Tower is located in Paris": True,
                "The Great Wall of China is visible from space": False,
                "Bananas grow on trees": False,
            },
        ),
        (
            "The Declaration of Independence was signed in 1776. The Roman Empire fell in 476 AD.",
            {
                "The Declaration of Independence was signed in 1776": True,
                "The Roman Empire fell in 476 AD": True,
            },
        ),
        ("Hello! How are you today? It's a nice day.", {}),
    ]

    for i, (text, expected) in enumerate(tests):
        for j in range(len(expected)):
            hard_results[f"test_{i}_fact_{j}"] = False
        if len(expected) == 0:
            hard_results[f"test_{i}"] = False

    for i, (text, expected) in enumerate(tests):
        outputs[f"test_{i}"] = None

        try:
            outputs[f"test_{i}"] = main(text, nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e

        if outputs.get(f"test_{i}", None) is not None:
            actual_checks = {}
            for fact, is_true, _ in outputs[f"test_{i}"]:
                actual_checks[fact.replace(".", "").strip()] = is_true
            if len(expected) == 0:
                hard_results[f"test_{i}"] = len(actual_checks) == 0
            else:
                for j, fact in enumerate(expected):
                    if fact in actual_checks:
                        hard_results[f"test_{i}_fact_{j}"] = (
                            actual_checks[fact.replace(".", "").strip()] == expected[fact.replace(".", "").strip()]
                        )
                    else:
                        hard_results[f"test_{i}_fact_{j}"] = False

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
