from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class FilterEvenNumbersLLMResult(BaseModel):
    filtered_numbers: List[int] = Field(default_factory=list)


def main(numbers: List[int], prop, nj_llm):
    result: FilterEvenNumbersLLMResult = nj_llm(
        "Filter out the numbers from the <numbers> that don't satisfy <prop>"
        "Return the list in the 'filtered_numbers' field.\n"
        f"<numbers>{numbers}</numbers>"
        f"<prop>{prop}</prop>",
        output_format=FilterEvenNumbersLLMResult,
    )

    return result.filtered_numbers


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    inps = [
        ([-1234, 2017, 1345134, 1802, 500, 2025], "Number must be a reasonable birth year", [2017, 2025]),
        ([42, 911, 128, 1337, 572], "cultural references", [42, 911, 1337]),
        ([1234, 1111, 2748, 7777, 8392], "might be a password", [1234, 1111, 7777]),
    ]
    outputs = {}
    errors = {}
    hard_results = {}

    for i, (inp, prop, expected) in enumerate(inps):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False

        try:
            outputs[f"test_{i}"] = main(inp, prop, nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] == expected
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
