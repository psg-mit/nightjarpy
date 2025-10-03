from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


def main(text: str, nj_llm):
    masked_text = nj_llm(
        f"Mask all occurrences of sensitive information found in the <text> by replacing them with asterisks of the same length as the characters being replaced.\n<text>{text}</text>"
    )
    return masked_text


#### Tests ####

from typing import Any, Dict, List, Tuple


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    inputs_outputs = [
        ("my email is alice@gmail.com", "my email is ***************"),
        ("call me at 101-456-8099", "call me at ************"),
    ]
    outputs = {}
    errors = {}
    hard_results = {}

    for i, (input_value, expected_output) in enumerate(inputs_outputs):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False

        try:
            res = main(input_value, nj_llm)
            outputs[f"test_{i}"] = res
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = res == expected_output
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
