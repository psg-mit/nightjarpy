from nightjarpy import nj_llm_factory


def main(prompt: str, expression: str, nj_llm) -> str:
    balanced_expression = nj_llm(
        "Insert/repair parentheses based on <prompt> in the <expression> string. Give just the "
        f"balanced expression\n<prompt>{prompt}</prompt>\n<expression>{expression}</expression>"
    )
    return balanced_expression


#### Tests ####

import logging
from typing import Any, Dict, List, Tuple


def run(nj_llm=None) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    inps = [
        (
            "Apply f to sum of x and 1, then divide by 1 substracted from g of x",
            "f x + 1 / g x - 1",
            [
                "f(x + 1) / (g(x) - 1)",
                "(f(x + 1)) / (g(x) - 1)",
                "(f(x + 1) / (g(x) - 1))",
                "((f(x + 1)) / (g(x) - 1))",
            ],
        ),
        (
            "Regex should capture just the area code of the phone number",
            r"^(\d{3}-\d{3}\d{4}$",
            [r"^(\d{3})-\d{3}\d{4}$"],
        ),
    ]
    outputs = {}
    errors = {}
    hard_results = {}

    for i, (prompt, expression, answers) in enumerate(inps):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False

        try:
            outputs[f"test_{i}"] = main(prompt, expression, nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] in answers
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


from nightjarpy.configs import LLMConfig

if __name__ == "__main__":

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm=nj_llm)
    print(results)
    print(hard_results)
    print(errors)
