import nightjarpy as nj


@nj.fn
def main(prompt: str, expression: str) -> str:
    """natural
    Insert/repair parentheses based on <prompt> in the <expression> string. Store only the result in <:balanced_expression>.
    """
    return balanced_expression


#### Tests ####
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
            outputs[f"test_{i}"] = main(prompt, expression)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] in answers
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
