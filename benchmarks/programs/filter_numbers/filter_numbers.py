from typing import Any, Dict, List, Tuple

import nightjarpy as nj


@nj.fn
def main(numbers: List[int], prop: str):
    """natural
    Filter out the numbers from <numbers> that don't satisfy <prop>.
    Store the result in <:filtered_numbers>.
    """
    return filtered_numbers


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

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
            outputs[f"test_{i}"] = main(inp, prop)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] == expected
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
