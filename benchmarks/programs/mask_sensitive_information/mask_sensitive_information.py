import nightjarpy as nj


@nj.fn
def main(text: str) -> str:
    """natural
    Mask all occurrences of sensitive information found in the <text> by replacing them with asterisks of the same length as the characters being replaced. Then store it in <:masked_text>.
    """
    return masked_text


#### Tests ####


from typing import Any, Dict, List, Tuple


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
            res = main(input_value)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
