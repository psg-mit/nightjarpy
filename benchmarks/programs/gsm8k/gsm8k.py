import nightjarpy as nj


@nj.fn
def main(problem):
    """natural
    Save to <:ans> the answer to the math <problem>.
    """

    return ans


#### Tests ####
from typing import Any, Dict, List, Tuple


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    examples = [
        (
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            18,
        ),
        ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
        (
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            70000,
        ),
    ]
    outputs = {}
    errors = {}
    hard_results = {
        "test_0": False,
        "test_1": False,
        "test_2": False,
    }

    for i, (query, ans) in enumerate(examples):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None

        try:
            outputs[f"test_{i}"] = main(query)
            hard_results[f"test_{i}"] = outputs[f"test_{i}"] == ans
        except Exception as e:
            errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
