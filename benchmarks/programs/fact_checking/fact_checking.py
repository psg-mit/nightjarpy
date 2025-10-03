from typing import Any, Dict, List, Tuple

import nightjarpy as nj


@nj.fn
def main(passage: str):
    """natural
    Save to <:facts_to_check> a list of claims in the <passage> that need to be fact-checked, whether you think they are true or false.
    """

    checks = []
    for fact in facts_to_check:
        """natural
        Provide a reasoning explanation for why the <fact> might be true or might be false

        The explanation should be a short paragraph that uses background knowledge or logical inference to support or refute the <fact>.

        Append a tuple of (the <fact>, true/false, explanation) to <checks>
        """

    return checks


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
            {"The Declaration of Independence was signed in 1776": True, "The Roman Empire fell in 476 AD": True},
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
            outputs[f"test_{i}"] = main(text)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
