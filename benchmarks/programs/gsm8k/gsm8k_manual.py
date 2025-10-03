from openai import BaseModel

from nightjarpy import nj_llm_factory


def main(problem, nj_llm):

    res = nj_llm(
        f"Write the expression to evaluate with `eval` to answer to the math problem:{problem}. Give just the expression."
    )
    ans = eval(res)

    return ans


#### Tests ####
from typing import Any, Dict, List, Tuple


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

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
            outputs[f"test_{i}"] = main(query, nj_llm)
            hard_results[f"test_{i}"] = outputs[f"test_{i}"] == ans
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
