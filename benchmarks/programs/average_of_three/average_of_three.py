import nightjarpy as nj


@nj.fn
def main(p, q, r):
    """natural
    Parse and compute the average of <p>, <q>, and <r>, and store it in <:average> as a float
    """

    return average


#### Tests ####

from typing import Any, Dict, List, Tuple


def run(nj_llm=None) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    inps = [
        (1.0, 2.0, 3.0),
        ("one", "two", "three"),
        ("uno", "二", "три"),
    ]
    outputs = {}
    errors = {}
    hard_results = {}

    for i, inp in enumerate(inps):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False

        try:
            outputs[f"test_{i}"] = main(*inp)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] == 2
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":

    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
