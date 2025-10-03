import nightjarpy as nj


@nj.fn
def main(s: str):
    """natural
    Classify <s> as either positive, neutral, or negative and save the result to <:sentiment>
    """
    return sentiment


#### Tests ####
from typing import Any, Dict, List, Tuple


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    """
    Executes main() and returns a list of tuples of input/output pairs.
    """
    outputs = {}
    errors = {}
    hard_results = {
        "sentiment_is_string_1": False,
        "sentiment_is_positive": False,
        "sentiment_is_string_2": False,
        "sentiment_is_negative": False,
    }

    try:
        input_val = "I absolutely loved the new movie; it was a thrilling experience from start to finish!"
        outputs["test_0"] = main(input_val)
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            hard_results["sentiment_is_string_1"] = isinstance(outputs["test_0"], str)
            hard_results["sentiment_is_positive"] = outputs["test_0"].lower() == "positive"
        except Exception as e:
            errors["test_0"] = e

    try:
        input_val = "The service at the restaurant was disappointing and ruined our evening."
        outputs["test_1"] = main(input_val)
    except Exception as e:
        errors["test_1"] = e
    else:
        try:
            hard_results["sentiment_is_string_2"] = isinstance(outputs["test_1"], str)
            hard_results["sentiment_is_negative"] = outputs["test_1"].lower() == "negative"
        except Exception as e:
            errors["test_1"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
