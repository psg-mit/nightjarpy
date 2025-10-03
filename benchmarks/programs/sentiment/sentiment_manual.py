from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel

from nightjarpy import nj_llm_factory


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]


def main(s: str, nj_llm) -> Literal["positive", "neutral", "negative"]:
    result: SentimentResult = nj_llm(
        "Classify <s> as either positive, neutral, or negative and save the result to " f"`sentiment`.\n<s>{s}</s>",
        output_format=SentimentResult,
    )
    return result.sentiment


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        outputs["test_0"] = main(input_val, nj_llm)
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
        outputs["test_1"] = main(input_val, nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
