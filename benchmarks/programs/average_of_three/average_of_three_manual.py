from pydantic import BaseModel

from nightjarpy import nj_llm_factory


class Response(BaseModel):
    res: int


def main(p, q, r, nj_llm) -> float:
    average = nj_llm(
        f"Compute the average of <p>, <q>, and <r>.\n<p>{p}</p>\n<q>{q}</q>\n<r>{r}</r>",
        output_format=Response,
    ).res
    return average


#### Tests ####
import logging
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)


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
            outputs[f"test_{i}"] = main(*inp, nj_llm=nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"] == 2
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm=nj_llm)
    print(results)
    print(hard_results)
    print(errors)
