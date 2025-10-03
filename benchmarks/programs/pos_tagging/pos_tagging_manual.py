from typing import Callable, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class TokenTag(BaseModel):
    token: str
    tag: str


class PosTaggingInitResult(BaseModel):
    tags: List[str]


class PosTaggingAnnotateResult(BaseModel):
    pairs: List[TokenTag]


def main(language: str, nj_llm) -> Tuple[List[str], Callable[[str], List[Tuple[str, str]]]]:
    init_result: PosTaggingInitResult = nj_llm(
        "Given the <language>, list the canonical parts-of-speech tags used to annotate text. "
        "Save them to `tags` as uppercase strings (e.g., 'NOUN', 'VERB', 'ADJ'). Return only "
        f"tags that are appropriate for the language.\n<language>{language}</language>",
        output_format=PosTaggingInitResult,
    )

    tags = init_result.tags

    def annotate(text: str) -> List[Tuple[str, str]]:
        annotate_result: PosTaggingAnnotateResult = nj_llm(
            "Tokenize the <text> and assign each token a part-of-speech tag from <tags>. "
            "Return <:pairs> as a list of objects with fields `token` and `tag`. Use only tags "
            "from <tags> and preserve token order.\n"
            f"<language>{language}</language>\n<text>{text}</text>\n<tags>{tags}</tags>",
            output_format=PosTaggingAnnotateResult,
        )

        return [(p.token, p.tag) for p in annotate_result.pairs]

    return tags, annotate


#### Tests ####

from typing import Any, Callable, Dict, List, Tuple


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    language = "English"
    text = "The quick brown fox jumps over the lazy dog"
    expected_output = [
        ("The", "DT"),
        ("quick", "JJ"),
        ("brown", "NN"),
        ("fox", "NN"),
        ("jumps", "VBZ"),
        ("over", "IN"),
        ("the", "DT"),
        ("lazy", "JJ"),
        ("dog", "NN"),
    ]
    outputs = {}
    errors = {}
    hard_results = {}
    for i in range(len(expected_output)):
        hard_results[f"test_{i}"] = False
    try:
        tags, annotator = main(language, nj_llm)
        outputs["test"] = annotator(text)
    except Exception as e:
        errors["test"] = e
    else:

        try:
            for i, (expected_tag, output_tag) in enumerate(zip(expected_output, outputs["test"])):
                hard_results[f"test_{i}"] = False
                try:
                    hard_results[f"test_{i}"] = expected_tag == output_tag
                except Exception as e:
                    errors[f"test_{i}"] = e

        except Exception as e:
            errors[f"test"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
