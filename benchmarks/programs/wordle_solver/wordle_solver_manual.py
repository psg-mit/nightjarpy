from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from nightjarpy import nj_llm_factory


class SearchAcrossResult(BaseModel):
    proposed_word: str


class ValidateResult(BaseModel):
    valid: bool
    reason: Optional[str]


class HintsResult(BaseModel):
    hints: str


def search_across(word_pattern: str, nj_llm):
    def search_across_coroutine(hints: str):
        result: SearchAcrossResult = nj_llm(
            f"Given a Wordle pattern <word_pattern> and optional <hints>, propose a single valid "
            "English word that fits. Respond only with proposed_word.\n"
            f"<word_pattern>{word_pattern}</word_pattern>\n<hints>{hints}</hints>",
            output_format=SearchAcrossResult,
        )
        return result.proposed_word

    return search_across_coroutine


def match_check(word_pattern: str, search_coroutine, nj_llm):
    """Function that checks proposed words against the clue pattern."""
    proposed_word = search_coroutine("")  # Start the coroutine
    for i in range(5):
        validate: ValidateResult = nj_llm(
            "Evaluate if <word> satisfies the Wordle <pattern> and is a real English word. "
            "Return valid=true only if both are satisfied.\n"
            f"<word>{proposed_word}</word>\n<pattern>{word_pattern}</pattern>",
            output_format=ValidateResult,
        )
        if validate.valid:
            break
        else:
            hints_res: HintsResult = nj_llm(
                "Given the Wordle <pattern> and the invalid <word>, provide concise hints to "
                "adjust the next proposal (e.g., letter positions, inclusions/exclusions).\n"
                f"<pattern>{word_pattern}</pattern>\n<word>{proposed_word}</word>",
                output_format=HintsResult,
            )
            proposed_word = search_coroutine(hints_res.hints)

    return proposed_word


def main(word_pattern: str, nj_llm):
    search_coroutine = search_across(word_pattern, nj_llm)
    return match_check(word_pattern, search_coroutine, nj_llm)


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    hard_results = {}

    tests = [
        (
            "S_A_E",
            [
                "scale",
                "scape",
                "scare",
                "seame",
                "seare",
                "sease",
                "seaze",
                "shade",
                "shake",
                "shale",
                "shame",
                "shape",
                "share",
                "shave",
                "skate",
                "slade",
                "slake",
                "slane",
                "slate",
                "slave",
                "smaze",
                "snake",
                "snare",
                "soare",
                "soave",
                "space",
                "spade",
                "spake",
                "spale",
                "spane",
                "spare",
                "spate",
                "stade",
                "stage",
                "stake",
                "stale",
                "stane",
                "stare",
                "state",
                "stave",
                "suave",
                "swage",
                "swale",
                "sware",
            ],
        ),
        (
            "_O__D",
            [
                "abode",
                "adopt",
                "blond",
                "blood",
                "board",
                "broad",
                "chord",
                "cloud",
                "crowd",
                "demon",
                "flood",
                "found",
                "hoard",
                "hound",
                "modal",
                "model",
                "moldy",
                "mound",
                "proud",
                "scold",
                "sound",
                "sword",
                "world",
                "would",
            ],
        ),
        (
            "__IN_",
            [
                "acing",
                "aging",
                "aping",
                "being",
                "blind",
                "blink",
                "bring",
                "china",
                "cling",
                "doing",
                "drink",
                "dying",
                "eking",
                "eying",
                "faint",
                "fling",
                "going",
                "grind",
                "icing",
                "lying",
                "opine",
                "ovine",
                "owing",
                "paint",
                "point",
                "print",
                "saint",
                "shine",
                "sling",
                "slink",
                "spine",
                "sting",
                "stink",
                "suing",
                "swing",
                "taint",
                "thing",
                "twine",
                "tying",
                "urine",
                "using",
                "whine",
                "wring",
            ],
        ),
    ]

    for i, (word_pattern, expected) in enumerate(tests):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False

        try:
            outputs[f"test_{i}"] = main(word_pattern, nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e

        try:
            if outputs[f"test_{i}"] is not None:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"].lower() in expected
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
