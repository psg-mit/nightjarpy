from typing import Any, Dict, List, Tuple

import nightjarpy as nj


@nj.fn
def search_across(word_pattern: str):
    def search_across_coroutine(hint: str):
        """natural
        Return a word that fits the <word_pattern> based on the <hint>.
        """

    return search_across_coroutine


@nj.fn
def match_check(word_pattern: str, search_coroutine):
    """Function that checks proposed words against the clue pattern."""
    proposed_word = search_coroutine("")  # Start the coroutine
    for i in range(5):
        """natural
        Does the word <proposed_word> satisfy the pattern provided in <word_pattern>? Is it a real word? If it's valid, break. If not valid, provide hints to adjust word proposal based on pattern compliance and store them as a string to <:hints>
        """
        proposed_word = search_coroutine(hints)

    return proposed_word


def main(word_pattern: str):
    search_coroutine = search_across(word_pattern)
    return match_check(word_pattern, search_coroutine)


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
            outputs[f"test_{i}"] = main(word_pattern)
        except Exception as e:
            errors[f"test_{i}"] = e

        try:
            if outputs[f"test_{i}"] is not None:
                hard_results[f"test_{i}"] = outputs[f"test_{i}"].lower() in expected
        except Exception as e:
            errors[f"test_{i}"] = e
    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
