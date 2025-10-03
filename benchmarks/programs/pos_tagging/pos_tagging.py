import nightjarpy as nj


@nj.fn
def main(language) -> callable:
    """natural
    Based on the <language>, define a function called <:annotate> that performs part-of-speech tagging. The function should take a sentence `text` and return a list of (word, tag) pairs, where the tags are appropriate for the grammar of the given <language>. Also, save the possible parts-of-speech tags of the <language> to <:tags>
    """

    return tags, annotate


#### Tests ####

from typing import Any, Callable, Dict, List, Tuple


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        tags, annotator = main(language)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
