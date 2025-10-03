from typing import Any, Dict, List, Tuple

import nightjarpy as nj


@nj.fn
def proposer(query):
    """Coroutine that proposes random values."""
    suggestion = ""
    while True:
        """natural
        Given <query> and <suggestion>, propose a solution and store it to <:proposal>.
        """
        suggestion = yield proposal


@nj.fn
def verifier(query, proposer_coroutine):
    """Function that verifies proposed values."""
    proposal = next(proposer_coroutine)  # Start the coroutine
    while True:
        """natural
        Does the proposed solution <proposal> satisfy the user query: <query>, assign your answer to <:accepted>.
        """
        if accepted:
            break
        else:
            """natural
            come up with suggestions for improvement given the unaccepted <proposal> for <query> and store them to <:suggestions>
            """
            proposal = proposer_coroutine.send(suggestions)

    return proposal


def main(riddle: str):
    proposer_coroutine = proposer(riddle)
    return verifier(riddle, proposer_coroutine)


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    riddle = "I am the beginning of the end, and the end of time and space. I am essential to creation, and I surround every place. What am I?"

    outputs = {}
    errors = {}
    hard_results = {
        "proposal_is_string": False,
        "proposal_is_non_empty": False,
    }

    try:
        outputs["test_0"] = main(riddle)
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            hard_results["proposal_is_string"] = isinstance(outputs["test_0"], str)
            hard_results["proposal_is_non_empty"] = len(outputs["test_0"].strip()) > 0
        except Exception as e:
            errors["test_0"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
