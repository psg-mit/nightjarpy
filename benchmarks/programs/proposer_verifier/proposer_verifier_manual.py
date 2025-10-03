from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class ProposalResult(BaseModel):
    """LLM output for proposing a solution."""

    proposal: str = Field(description="A concrete proposed solution to the query, concise but specific.")


class VerificationResult(BaseModel):
    """LLM output for verifying a proposal against the query."""

    accepted: bool = Field(description="True if the proposal satisfies the query; False otherwise.")


class SuggestionResult(BaseModel):
    """LLM output for suggesting improvements to a rejected proposal."""

    suggestions: str = Field(description="Actionable suggestions on how to modify the proposal to satisfy the query.")


def proposer(query, nj_llm):
    """Coroutine that proposes values, refining based on verifier feedback."""
    suggestion = ""
    while True:
        prop_out: ProposalResult = nj_llm(
            "Given <query> and <suggestion>, propose a solution and store it to `proposal`.\n"
            f"<query>{query}</query>\n<suggestion>{suggestion}</suggestion>",
            output_format=ProposalResult,
        )
        proposal = prop_out.proposal
        suggestion = yield proposal  # Receive next suggestions from verifier


def verifier(query, proposer_coroutine, nj_llm):
    """Function that verifies proposed values."""
    proposal = next(proposer_coroutine)  # Start the coroutine and get initial proposal
    while True:
        ver_out: VerificationResult = nj_llm(
            "Does the proposed solution <proposal> satisfy the user query: <query>? Assign your "
            f"answer to `accepted`.\n<proposal>{proposal}</proposal>\n<query>{query}</query>",
            output_format=VerificationResult,
        )
        accepted = ver_out.accepted

        if accepted:
            break
        else:
            sug_out: SuggestionResult = nj_llm(
                "Come up with suggestions for improvement given the unaccepted <proposal> for "
                "<query> and store them to `suggestions`.\n"
                f"<proposal{proposal}</proposal>\n<query>{query}</query>",
                output_format=SuggestionResult,
            )
            suggestions = sug_out.suggestions
            proposal = proposer_coroutine.send(suggestions)

    return proposal


def main(riddle: str, nj_llm):
    proposer_coroutine = proposer(riddle, nj_llm)
    return verifier(riddle, proposer_coroutine, nj_llm)


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    riddle = "I am the beginning of the end, and the end of time and space. I am essential to creation, and I surround every place. What am I?"

    outputs = {}
    errors = {}
    hard_results = {
        "proposal_is_string": False,
        "proposal_is_non_empty": False,
    }

    try:
        outputs["test_0"] = main(riddle, nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
