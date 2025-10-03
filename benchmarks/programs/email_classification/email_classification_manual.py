from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Category(Enum):
    PRESALE_QUESTION = "Pre-sale question"
    DEFECTIVE_ITEM = "Defective item"
    BILLING_QUESTION = "Billing question"


class OtherCategoryException(Exception):
    category: str

    def __init__(self, category: str):
        super().__init__()
        self.category = category

    def __str__(self):
        return f"OtherCategoryException({self.category})"


class Classified(BaseModel):
    type: Literal["classified"]
    category: Category


class Other(BaseModel):
    type: Literal["other"]
    new_category: str


class EmailClassificationLLMResult(BaseModel):
    result: Classified | Other


def email_classification(email: str, nj_llm):
    categories_str = ", ".join([c.value for c in Category])
    prompt = (
        "Classify the <email> as one of the options listed in <categories>.\n"
        "If none of the options fit well, respond with type='other' and include a concise suggested category name in new_category.\n"
        f"<categories>{categories_str}</categories>\n"
        f"<email>{email}</email>"
    )

    result: EmailClassificationLLMResult = nj_llm(
        prompt,
        output_format=EmailClassificationLLMResult,
    )

    if isinstance(result.result, Classified):
        return result.result.category
    elif isinstance(result.result, Other):
        raise OtherCategoryException(result.result.new_category)
    else:
        raise OtherCategoryException("Unknown")


def main(email: str, nj_llm):
    try:
        return email_classification(email, nj_llm)
    except OtherCategoryException as e:
        return f"Other category found: {e.category}"


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    emails = [
        "I received my order today with a cracked screen and need a replacement as soon as possible.",
        "Thank you for the helpful customer support during my recent service call.",
        "I'm interested in your product and would like to know if it comes with a two-year warranty and is available in different colors.",
        "I noticed an unexpected charge on my invoice and would appreciate clarification regarding this billing discrepancy.",
    ]
    expected_outputs = [
        Category.DEFECTIVE_ITEM,
        None,
        Category.PRESALE_QUESTION,
        Category.BILLING_QUESTION,
    ]
    outputs = {}
    errors = {}
    hard_results = {}

    for i, (email, expected) in enumerate(zip(emails, expected_outputs)):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}_correct_type"] = False
        hard_results[f"test_{i}_correct_result"] = False

        try:
            outputs[f"test_{i}"] = main(email, nj_llm)
        except Exception as e:
            errors[f"test_{i}"] = e
        else:
            try:
                # do hard eval
                if expected is not None:
                    # Check if result is a Category instance or an exception message.
                    hard_results[f"test_{i}_correct_type"] = isinstance(outputs[f"test_{i}"], Category)

                    # If result is a Category, ensure it's one of the valid enum values.
                    if hard_results[f"test_{i}_correct_type"]:
                        hard_results[f"test_{i}_correct_result"] = outputs[f"test_{i}"] == expected
                else:
                    hard_results[f"test_{i}_correct_type"] = isinstance(outputs[f"test_{i}"], str)
                    hard_results[f"test_{i}_correct_result"] = outputs[f"test_{i}"].startswith("Other category found:")

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
