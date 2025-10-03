from enum import Enum
from typing import Any, Dict, Tuple

import nightjarpy as nj


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


@nj.fn
def email_classification(email: str):
    """natural
    Classify <email> as one of the options in <Category> enum and save it as <:answer> as a <Category> type.

    Raise <OtherCategoryException> with the alternative category name if the email does not fit any of the categories.
    """
    return answer


def main(email: str):
    try:
        return email_classification(email)
    except OtherCategoryException as e:
        return f"Other category found: {e.category}"


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
            outputs[f"test_{i}"] = main(email)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
