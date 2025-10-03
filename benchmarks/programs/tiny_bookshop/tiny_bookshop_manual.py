from typing import Any, Dict, List, Tuple

from nightjarpy import nj_llm_factory


class Book:
    def __init__(self, title: str, author: str, description: str):
        self.title = title
        self.author = author
        self.description = description

    def __str__(self):
        return f"Book(title={self.title}, author={self.author}, description={self.description})"


def main(catalog: List[Book], request: str, nj_llm):
    prompt = (
        "Given the following list of books:\n"
        + "\n".join([f"{i}: {book.title} by {book.author}. {book.description}" for i, book in enumerate(catalog)])
        + f"\n\nCustomer request: {request}\n"
        "Which book (by index) best fulfills the customer's request? Reply with the index only."
    )
    response = nj_llm(prompt)
    try:
        idx = int(response.strip())
        recommended_book = catalog.pop(idx)
    except Exception:
        recommended_book = catalog.pop(0) if catalog else None

    return catalog, recommended_book


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    books_data = [
        (
            "Brave New World",
            "Aldous Huxley",
            "In a world where people are engineered for their roles, Bernard Marx begins to question the cost of a perfectly ordered society.",
        ),
        (
            "1984",
            "George Orwell",
            "Winston Smith risks everything as he secretly defies a totalitarian regime that watches every move and controls every thought.",
        ),
        (
            "Kiki's Delivery Service",
            "Eiko Kadono",
            "A spirited young witch needs to learn how to stand on her own two feet, as she arrives in a bustling and charming seaside town.",
        ),
    ]
    books = [Book(title, author, description) for title, author, description in books_data]
    request = "Did you get anything new in? I want a book with a coming-of-age plotline. I don't really like scary stuff, though."

    outputs = {}
    errors = {}
    hard_results = {
        "test_0": False,
        "test_1": False,
        "test_2": False,
    }

    try:
        outputs["test"] = main(books, request, nj_llm)
        updated_catalog, recommended_book = outputs["test"]
        catalog_titles = [book.title for book in updated_catalog]
    except Exception as e:
        errors["test"] = e
    else:
        try:
            hard_results["test_0"] = recommended_book.title == "Kiki's Delivery Service"
        except Exception as e:
            errors[f"test_0"] = e

        try:
            hard_results["test_1"] = "Kiki's Delivery Service" not in catalog_titles
        except Exception as e:
            errors[f"test_1"] = e

        try:
            hard_results["test_2"] = "1984" in catalog_titles and "Brave New World" in catalog_titles
        except Exception as e:
            errors[f"test_2"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
