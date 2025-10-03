from typing import Any, Dict, List, Tuple

import nightjarpy as nj


class Book:
    def __init__(self, title: str, author: str, description: str):
        self.title = title
        self.author = author
        self.description = description

    def __str__(self):
        return f"Book(title={self.title}, author={self.author}, description={self.description})"


@nj.fn
def main(catalog: List[Book], request: str):
    """natural
    save to <:recommended_book> the <Book> that fulfills the customer's <request> and remove the book from <catalog>
    """

    return catalog, recommended_book


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        outputs["test"] = main(books, request)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
