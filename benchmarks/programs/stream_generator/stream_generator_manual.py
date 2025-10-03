from typing import Any, Dict, Generator, List, Tuple

from pydantic import BaseModel

from nightjarpy import nj_llm_factory


class NextItemResult(BaseModel):
    item: str


def _generate_stream_items(topic: str, nj_llm) -> Generator[str, None, None]:
    seen_items: List[str] = []
    while True:
        result: NextItemResult = nj_llm(
            "Given the <topic> and the list of <seen_items> that were already yielded, "
            "produce exactly one new, unique, brief item related to the topic and save it to "
            "`item`. It must not duplicate any entry in <seen_items>.\n"
            f"<topic>{topic}</topic>\n<seen_items>{seen_items}</seen_items>",
            output_format=NextItemResult,
        )

        next_item: str = result.item.strip()
        if not next_item or next_item in seen_items:
            continue

        seen_items.append(next_item)
        yield next_item


def main(topic: str, nj_llm) -> Generator[str, None, None]:
    gen = _generate_stream_items(topic, nj_llm)
    return gen


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    hard_results = {
        "uspres_1": False,
        "uspres_2": False,
        "uspres_3": False,
        "p_and_p_1": False,
        "p_and_p_2": False,
        "p_and_p_3": False,
    }

    us_presidents = [
        "George Washington",
        "John Adams",
        "Thomas Jefferson",
        "James Madison",
        "James Monroe",
        "John Quincy Adams",
        "Andrew Jackson",
        "Martin Van Buren",
        "William Henry Harrison",
        "John Tyler",
        "James K. Polk",
        "Zachary Taylor",
        "Millard Fillmore",
        "Franklin Pierce",
        "James Buchanan",
        "Abraham Lincoln",
        "Andrew Johnson",
        "Ulysses S. Grant",
        "Rutherford B. Hayes",
        "James A. Garfield",
        "Chester A. Arthur",
        "Grover Cleveland",
        "Benjamin Harrison",
        "Grover Cleveland",  # (Second, non-consecutive term)
        "William McKinley",
        "Theodore Roosevelt",
        "William Howard Taft",
        "Woodrow Wilson",
        "Warren G. Harding",
        "Calvin Coolidge",
        "Herbert Hoover",
        "Franklin D. Roosevelt",
        "Harry S. Truman",
        "Dwight D. Eisenhower",
        "John F. Kennedy",
        "Lyndon B. Johnson",
        "Richard Nixon",
        "Gerald Ford",
        "Jimmy Carter",
        "Ronald Reagan",
        "George H. W. Bush",
        "Bill Clinton",
        "George W. Bush",
        "Barack Obama",
        "Donald Trump",
        "Joe Biden",
    ]

    try:
        prompt = "US Presidents"
        outputs["test_0"] = main(prompt, nj_llm)
    except Exception as e:
        errors["test_0"] = e
    else:
        if outputs["test_0"] is not None:
            for i in range(3):
                try:
                    res = next(outputs["test_0"])
                    hard_results[f"uspres_{i+1}"] = res in us_presidents
                    outputs[f"test_0_item_{i+1}"] = res
                except Exception as e:
                    hard_results[f"uspres_{i+1}"] = False
                    errors[f"uspres_{i+1}"] = e
                    outputs[f"test_0_item_{i+1}"] = None

    pride_and_prejudice_characters = [
        "Elizabeth Bennet",
        "Fitzwilliam Darcy",
        "Mr. Darcy",
        "Jane Bennet",
        "Charles Bingley",
        "Mr. Bennet",
        "Mrs. Bennet",
        "Lydia Bennet",
        "Mary Bennet",
        "Catherine (Kitty) Bennet",
        "George Wickham",
        "William Collins",
        "Charlotte Lucas",
        "Lady Catherine de Bourgh",
        "Georgiana Darcy",
        "Caroline Bingley",
        "Louisa Hurst",
        "Mr. Hurst",
        "Colonel Fitzwilliam",
        "Sir William Lucas",
        "Maria Lucas",
        "Mr. Gardiner",
        "Mrs. Gardiner",
        "Mr. Phillips",
        "Mrs. Phillips",
        "Anne de Bourgh",
    ]

    try:
        prompt = "Pride and Prejudice Characters"
        outputs["test_1"] = main(prompt, nj_llm)
    except Exception as e:
        errors["test_1"] = e
    else:
        if outputs["test_1"] is not None:
            for i in range(3):
                try:
                    res = next(outputs["test_1"])
                    hard_results[f"p_and_p_{i+1}"] = res in pride_and_prejudice_characters
                    outputs[f"test_1_item_{i+1}"] = res
                except Exception as e:
                    hard_results[f"p_and_p_{i+1}"] = False
                    errors[f"p_and_p_{i+1}"] = e
                    outputs[f"test_1_item_{i+1}"] = None
    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
