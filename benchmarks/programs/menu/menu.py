from typing import Any, Dict, List, Tuple

import nightjarpy as nj


@nj.fn
def main(inventory: Dict[str, int], orders: List[str]):

    for order in orders:
        """natural
        Based on customer's <order>, subtract the sold quantity from the <inventory>.
        """

    return inventory


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    inventory = {
        "sushi": 10,
        "ramen": 5,
        "sashimi": 7,
        "salad": 3,
    }
    orders = [
        "Hi, I'd like to order one sushi set",
        "Hi, I’d like to order three mixed sashimis for pickup, please.",
        "can I get two ramens please, and no salad with them?",
    ]

    outputs = {}
    errors = {}
    hard_results = {
        "test_sushi": False,
        "test_ramen": False,
        "test_sashimi": False,
        "test_salad": False,
    }
    expected_inventory = {
        "sushi": 9,
        "ramen": 3,
        "sashimi": 4,
        "salad": 3,
    }

    try:
        result = main(inventory, orders)
    except Exception as e:
        errors["test_0"] = e
    else:
        outputs["test_0"] = result
        for item in inventory:
            try:
                hard_results[f"test_{item}"] = result[item] == expected_inventory[item]
            except Exception as e:
                errors[f"test_{item}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
