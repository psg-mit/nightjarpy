from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class UpdateInventory(BaseModel):
    item: str
    quantity: int


class MenuLLMResult(BaseModel):
    updates: List[UpdateInventory]


def main(inventory: Dict[str, int], orders: List[str], nj_llm):
    result: MenuLLMResult = nj_llm(
        "Look at the current <inventory> and the incoming <orders>. For each entry in <orders>, "
        "create an `updates` list of objects with fields `item` and `quantity` where `quantity` is "
        "the amount to subtract from the inventory for that item. Include exactly one update per "
        f"order item.\n<inventory>{inventory}</inventory>\n<orders>{orders}</orders>",
        output_format=MenuLLMResult,
    )

    for update in result.updates:
        inventory[update.item] = inventory.get(update.item, 0) - update.quantity

    return inventory


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        result = main(inventory, orders, nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
