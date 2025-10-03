from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Store(ABC):
    def __init__(self, name: str, city: str):
        self.name = name
        self.city = city

    def get_description(self):
        return f"{self.name}, located in {self.city}"

    @abstractmethod
    def get_inventory(self) -> List[str]: ...


# Structured output schema
class StoreSpec(BaseModel):
    """Specification of a store extracted from prompt."""

    subclass_name: str = Field(description="The name of the Store subclass to define.")
    name: str = Field(description="The name of the store instance.")
    city: str = Field(description="The city where the store is located.")
    inventory: List[str] = Field(description="The inventory items for this store.")


class StoreListResult(BaseModel):
    stores: List[StoreSpec] = Field(description="List of store specifications extracted from the prompt.")


def main(prompt: str, nj_llm):
    result: StoreListResult = nj_llm(
        "Read the <prompt> and extract what kinds of `Store`s are described. "
        "For each identified store type: \n"
        "- Create a subclass of `Store` with a custom `get_inventory` method\n"
        "- Instantiate a store of the subclass with appropriate name and city based on the prompt\n"
        "- Store the store instances in a list called `stores`\n"
        f"<prompt>{prompt}</prompt>",
        output_format=StoreListResult,
    )

    stores: List[Store] = []

    # Dynamically build subclasses and instantiate them
    for spec in result.stores:
        subclass = type(
            spec.subclass_name,
            (Store,),
            {"get_inventory": lambda self, inv=spec.inventory: inv},
        )
        store_instance = subclass(spec.name, spec.city)
        stores.append(store_instance)

    return stores


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    prompt = """i want a steampunk tokyo store for jewelry with the name `Kogane Clockwork`. it should sell gearlock pendants, kintsugi circuit cuffs, chrono-katana necklaces.

I also want another store called `Capybara Stardust Cafe` that sells star sprinkle mochi, capybara cloud latte, and interactive capybara feeding experiences. it's in Coconut Dog City.
"""
    outputs = {}
    errors = {}
    hard_results = {
        "stores_type": False,
        "stores_count": False,
        "stores_instance_type_1": False,
        "stores_instance_type_2": False,
        "stores_instance_name_1": False,
        "stores_instance_city_1": False,
        "stores_instance_name_2": False,
        "stores_instance_city_2": False,
        "stores_instance_inventory_1": False,
        "stores_instance_inventory_2": False,
    }

    try:
        outputs["test_0"] = main(prompt, nj_llm)
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            hard_results["stores_type"] = isinstance(outputs["test_0"], list)
            hard_results["stores_count"] = len(outputs["test_0"]) == 2
            if hard_results["stores_type"] and len(outputs["test_0"]) > 0:
                if outputs["test_0"][0].name.lower() == "kogane clockwork":
                    kogane_clockwork = outputs["test_0"][0]
                    capybara_stardust_cafe = outputs["test_0"][1]
                else:
                    kogane_clockwork = outputs["test_0"][1]
                    capybara_stardust_cafe = outputs["test_0"][0]
                hard_results["stores_instance_type_1"] = isinstance(kogane_clockwork, Store)
                hard_results["stores_instance_type_2"] = isinstance(capybara_stardust_cafe, Store)
                hard_results["stores_instance_name_1"] = kogane_clockwork.name.lower() == "kogane clockwork"
                hard_results["stores_instance_city_1"] = kogane_clockwork.city.lower() == "tokyo"
                hard_results["stores_instance_name_2"] = capybara_stardust_cafe.name.lower() == "capybara stardust cafe"
                hard_results["stores_instance_city_2"] = capybara_stardust_cafe.city.lower() == "coconut dog city"
                for inventory in kogane_clockwork.get_inventory():
                    if inventory.lower() not in [
                        "gearlock pendants",
                        "kintsugi circuit cuffs",
                        "chrono-katana necklaces",
                    ]:
                        hard_results["stores_instance_inventory_1"] = False
                        break
                hard_results["stores_instance_inventory_1"] = True
                for inventory in capybara_stardust_cafe.get_inventory():
                    if inventory.lower() not in [
                        "star sprinkle mochi",
                        "capybara cloud latte",
                        "interactive capybara feeding experiences",
                    ]:
                        hard_results["stores_instance_inventory_2"] = False
                        break
                hard_results["stores_instance_inventory_2"] = True
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
