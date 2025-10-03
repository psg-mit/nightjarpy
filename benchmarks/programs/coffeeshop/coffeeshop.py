from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

import nightjarpy as nj


class ListOrderIndependentMixin:
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        for field in getattr(self, "model_fields", {}):
            self_value = getattr(self, field)
            other_value = getattr(other, field)
            if isinstance(self_value, list):
                if not isinstance(other_value, list):
                    return False
                if len(self_value) != len(other_value):
                    return False
                # Convert lists to sets for order-independent comparison
                # We need to convert items to tuples to make them hashable
                self_set = {
                    tuple(item.model_dump().items()) if hasattr(item, "model_dump") else item for item in self_value
                }
                other_set = {
                    tuple(item.model_dump().items()) if hasattr(item, "model_dump") else item for item in other_value
                }
                if self_set != other_set:
                    return False
            elif self_value != other_value:
                return False
        return True


# Basic types first
CoffeeTemperature = Literal["hot", "extra hot", "warm", "iced"]
CoffeeSize = Literal["short", "tall", "grande", "venti"]
EspressoSize = Literal["solo", "doppio", "triple", "quad"]
OptionQuantity = Literal["no", "light", "regular", "extra"] | int


# Basic options and preparations
class BakeryPreparation(BaseModel):
    name: Literal["warmed", "cut in half"]


class BakeryOption(BaseModel):
    name: Literal["butter", "strawberry jam", "cream cheese"]
    optionQuantity: Optional[OptionQuantity]


class Caffeine(BaseModel):
    name: Literal["regular", "two thirds caf", "half caf", "one third caf", "decaf"]


class Milk(BaseModel):
    name: Literal[
        "whole milk", "two percent milk", "nonfat milk", "coconut milk", "soy milk", "almond milk", "oat milk"
    ]


class Creamer(BaseModel):
    name: Literal[
        "whole milk creamer",
        "two percent milk creamer",
        "one percent milk creamer",
        "nonfat milk creamer",
        "coconut milk creamer",
        "soy milk creamer",
        "almond milk creamer",
        "oat milk creamer",
        "half and half",
        "heavy cream",
    ]


class Topping(BaseModel):
    name: Literal["cinnamon", "foam", "ice", "nutmeg", "whipped cream", "water"]
    optionQuantity: Optional[OptionQuantity]


class LattePreparation(BaseModel):
    name: Literal["for here cup", "lid", "with room", "to go", "dry", "wet"]


class Sweetener(BaseModel):
    name: Literal["equal", "honey", "splenda", "sugar", "sugar in the raw", "sweet n low", "espresso shot"]
    optionQuantity: Optional[OptionQuantity]


class Syrup(BaseModel):
    name: Literal[
        "almond syrup",
        "buttered rum syrup",
        "caramel syrup",
        "cinnamon syrup",
        "hazelnut syrup",
        "orange syrup",
        "peppermint syrup",
        "raspberry syrup",
        "toffee syrup",
        "vanilla syrup",
    ]
    optionQuantity: Optional[OptionQuantity]


# Type aliases for options
CaffeineOptions = Caffeine | Milk | Creamer
LatteOptions = CaffeineOptions | Topping | LattePreparation | Sweetener


# Main product classes
class BakeryProduct(BaseModel, ListOrderIndependentMixin):
    name: Literal["apple bran muffin", "blueberry muffin", "lemon poppyseed muffin", "bagel"]
    options: List[BakeryOption | BakeryPreparation]


class LatteDrink(BaseModel, ListOrderIndependentMixin):
    name: Literal["cappuccino", "flat white", "latte", "latte macchiato", "mocha", "chai latte"]
    temperature: Optional[CoffeeTemperature]
    size: Optional[CoffeeSize] = "grande"
    options: List[Creamer | Sweetener | Syrup | Topping | Caffeine | LattePreparation] = []


class EspressoDrink(BaseModel, ListOrderIndependentMixin):
    name: Literal["espresso", "lungo", "ristretto", "macchiato"]
    temperature: Optional[CoffeeTemperature]
    size: Optional[EspressoSize] = "doppio"
    options: List[Creamer | Sweetener | Syrup | Topping | Caffeine | LattePreparation] = []


class CoffeeDrink(BaseModel, ListOrderIndependentMixin):
    name: Literal["americano", "coffee"]
    temperature: Optional[CoffeeTemperature]
    size: Optional[CoffeeSize] = "grande"
    options: List[Creamer | Sweetener | Syrup | Topping | Caffeine | LattePreparation] = []


# Product type union
Product = BakeryProduct | LatteDrink | EspressoDrink | CoffeeDrink


class LineItem(BaseModel):
    product: Product
    quantity: int


class Cart(BaseModel, ListOrderIndependentMixin):
    items: List[LineItem]
    # coffee: str


def option_match_report(option1, option2, prefix="option") -> Dict[str, bool]:
    report = {}

    if type(option1) != type(option2):
        report[f"{prefix}_type_matches"] = False
        return report

    report[f"{prefix}_type_matches"] = True
    report[f"{prefix}_name_matches"] = option1.name == option2.name

    if hasattr(option1, "optionQuantity") and hasattr(option2, "optionQuantity"):
        report[f"{prefix}_quantity_matches"] = option1.optionQuantity == option2.optionQuantity

    return report


def product_match_report(product1, product2, prefix="product") -> Dict[str, bool]:
    report = {}

    report[f"{prefix}_type_matches"] = type(product1) == type(product2)
    report[f"{prefix}_name_matches"] = product1.name == product2.name

    if hasattr(product1, "size") and hasattr(product2, "size"):
        report[f"{prefix}_size_matches"] = product1.size == product2.size

    if hasattr(product1, "temperature") and hasattr(product2, "temperature"):
        report[f"{prefix}_temperature_matches"] = product1.temperature == product2.temperature

    # Options matching (by order of appearance)
    max_len = max(len(product1.options), len(product2.options))
    for i in range(max_len):
        opt_prefix = f"{prefix}_option_{i}"
        if i < len(product1.options) and i < len(product2.options):
            opt1 = product1.options[i]
            opt2 = product2.options[i]
            opt_report = option_match_report(opt1, opt2, prefix=opt_prefix)
            report.update(opt_report)
        elif i < len(product1.options):
            opt1 = product1.options[i]
            opt_report = option_match_report(opt1, opt1, prefix=opt_prefix)
            report.update({k: False for k in opt_report})
        elif i < len(product2.options):
            opt2 = product2.options[i]
            opt_report = option_match_report(opt2, opt2, prefix=opt_prefix)
            report.update({k: False for k in opt_report})

    return report


def line_item_match_report(item1, item2, prefix="line_item") -> Dict[str, bool]:
    report = {}

    prod_prefix = f"{prefix}_product"
    prod_report = product_match_report(item1.product, item2.product, prefix=prod_prefix)
    report.update(prod_report)

    report[f"{prefix}_quantity_matches"] = item1.quantity == item2.quantity

    return report


def cart_match_report(cart1: Cart, cart2: Cart, prefix="cart") -> Dict[str, bool]:
    report = {}

    max_len = max(len(cart1.items), len(cart2.items))
    for i in range(max_len):
        item_prefix = f"{prefix}_line_item_{i}"
        if i < len(cart1.items) and i < len(cart2.items):
            item_report = line_item_match_report(cart1.items[i], cart2.items[i], prefix=item_prefix)
            report.update(item_report)
        elif i < len(cart1.items):
            item_report = line_item_match_report(cart1.items[i], cart1.items[i], prefix=item_prefix)
            report.update({k: False for k in item_report})
        elif i < len(cart2.items):
            item_report = line_item_match_report(cart2.items[i], cart2.items[i], prefix=item_prefix)
            report.update({k: False for k in item_report})

    return report


@nj.fn
def main(coffee_order: str):
    """natural
    Extract the items the customer wants to order from the <coffee_order> as a <Cart> and save it as <:cart>. Only parse English orders.

    If any item isn't extractable into the <Cart>, raise a <ValueError> with the error message "I did not understand the following" and all the items that weren't understood.
    """
    return cart


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    hard_results = {}
    test = [
        (
            "i'd like a latte that's it",
            Cart(items=[LineItem(product=LatteDrink(name="latte", temperature=None, options=[]), quantity=1)]),
        ),
        (
            "i'd like a tall decaf latte iced a grande cappuccino double espresso and a warmed poppyseed muffin sliced in half",
            Cart(
                items=[
                    LineItem(
                        product=LatteDrink(
                            name="latte",
                            temperature="iced",
                            size="tall",
                            options=[Caffeine(name="decaf")],
                        ),
                        quantity=1,
                    ),
                    LineItem(
                        product=LatteDrink(
                            name="cappuccino",
                            temperature=None,
                            size="grande",
                            options=[],
                        ),
                        quantity=1,
                    ),
                    LineItem(
                        product=EspressoDrink(
                            name="espresso",
                            temperature=None,
                            options=[],
                            size="doppio",
                        ),
                        quantity=1,
                    ),
                    LineItem(
                        product=BakeryProduct(
                            name="lemon poppyseed muffin",
                            options=[BakeryPreparation(name="warmed"), BakeryPreparation(name="cut in half")],
                        ),
                        quantity=1,
                    ),
                ]
            ),
        ),
        ("two lawnmowers, a grande latte and a tall tree", ["lawnmowers", "tree"]),
        ("un petit cafe", ["cafe"]),
    ]

    for i, (inp, expected) in enumerate(test):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        if isinstance(expected, Cart):
            hard_results.update(cart_match_report(Cart(items=[]), expected, prefix=f"test_{i}"))
        else:
            hard_results[f"test_{i}"] = False

        try:
            outputs[f"test_{i}"] = main(inp)
        except ValueError as e:
            try:
                outputs[f"test_{i}"] = e.args[0]
            except Exception as e:
                errors[f"test_{i}"] = e
        except Exception as e:
            errors[f"test_{i}"] = e

        if outputs[f"test_{i}"] is not None:
            try:
                if isinstance(expected, Cart):
                    hard_results.update(cart_match_report(outputs[f"test_{i}"], expected, prefix=f"test_{i}"))
                else:
                    pass_ = (
                        isinstance(outputs[f"test_{i}"], str)
                        and "I did not understand the following" in outputs[f"test_{i}"]
                    )
                    for item in expected:
                        pass_ = pass_ and item in outputs[f"test_{i}"]
                    hard_results[f"test_{i}"] = pass_
            except Exception as e:
                errors[f"test_{i}"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
