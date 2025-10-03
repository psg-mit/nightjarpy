from typing import Any, Dict, List, Tuple

import nightjarpy as nj


class Character:
    def __init__(self, name: str, traits: List[str]):
        self.name = name
        self.traits = traits

    def __repr__(self) -> str:
        return f"Character(name={self.name}, traits={self.traits})"


@nj.fn
def main(setting: str):
    """natural
    The setting is <setting>.

    Create a <Character> object that is a good fit for the <setting>.
    Choose the name of the character from the following list:
    - Bruce Wayne
    - Aixin-Jueluo Xuanye
    - Ivan Petrov

    Choose 3 distinct traits that are a good fit for the <setting> from the following list:
    - Diplomatic
    - Adventurous
    - Brave
    - Short-lived
    - Lazy
    - Illiterate

    Save as <:character>
    """

    return character


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    setting = "18th century China"
    outputs = {
        "test_1": None,
    }
    errors = {
        "test_1": None,
        "test_name": None,
        "test_trait_diplomatic": None,
        "test_trait_adventurous": None,
        "test_trait_brave": None,
    }
    hard_results = {
        "test_name": False,
        "test_trait_diplomatic": False,
        "test_trait_adventurous": False,
        "test_trait_brave": False,
    }
    try:
        outputs["test_1"] = main(setting)
    except Exception as e:
        errors["test_1"] = e
    else:
        try:
            hard_results["test_name"] = outputs["test_1"].name == "Aixin-Jueluo Xuanye"
        except Exception as e:
            errors["test_character"] = e
        else:
            try:
                hard_results["test_trait_diplomatic"] = "Diplomatic" in outputs["test_1"].traits
            except Exception as e:
                errors["test_trait_diplomatic"] = e
            else:
                try:
                    hard_results["test_trait_adventurous"] = "Adventurous" in outputs["test_1"].traits
                except Exception as e:
                    errors["test_trait_adventurous"] = e
                else:
                    try:
                        hard_results["test_trait_brave"] = "Brave" in outputs["test_1"].traits
                    except Exception as e:
                        errors["test_trait_brave"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
