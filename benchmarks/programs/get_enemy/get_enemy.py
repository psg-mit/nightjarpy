import nightjarpy as nj


@nj.fn
def main():
    """natural
    Save to a variable an object named enemy representing an enemy that the player can fight (can be any creature or man. Be creative). The enemy should have:
    - A `name` attribute (a string).
    - A `description` attribute (a string).
    Create a <:get_enemy> function that returns the created enemy object.
    """
    return get_enemy


#### Tests ####
from typing import Any, Dict, List, Tuple


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    hard_results = {
        "enemy_not_predefined": False,
        "enemy_exists": False,
        "enemy_has_name": False,
        "enemy_name_is_str": False,
        "enemy_has_description": False,
        "enemy_description_is_str": False,
    }

    try:
        get_enemy = main()
        outputs["test_0"] = get_enemy
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            _ = enemy
        except NameError:
            enemy_not_predefined = True
        else:
            enemy_not_predefined = False

        try:
            enemy_obj = get_enemy()
        except Exception as e:
            errors["test_0"] = e
        else:
            try:
                hard_results["enemy_not_predefined"] = enemy_not_predefined
                hard_results["enemy_exists"] = enemy_obj is not None
                hard_results["enemy_has_name"] = hasattr(enemy_obj, "name")
                if hard_results["enemy_has_name"]:
                    hard_results["enemy_name_is_str"] = isinstance(enemy_obj.name, str)
                hard_results["enemy_has_description"] = hasattr(enemy_obj, "description")
                if hard_results["enemy_has_description"]:
                    hard_results["enemy_description_is_str"] = isinstance(enemy_obj.description, str)
            except Exception as e:
                errors["test_0"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
