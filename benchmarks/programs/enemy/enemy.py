from typing import Any, Dict, List, Tuple

import nightjarpy as nj


class Player:
    def __init__(self):
        self.name = "Knight Capy"
        self.health = 100
        self.attack_power = 10


@nj.fn
def main():
    player1 = Player()

    """natural
    Create an object representing an enemy that the player can fight (can be any creature or man. Be creative). Store the object in <:enemy>. The enemy should have:
    - A <:enemy.name> attribute (a string).
    - A <:enemy.description> attribute (a string).
    - A <:enemy.health> attribute (an integer).
    - An <:enemy.attack_power> attribute (an integer).
    - An <:enemy.speak> method that returns a string.
    Then, give <player1> the <:player1.attack> method that attacks an enemy, given by the argument.
    """
    return player1, enemy


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    outputs = {}
    errors = {}
    hard_results = {
        "enemy_not_none": False,
        "enemy_name_str": False,
        "enemy_description_str": False,
        "enemy_health_int": False,
        "enemy_attack_power_int": False,
        "enemy_speak_method": False,
        "player1_attack_method": False,
        "enemy_before_health_gt_enemy_after_health": False,
        "speak_output_not_none": False,
    }
    try:
        player1, enemy = main()  # result is a tuple (player1, enemy)
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            enemy_before = {
                "name": enemy.name,
                "health": enemy.health,
                "attack_power": enemy.attack_power,
            }
        except Exception as e:
            errors["test_0"] = e
        else:
            try:
                player1.attack(enemy)
            except Exception as e:
                errors["test_0"] = e
            else:
                try:
                    speak_output = enemy.speak()
                except Exception as e:
                    errors["test_0"] = e
                else:
                    outputs["test_0"] = (player1, enemy_before, enemy, speak_output)

                    # hard eval
                    try:
                        hard_results["enemy_not_none"] = enemy is not None
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_name_str"] = isinstance(enemy.name, str)
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_description_str"] = isinstance(enemy.description, str)
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_health_int"] = isinstance(enemy.health, int)
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_attack_power_int"] = isinstance(enemy.attack_power, int)
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_speak_method"] = callable(getattr(enemy, "speak", None))
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["player1_attack_method"] = callable(getattr(player1, "attack", None))
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["enemy_before_health_gt_enemy_after_health"] = (
                            enemy_before["health"] > enemy.health
                        )
                    except Exception as e:
                        errors["test_0"] = e

                    try:
                        hard_results["speak_output_not_none"] = speak_output is not None
                    except Exception as e:
                        errors["test_0"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
