import nightjarpy as nj


class Player:
    def __init__(self):
        self.name = "Knight Giar"
        self.health = 100
        self.attack_power = 10

    def __str__(self):
        return f"Player(name={self.name}, health={self.health}, attack_power={self.attack_power})"

    def __eq__(self, other):
        return self.name == other.name and self.health == other.health and self.attack_power == other.attack_power


class Enemy:
    def __init__(self, name: str, health: int, attack_power: int):
        self.name = name
        self.health = health
        self.attack_power = attack_power

    def __str__(self):
        return f"Enemy(name={self.name}, health={self.health}, attack_power={self.attack_power})"

    def __eq__(self, other):
        return self.name == other.name and self.health == other.health and self.attack_power == other.attack_power


@nj.fn
def main(enemies, player_action):
    player = Player()

    """natural
    Update each entity in <enemies> based on the <player_action> and <player> stats.
    """

    return player, enemies


#### Tests ####
from typing import Any, Dict, List, Tuple


def run(nj_llm=None) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    enemies = [Enemy("Lira the Swift", 25, 4), Enemy("Grobnar the Brutish", 40, 6), Enemy("Varnok the Cruel", 60, 8)]
    action = "I rush at Lira and swing my sword, trying to cut her off before she can dodge. Then I ready myself for Grobnar's attack."

    outputs = {}
    errors = {}
    hard_results = {
        "test_0": False,
        "test_1": False,
        "test_2": False,
    }

    try:
        outputs["test"] = main(enemies, action)
    except Exception as e:
        errors["test"] = e
    else:
        expected_output = [
            Enemy("Lira the Swift", 15, 4),
            Enemy("Grobnar the Brutish", 40, 6),
            Enemy("Varnok the Cruel", 60, 8),
        ]
        try:
            for i, (expected_enemy, output_enemy) in enumerate(zip(expected_output, outputs["test"][1])):
                hard_results[f"test_{i}"] = False

                try:
                    hard_results[f"test_{i}"] = expected_enemy == output_enemy
                except Exception as e:
                    errors[f"test_{i}"] = e

        except Exception as e:
            errors[f"test"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":

    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
