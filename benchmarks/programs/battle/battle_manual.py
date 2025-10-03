from typing import Any, Dict, List, Tuple

from pydantic import BaseModel


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


class Response(BaseModel):
    attacked_enemies: list[str]


def main(enemies, player_action, nj_llm):
    player = Player()

    res = nj_llm(
        f"Return a list of enemy names that the player attacked (if any): {player_action}",
        output_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"attacked_enemies": {"type": "array", "items": {"type": "string"}}},
                    "required": ["attacked_enemies"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    for enemy_name in res["attacked_enemies"]:
        for enemy in enemies:
            if enemy.name == enemy_name:
                enemy.health -= player.attack_power

    return player, enemies


#### Tests ####
import logging

from nightjarpy import nj_llm_factory


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    enemies = [
        Enemy("Lira the Swift", 25, 4),
        Enemy("Grobnar the Brutish", 40, 6),
        Enemy("Varnok the Cruel", 60, 8),
    ]

    action = "I rush at Lira and swing my sword, trying to cut her off before she can dodge. Then I ready myself for Grobnar's attack."

    outputs = {}
    errors = {}
    hard_results = {
        "test_0": False,
        "test_1": False,
        "test_2": False,
    }

    try:
        outputs["test"] = main(enemies, action, nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm=nj_llm)
    print(results)
    print(hard_results)
    print(errors)
