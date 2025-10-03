from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Player:
    def __init__(self):
        self.name = "Knight Capy"
        self.health = 100
        self.attack_power = 10


class Enemy:
    def __init__(
        self,
        name: str,
        description: str,
        health: int,
        attack_power: int,
        speak_line: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.health = health
        self.attack_power = attack_power
        self._speak_line = speak_line

    def speak(self) -> str:
        if self._speak_line:
            return self._speak_line
        return f"You dare challenge {self.name}? Prepare to face the {self.description}!"


class EnemySpec(BaseModel):
    name: str
    description: str
    health: int
    attack_power: int
    speak_line: Optional[str]


class CreateEnemy(BaseModel):
    type: Literal["create_enemy"]
    enemy: EnemySpec


class BindAttack(BaseModel):
    type: Literal["bind_attack"]
    target_player: Literal["player1"]


Command = Union[CreateEnemy, BindAttack]


class EnemyLLMResult(BaseModel):
    commands: List[Command] = Field(default_factory=list)


def main(nj_llm):
    player1 = Player()
    enemy: Optional[Enemy] = None

    result: EnemyLLMResult = nj_llm(
        "Create an enemy for the <player1> to fight and attach an attack method to the player."
        "Respond with structured commands. First, `CreateEnemy` with an enemy spec. Then, "
        "`BindAttack` to attach player1.attack that subtracts player1.attack_power from the "
        f"target enemy's health.\n<player1>name={player1.name}; health={player1.health}; "
        f"attack_power={player1.attack_power}</player1>",
        output_format=EnemyLLMResult,
    )

    for cmd in result.commands:
        if isinstance(cmd, CreateEnemy):
            spec = cmd.enemy
            enemy = Enemy(
                name=spec.name,
                description=spec.description,
                health=spec.health,
                attack_power=spec.attack_power,
                speak_line=spec.speak_line,
            )
        elif isinstance(cmd, BindAttack):

            def attack(self, target_enemy: Enemy):
                target_enemy.health -= self.attack_power
                return f"{self.name} attacks {target_enemy.name} for {self.attack_power} damage!"

            setattr(player1, "attack", attack.__get__(player1, Player))

    # Fallbacks if LLM not provided or did not bind attack/enemy
    if not hasattr(player1, "attack"):

        def attack(self, target_enemy: "Enemy"):
            target_enemy.health -= self.attack_power
            return f"{self.name} attacks {target_enemy.name} for {self.attack_power} damage!"

        setattr(player1, "attack", attack.__get__(player1, Player))

    if enemy is None:
        enemy = Enemy(
            name="Goblin",
            description="a mischievous cave dweller",
            health=50,
            attack_power=8,
        )

    return player1, enemy


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        player1, enemy = main(nj_llm)  # result is a tuple (player1, enemy)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
