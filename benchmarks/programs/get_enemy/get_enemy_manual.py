from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from nightjarpy import nj_llm_factory


class Enemy:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class EnemySpec(BaseModel):
    name: str
    description: str


class EnemyResult(BaseModel):
    enemy: EnemySpec


def main(nj_llm):
    enemy_obj: Optional[Enemy] = None

    try:
        result: EnemyResult = nj_llm(
            "Create an enemy object for a game. The enemy spec must include name (string) and description (string).",
            output_format=EnemyResult,
        )

        if result is not None and result.enemy is not None:
            enemy_obj = Enemy(
                name=result.enemy.name,
                description=result.enemy.description,
            )
    except Exception:
        pass

    if enemy_obj is None:
        enemy_obj = Enemy(
            name="Goblin Scout",
            description="a sneaky cavern dweller lurking in the shadows",
        )

    def get_enemy():
        return enemy_obj

    return get_enemy


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        get_enemy = main(nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
