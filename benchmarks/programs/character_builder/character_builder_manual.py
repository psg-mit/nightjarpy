from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Character:
    def __init__(self, name: str, traits: List[str]):
        self.name = name
        self.traits = traits

    def __repr__(self) -> str:
        return f"Character(name={self.name}, traits={self.traits})"


ALLOWED_NAMES: List[str] = [
    "Bruce Wayne",
    "Aixin-Jueluo Xuanye",
    "Ivan Petrov",
]

ALLOWED_TRAITS: List[str] = [
    "Diplomatic",
    "Adventurous",
    "Brave",
    "Short-lived",
    "Lazy",
    "Illiterate",
]

NameLiteral = Literal["Bruce Wayne", "Aixin-Jueluo Xuanye", "Ivan Petrov"]
TraitLiteral = Literal[
    "Diplomatic",
    "Adventurous",
    "Brave",
    "Short-lived",
    "Lazy",
    "Illiterate",
]


class CharacterLLMResult(BaseModel):
    name: NameLiteral = Field(..., description="Chosen character name from the allowed list")
    traits: List[TraitLiteral] = Field(
        description="Exactly 3 distinct traits chosen from the allowed list",
    )


def main(setting: str, nj_llm) -> Character:
    result: CharacterLLMResult = nj_llm(
        "Create a Character that is a good fit for the given setting.\n\n"
        f"Setting: {setting}\n\n"
        "Choose the name from the following options exactly as written:\n"
        "- Bruce Wayne\n- Aixin-Jueluo Xuanye\n- Ivan Petrov\n\n"
        "Choose 3 distinct traits that fit the setting from the following options, using exact spelling:\n"
        "- Diplomatic\n- Adventurous\n- Brave\n- Short-lived\n- Lazy\n- Illiterate\n\n"
        "Return only the structured object matching the output schema.",
        output_format=CharacterLLMResult,
    )

    distinct_traits: List[str] = []
    for trait in result.traits:
        if trait not in distinct_traits:
            distinct_traits.append(trait)
    if len(distinct_traits) < 3:
        for trait in ALLOWED_TRAITS:
            if trait not in distinct_traits:
                distinct_traits.append(trait)
            if len(distinct_traits) == 3:
                break
    distinct_traits = distinct_traits[:3]

    chosen_name = result.name if result.name in ALLOWED_NAMES else ALLOWED_NAMES[0]

    return Character(name=chosen_name, traits=distinct_traits)


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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
        outputs["test_1"] = main(setting, nj_llm)
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
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
