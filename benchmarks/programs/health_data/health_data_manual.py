from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Quantity(BaseModel):
    value: float
    units: str

    def __str__(self):
        return f"Quantity(Value: {self.value}, Units: {self.units})"


class Medication(BaseModel):
    name: str
    dose: Quantity
    frequency: Quantity
    strength: Quantity

    def __str__(self):
        return (
            f"Medication(Name: {self.name}, Dose: {self.dose}, Frequency: {self.frequency}, Strength: {self.strength})"
        )


class Condition(BaseModel):
    name: str
    startDate: datetime
    status: Literal[
        "active",
        "recurrence",
        "relapse",
        "inactive",
        "remission",
        "resolved",
        "unknown",
    ]
    endDate: Optional[datetime]

    def __str__(self):
        return f"Condition(Name: {self.name}, Start Date: {self.startDate}, End Date: {self.endDate}, Status: {self.status})"


class OtherHealthData(BaseModel):
    text: str
    when: Optional[datetime]

    def __str__(self):
        return f"OtherHealthData(Text: {self.text}, When: {self.when})"


class HealthData(BaseModel):
    medication: List[Medication]
    condition: List[Condition]
    other: List[OtherHealthData]

    def __str__(self):
        return f"Medication: {self.medication}\nCondition: {self.condition}\nOther: {self.other}"


class ChatMessage(BaseModel):
    source: Literal["system", "user", "assistant"]
    body: Optional[str]

    def __str__(self):
        return f"{self.source}: {self.body}"


class HealthDataLLMResult(BaseModel):
    assistant_message: str
    health_data: Optional[HealthData]


def main(user_messages: List[str], nj_llm):
    history = [
        ChatMessage(
            source="system",
            body="""Help the user enter their health data step by step.
Ask specific questions to gather required and optional fields they have not already provided.
Stop asking if they don't know the answer
Automatically fix their spelling mistakes
Their health data may be complex: always record and return ALL of it.
Always return a response:
- If you don't understand what they say, ask a question.
- At least respond with an OK message.""",
        )
    ]
    health_data = None
    for user_message in user_messages:
        history.append(ChatMessage(source="user", body=user_message))
        history_str = "\n".join([str(msg) for msg in history])

        # Assistant
        result: HealthDataLLMResult = nj_llm(
            "Look at <history_str> to see the history of the conversation. If there is enough "
            "information to create a fully valid `HealthData` object (no missing required fields), "
            "set `health_data` to the validated object and set `assistant_message` to a response "
            "that acknowledges their given information. Otherwise, set `health_data` to None and "
            "set `assistant_message` to a question asking for the missing information.\n"
            f"<history_str>{history_str}</history_str>",
            output_format=HealthDataLLMResult,
        )

        assistant_message = result.assistant_message
        health_data = result.health_data

        history.append(ChatMessage(source="assistant", body=assistant_message))

        if health_data is not None:
            break
    return health_data, history


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    outputs = {}
    errors = {}
    user_messages = ["I broke my foot\nI broke it in high school\n2001 january 1st\nThe foot took a year to be ok\n"]
    correct_health_data = HealthData(
        medication=[],
        condition=[
            Condition(
                name="foot",
                startDate=datetime(2001, 1, 1),
                status="resolved",
                endDate=datetime(2002, 1, 1),
            )
        ],
        other=[],
    )
    hard_results = {
        "health_data_type": False,
        "health_data_condition": False,
        "health_data_condition_name": False,
        "health_data_condition_start_date": False,
        "health_data_condition_status": False,
        "health_data_condition_end_date": False,
    }
    try:
        res = main(user_messages, nj_llm)
    except Exception as e:
        errors["test_0"] = e
    else:
        outputs["test_0"] = res
        try:
            hard_results["health_data_type"] = isinstance(res[0], HealthData)
            if hard_results["health_data_type"]:
                hard_results["health_data_condition"] = len(res[0].condition) == 1
                hard_results["health_data_condition_name"] = "foot" in res[0].condition[0].name.lower()
                hard_results["health_data_condition_start_date"] = res[0].condition[0].startDate == datetime(2001, 1, 1)
                hard_results["health_data_condition_status"] = res[0].condition[0].status == "resolved"
                hard_results["health_data_condition_end_date"] = res[0].condition[0].endDate == datetime(2002, 1, 1)

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
