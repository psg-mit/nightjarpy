import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

import nightjarpy as nj


class Event(BaseModel):
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participants: List[str]
    description: str
    start_time: datetime
    end_time: datetime


class Calendar:
    def __init__(self, events: List[Event]):
        self._events = events

    def __eq__(self, other):
        for event in self._events:
            if event not in other._events:
                return False
        for event in other._events:
            if event not in self._events:
                return False
        return True

    def __str__(self):
        s = "===== Calendar =====\n"
        for event in self._events:
            s += f"{event.start_time} - {event.end_time}: {event.description} ({', '.join(event.participants)})\n"
        s += "====================\n"
        return s


@nj.fn
def main(calendar: Calendar, request: str) -> str:
    def add_event(event: Event):
        calendar._events.append(event)

    def remove_event(event_id: int):
        calendar._events = [event for event in calendar._events if event.id != event_id]

    def add_participants(event_id: int, participants: List[str]):
        for event in calendar._events:
            if event.id == event_id:
                event.participants.extend(participants)

    """natural
    Parse the <request> and update the <calendar> and <Event>s as needed using <add_event>, <remove_event>, and <add_participants>. If a response is expected by the request, save it to <:response> as a string, otherwise save <:response> as None.
    """
    return response


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    outputs = {}
    errors = {}
    hard_results = {}
    for i in range(1, 4):
        outputs[f"test_{i}"] = None
        errors[f"test_{i}"] = None
        hard_results[f"test_{i}"] = False
    calendar_start = Calendar(
        [
            Event(
                participants=["Gavin"],
                description="Weekly 1:1",
                start_time=datetime(2025, 4, 7, 12, 0),
                end_time=datetime(2025, 4, 7, 13, 0),
            ),
            Event(
                participants=["Gary", "Mary", "John"],
                description="Stand Up",
                start_time=datetime(2025, 4, 7, 13, 0),
                end_time=datetime(2025, 4, 7, 14, 0),
            ),
            Event(
                participants=["Gary"],
                description="Coffee Chat",
                start_time=datetime(2025, 4, 8, 14, 0),
                end_time=datetime(2025, 4, 8, 15, 0),
            ),
            Event(
                participants=["Gary", "Mary", "John"],
                description="Scrum",
                start_time=datetime(2025, 4, 16, 15, 0),
                end_time=datetime(2025, 4, 16, 16, 0),
            ),
        ]
    )
    calendar = deepcopy(calendar_start)
    test1_input = "I need to get my tires changed from 12:00 to 2:00 pm on Friday April 11, 2025"
    try:
        outputs["test_1"] = main(calendar, test1_input)
        print(calendar)
    except Exception as e:
        errors["test_1"] = e
    else:
        try:
            # Test 1: calendar has event
            for event in calendar._events:
                if (
                    event.start_time.date() == datetime(2025, 4, 11, 12, 0).date()
                    and event.start_time.time() == datetime(2025, 4, 11, 12, 0).time()
                    and event.end_time.date() == datetime(2025, 4, 11, 14, 0).date()
                    and event.end_time.time() == datetime(2025, 4, 11, 14, 0).time()
                ):
                    hard_results["test_1"] = True
                    break
        except Exception as e:
            errors["test_1"] = e

    calendar = deepcopy(calendar_start)
    test2_input = "Search for any meetings with Gavin this week"
    try:
        outputs["test_2"] = main(calendar, test2_input)
    except Exception as e:
        errors["test_2"] = e
    else:
        try:
            # Test 2: calendar hasn't changed
            hard_results["test_2"] = calendar == calendar_start
        except Exception as e:
            errors["test_2"] = e

    calendar = deepcopy(calendar_start)
    test3_input = "Please add Jennifer to the scrum next Thursday"
    try:
        outputs["test_3"] = main(calendar, test3_input)
    except Exception as e:
        errors["test_3"] = e
    else:
        try:
            for event in calendar._events:
                if event.description == "Scrum" and "Jennifer" in event.participants:
                    hard_results["test_3"] = True
                    break
        except Exception as e:
            errors["test_3"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
