from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class Task:
    def __init__(self, task_id: str, description: str, completed: bool):
        self.task_id = task_id
        self.description = description
        self.completed = completed

    def __str__(self):
        return f"('{self.task_id}', '{self.description}', {self.completed})"


class MarkTaskCompleted(BaseModel):
    type: Literal["mark_task_completed"]
    task_id: str


class TaskCompletionLLMResult(BaseModel):
    commands: List[MarkTaskCompleted]


def main(tasks: List[Task], nj_llm):
    tasks_block = "\n".join([f"{t.task_id}\t{t.description}\tcompleted={t.completed}" for t in tasks])

    result: TaskCompletionLLMResult = nj_llm(
        "Review the <tasks> list and identify all tasks related to housework "
        "(e.g., cleaning, chores, home maintenance, laundry, dishes, vacuuming). "
        "For each housework-related task, add a `MarkTaskCompleted` command with the "
        "matching task_id."
        f"<tasks>{tasks_block}</tasks>",
        output_format=TaskCompletionLLMResult,
    )

    # Execute structured commands
    for cmd in result.commands:
        if isinstance(cmd, MarkTaskCompleted):
            for task in tasks:
                if task.task_id == cmd.task_id:
                    task.completed = True
                    break


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    tasks_data = [
        ("T001", "Clean the house", False),
        ("T002", "Prepare dinner", True),
        ("T003", "Go to gym", False),
    ]
    tasks = [Task(task_id, desc, done) for task_id, desc, done in tasks_data]
    initial_tasks = tasks.copy()
    outputs = {}
    errors = {}
    hard_results = {
        "test_0_completion_0": False,
        "test_0_completion_1": False,
        "test_0_completion_2": False,
    }

    try:
        main(tasks, nj_llm)
        outputs["test_0"] = tasks
    except Exception as e:
        errors["test_0"] = e
    else:
        try:
            hard_results["test_0_completion_0"] = tasks[0].completed == True
        except Exception as e:
            errors[f"test_0"] = e
        try:
            hard_results[f"test_0_completion_1"] = tasks[1].completed == True
        except Exception as e:
            errors[f"test_0"] = e
        try:
            hard_results[f"test_0_completion_2"] = tasks[2].completed == False
        except Exception as e:
            errors[f"test_0"] = e

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
