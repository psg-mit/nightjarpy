from typing import Any, Dict, List, Tuple

import nightjarpy as nj


class Task:
    def __init__(self, task_id: str, description: str, completed: bool):
        self.task_id = task_id
        self.description = description
        self.completed = completed

    def __str__(self):
        return f"('{self.task_id}', '{self.description}', {self.completed})"


@nj.fn
def main(tasks: List[Task]):
    """natural
    Mark all task in <tasks> related to housework as completed.
    """


#### Tests ####


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
    tasks_data = [("T001", "Clean the house", False), ("T002", "Prepare dinner", True), ("T003", "Go to gym", False)]
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
        main(tasks)
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
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
