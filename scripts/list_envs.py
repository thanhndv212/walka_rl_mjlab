"""List available mjlab environments, including walka-specific tasks."""

import mjlab
import mjlab.tasks
import tyro
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable

import src.tasks  # noqa: F401


def list_environments(keyword: str | None = None) -> int:
    table = PrettyTable(["#", "Task ID"])
    table.title = "Available Environments in mjlab"
    table.align["Task ID"] = "l"

    idx = 0
    for task_id in list_tasks():
        if keyword and keyword.lower() not in task_id.lower():
            continue
        table.add_row([idx + 1, task_id])
        idx += 1

    print(table)
    if idx == 0:
        msg = "[INFO] No tasks matched"
        if keyword:
            msg += f" keyword '{keyword}'"
        print(msg)
    return idx


def main() -> int:
    return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
    main()
