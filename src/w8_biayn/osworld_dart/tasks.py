from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from w8_biayn.osworld_custom.registry import CustomTask, repo_path, unique_by_id

DEFAULT_DART_TASKSET_NAME = "custom_train.json"
DEFAULT_DART_EXAMPLES_DIR = "examples_custom"


@dataclass(frozen=True)
class DartExportResult:
    out_dir: Path
    evaluation_examples_dir: Path
    task_file: Path
    osworld_root: Path
    manifest_path: Path
    task_type: str
    task_count: int
    task_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "evaluation_examples_dir": str(self.evaluation_examples_dir),
            "task_file": str(self.task_file),
            "osworld_root": str(self.osworld_root),
            "manifest_path": str(self.manifest_path),
            "task_type": self.task_type,
            "task_count": self.task_count,
            "task_ids": self.task_ids,
            "dart_config": {
                "task.task_file": str(self.task_file),
                "task.osworld_root": str(self.osworld_root),
            },
        }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_task_type(task_type: str) -> str:
    value = task_type.strip()
    if not value:
        raise ValueError("task_type must be non-empty")
    if "/" in value or "\\" in value:
        raise ValueError("task_type must not contain path separators")
    return value


def _ordered_unique_tasks(tasks: Iterable[CustomTask]) -> list[CustomTask]:
    unique = unique_by_id(tasks)
    if not unique:
        raise ValueError("no tasks selected for DART export")
    return unique


def export_custom_tasks(
    tasks: Iterable[CustomTask],
    out_dir: str | Path,
    *,
    task_type: str = "custom",
    taskset_name: str = DEFAULT_DART_TASKSET_NAME,
    examples_dir_name: str = DEFAULT_DART_EXAMPLES_DIR,
    force: bool = False,
) -> DartExportResult:
    """Export custom OSWorld tasks into DART-GUI's task_file/osworld_root layout.

    DART's TaskLoader expects task.task_file to be a JSON object of:

        {"<task_type>": ["<task_id>", ...]}

    and task.osworld_root to contain detail files at:

        <osworld_root>/<task_type>/<task_id>.json
    """

    selected = _ordered_unique_tasks(tasks)
    clean_task_type = _clean_task_type(task_type)
    out_path = Path(out_dir)
    evaluation_examples = out_path / "evaluation_examples"
    osworld_root = evaluation_examples / examples_dir_name
    task_type_root = osworld_root / clean_task_type
    task_file = evaluation_examples / taskset_name
    manifest_path = out_path / "manifest.json"

    if out_path.exists() and any(out_path.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty; pass --force to overwrite: {out_path}")

    task_type_root.mkdir(parents=True, exist_ok=True)

    task_ids: list[str] = []
    manifest_tasks: list[dict[str, Any]] = []
    for task in selected:
        task_ids.append(task.task_id)
        exported_task_path = task_type_root / f"{task.task_id}.json"
        _write_json(exported_task_path, task.data)
        manifest_tasks.append(
            {
                "task_id": task.task_id,
                "task_type": clean_task_type,
                "source_path": repo_path(task.path),
                "exported_path": str(exported_task_path),
                "domain": task.domain,
                "instruction": task.instruction,
            }
        )

    _write_json(task_file, {clean_task_type: task_ids})
    _write_json(
        manifest_path,
        {
            "schema_version": "w8-osworld-dart-export-v1",
            "task_count": len(task_ids),
            "task_type": clean_task_type,
            "task_file": str(task_file),
            "osworld_root": str(osworld_root),
            "examples_dir": str(task_type_root),
            "tasks": manifest_tasks,
        },
    )

    return DartExportResult(
        out_dir=out_path,
        evaluation_examples_dir=evaluation_examples,
        task_file=task_file,
        osworld_root=osworld_root,
        manifest_path=manifest_path,
        task_type=clean_task_type,
        task_count=len(task_ids),
        task_ids=task_ids,
    )
