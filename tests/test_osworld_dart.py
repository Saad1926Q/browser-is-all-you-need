from __future__ import annotations

import json

from w8_biayn.osworld_custom import registry
from w8_biayn.osworld_dart.tasks import export_custom_tasks


def test_export_custom_tasks_writes_dart_task_file_and_examples(tmp_path):
    tasks = [registry.load_task("src/w8_biayn/osworld_custom/tasks/add_todo_comment")]

    result = export_custom_tasks(tasks, tmp_path / "dart", task_type="custom")

    assert result.task_count == 1
    assert result.task_file == tmp_path / "dart" / "evaluation_examples" / "custom_train.json"
    assert result.osworld_root == tmp_path / "dart" / "evaluation_examples" / "examples_custom"
    task_id = tasks[0].task_id
    assert json.loads(result.task_file.read_text(encoding="utf-8")) == {"custom": [task_id]}

    exported = result.osworld_root / "custom" / f"{task_id}.json"
    exported_data = json.loads(exported.read_text(encoding="utf-8"))
    assert exported_data["id"] == task_id
    assert exported_data["instruction"].startswith("Add the comment # TODO: implement")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "w8-osworld-dart-export-v1"
    assert manifest["task_count"] == 1
    assert manifest["task_file"] == str(result.task_file)
    assert manifest["osworld_root"] == str(result.osworld_root)
