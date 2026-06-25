from __future__ import annotations

import json

from w8_biayn.osworld_custom import registry
from w8_biayn.osworld_dart.config import write_dart_run_config
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


def test_write_dart_run_config_writes_rollouter_files(tmp_path):
    task_file = tmp_path / "data" / "evaluation_examples" / "custom_train.json"
    osworld_root = tmp_path / "data" / "evaluation_examples" / "examples_custom"
    task_file.parent.mkdir(parents=True)
    osworld_root.mkdir(parents=True)
    task_file.write_text('{"custom": ["task-1"]}\n', encoding="utf-8")

    result = write_dart_run_config(
        run_id="dart-test",
        task_file=task_file,
        osworld_root=osworld_root,
        out_dir=tmp_path / "run",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        rollout_server_url="http://127.0.0.1:15959",
        max_steps=3,
        force=True,
    )

    assert result.rollouter_config.exists()
    assert result.env_file.exists()
    assert result.trainer_script.exists()
    assert result.manifest_path.exists()

    rollouter_config = result.rollouter_config.read_text(encoding="utf-8")
    assert f'task_file: "{task_file}"' in rollouter_config
    assert f'osworld_root: "{osworld_root}"' in rollouter_config
    assert "max_steps: 3" in rollouter_config
    assert "write_to_mysql: false" in rollouter_config

    env = result.env_file.read_text(encoding="utf-8")
    assert "export W8_DART_RUN_ID=dart-test" in env
    assert f"export W8_DART_TASK_FILE={task_file}" in env

    script = result.trainer_script.read_text(encoding="utf-8")
    assert "python3 -m src.run" in script
    assert "--config-path" in script
    assert "--config-name" in script

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "w8-osworld-dart-run-config-v1"
    assert manifest["run_id"] == "dart-test"
