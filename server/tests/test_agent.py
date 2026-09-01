from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, sleep

from PIL import Image

from labelone.agent import AgentRepository, AgentRunRequest, AgentService, AgentToolCall
from labelone.agent.planner import AgentPlan
from labelone.annotations import AnnotationStore
from labelone.datasets import DatasetScanRequest, scan_dataset
from labelone.datasets.repository import DatasetRepository
from labelone.jobs import JobRepository, JobService
from labelone.pipelines import PipelineEngine


def _service(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (64, 48), (20, 40, 60)).save(root / "image.png")
    (root / "image.json").write_text(json.dumps({"shapes": [{"label": "box", "shape_type": "rectangle", "points": [[1, 2], [10, 12]]}]}), encoding="utf-8")
    (root / "orphan.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
    scan = scan_dataset(DatasetScanRequest(dataset_id="dataset", root_dir=root, layout="same_directory"))
    datasets = DatasetRepository(tmp_path / "index.sqlite3")
    datasets.register(scan)
    annotations = AnnotationStore(datasets, tmp_path / "backups")
    pipeline = PipelineEngine(datasets, annotations, tmp_path / "artifacts")
    jobs = JobRepository(tmp_path / "index.sqlite3", datasets)
    job_service = JobService(jobs, datasets, pipeline, None)  # type: ignore[arg-type]
    agents = AgentRepository(tmp_path / "index.sqlite3")
    service = AgentService(agents, datasets, annotations, job_service)
    asset_id = datasets.selectable_asset_ids("dataset")[0]
    return service, agents, job_service, jobs, datasets, asset_id


def test_agent_read_tools_do_not_create_proposals(tmp_path: Path) -> None:
    service, agents, job_service, jobs, datasets, asset_id = _service(tmp_path)
    exceptions = service.run(AgentRunRequest(dataset_id="dataset", message="定位所有异常文件"))
    annotation = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="解释当前标注"))

    assert exceptions.state == "completed"
    assert "孤立 JSON 1" in exceptions.reply
    assert not exceptions.proposals
    assert annotation.state == "completed"
    assert "box×1" in annotation.reply
    job_service.close()
    agents.close()
    jobs.close()
    datasets.close()


def test_agent_write_tool_requires_confirmation_and_is_idempotent(tmp_path: Path) -> None:
    service, agents, job_service, jobs, datasets, asset_id = _service(tmp_path)
    proposed = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="给当前图设计增强流程"))
    ui_alias = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="为当前图设计增强流程"))

    assert proposed.state == "proposed"
    assert ui_alias.state == "proposed"
    assert proposed.proposals[0].requires_confirmation is True
    assert jobs.list().jobs == []
    executed = service.execute(proposed.run_id, proposed.proposals[0].id)
    duplicate = service.execute(proposed.run_id, proposed.proposals[0].id)

    assert executed.state == "completed"
    assert executed.proposals[0].executed is True
    assert duplicate.proposals[0].result == executed.proposals[0].result
    job_id = str(executed.proposals[0].result["job_id"])
    deadline = monotonic() + 5
    while monotonic() < deadline and jobs.get(job_id, include_items=False).state not in {"succeeded", "succeeded_with_errors"}:
        sleep(0.01)
    assert jobs.get(job_id).completed == 1
    job_service.close()
    agents.close()
    jobs.close()
    datasets.close()


def test_agent_uses_cloud_planner_only_for_unmatched_free_text(tmp_path: Path) -> None:
    service, agents, job_service, jobs, datasets, asset_id = _service(tmp_path)
    local_qa = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="解释当前标注"))
    assert "image.png" in local_qa.reply

    class Planner:
        def enabled(self) -> bool:
            return True

        def plan(self, message: str, *, has_asset: bool, history, operator_kinds) -> AgentPlan:  # noqa: ANN001
            assert message == "帮我概览一下这个项目"
            assert has_asset is False
            # Local QA replies contain display paths and must never be replayed
            # to the remote planner.
            assert history == []
            assert "crop" in operator_kinds
            return AgentPlan(reply="我先读取数据集统计。", tool_call=AgentToolCall(tool="dataset.stats"))

    service.cloud_planner = Planner()  # type: ignore[assignment]
    planned = service.run(AgentRunRequest(dataset_id="dataset", message="帮我概览一下这个项目"))
    local = service.run(AgentRunRequest(dataset_id="dataset", message="定位所有异常文件"))

    assert planned.state == "completed"
    assert "可见异常项" in planned.reply
    assert local.state == "completed"
    job_service.close()
    agents.close()
    jobs.close()
    datasets.close()


def test_agent_conversation_and_ui_actions_never_expose_source_mutation(tmp_path: Path) -> None:
    service, agents, job_service, jobs, datasets, asset_id = _service(tmp_path)

    class Planner:
        def enabled(self) -> bool:
            return True

        def plan(self, message: str, *, has_asset: bool, history, operator_kinds) -> AgentPlan:  # noqa: ANN001
            del history, operator_kinds
            if message == "聊一聊":
                return AgentPlan(reply="可以，我只能对话和操作 LabelOne 内的受控功能。")
            assert has_asset is True
            return AgentPlan(
                reply="我可以打开算子导入界面，确认后执行。",
                tool_call=AgentToolCall(tool="ui.import_operator"),
            )

    service.cloud_planner = Planner()  # type: ignore[assignment]
    conversation = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="聊一聊"))
    proposed = service.run(AgentRunRequest(dataset_id="dataset", asset_id=asset_id, message="导入算子"))

    assert conversation.state == "completed"
    assert "只能对话" in conversation.reply
    assert proposed.state == "proposed"
    assert proposed.proposals[0].tool == "ui.import_operator"
    assert proposed.proposals[0].requires_confirmation is True
    executed = service.execute(proposed.run_id, proposed.proposals[0].id)
    assert executed.proposals[0].result == {"action": "ui.import_operator"}
    assert jobs.list().jobs == []
    job_service.close()
    agents.close()
    jobs.close()
    datasets.close()


def test_agent_pipeline_draft_is_confirmed_ui_state_not_a_job_or_source_write(tmp_path: Path) -> None:
    service, agents, job_service, jobs, datasets, asset_id = _service(tmp_path)
    proposed = service.run(AgentRunRequest(
        dataset_id="dataset",
        asset_id=asset_id,
        message="生成处理流草案",
        tool_call=AgentToolCall(tool="pipeline.draft", arguments={
            "nodes": [
                {"id": "crop", "kind": "crop", "parameters": {"margin_ratio": 0.05}},
                {"id": "color", "kind": "color", "parameters": {"contrast": 1.1}},
            ],
        }),
    ))

    assert proposed.state == "proposed"
    assert proposed.proposals[0].title == "生成处理流草案"
    executed = service.execute(proposed.run_id, proposed.proposals[0].id)
    result = executed.proposals[0].result
    assert result["action"] == "pipeline.draft"
    assert [node["kind"] for node in result["pipeline_nodes"] if node["kind"] not in {"source", "output", "visualize"}] == ["crop", "color"]
    assert jobs.list().jobs == []
    job_service.close()
    agents.close()
    jobs.close()
    datasets.close()
