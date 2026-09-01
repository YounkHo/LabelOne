from __future__ import annotations

from collections import Counter
import json
from math import isfinite
from uuid import uuid4

from labelone.annotations import AnnotationStore
from labelone.datasets.repository import DatasetRepository
from labelone.errors import AgentBackendUnavailableError, InvalidPathError, ModelRuntimeError
from labelone.jobs import BatchJobRequest, JobService
from labelone.models import ModelManager
from labelone.models.types import AvailabilityState
from labelone.pipelines import PipelineNode, normalize_legacy_nodes, operator_catalog, validate_nodes

from .models import AgentCapability, AgentProposal, AgentRun, AgentRunRequest, AgentStatus, AgentToolCall, AgentToolResult
from .planner import CloudAgentPlanner
from .repository import AgentRepository
from .tools import (
    AnnotationQaArguments,
    DatasetDistributionArguments,
    DatasetSearchArguments,
    InferenceJobArguments,
    PipelineDraftArguments,
    PipelineJobArguments,
    UiActionArguments,
    legacy_tool_call,
    parse_tool_arguments,
    validate_json_budget,
)


class AgentService:
    def __init__(
        self,
        repository: AgentRepository,
        datasets: DatasetRepository,
        annotations: AnnotationStore,
        jobs: JobService,
        models: ModelManager | None = None,
        cloud_planner: CloudAgentPlanner | None = None,
    ) -> None:
        self.repository = repository
        self.datasets = datasets
        self.annotations = annotations
        self.jobs = jobs
        self.models = models
        self.cloud_planner = cloud_planner

    def status(self) -> AgentStatus:
        capabilities = [
            AgentCapability(tool="dataset.stats", group="inspect", title="数据集概况", description="统计有效条目与异常文件，不修改数据。", risk="read", requires_confirmation=False),
            AgentCapability(tool="dataset.search", group="inspect", title="筛选图片", description="按名称、状态或标注情况查询数据集。", risk="read", requires_confirmation=False),
            AgentCapability(tool="dataset.distribution", group="inspect", title="标签分布", description="汇总标签、形状类型和文件状态分布。", risk="read", requires_confirmation=False),
            AgentCapability(tool="annotation.qa", group="inspect", title="当前标注质检", description="检查空标注、越界、退化和重复对象。", risk="read", requires_confirmation=False, requires_asset=True),
            AgentCapability(tool="ui.open_dataset", group="prepare", title="打开数据集", description="生成打开数据集的界面动作提案。", risk="write", requires_confirmation=True),
            AgentCapability(tool="ui.import_operator", group="prepare", title="导入算子", description="生成进入算子库并选择安装包的界面动作。", risk="write", requires_confirmation=True),
            AgentCapability(tool="ui.open_models", group="prepare", title="选择模型", description="生成进入推理区并打开模型选择器的界面动作。", risk="write", requires_confirmation=True),
            AgentCapability(tool="pipeline.draft", group="prepare", title="处理流草案", description="仅使用已注册算子生成可审核草案。", risk="write", requires_confirmation=True),
            AgentCapability(tool="pipeline.create_job", group="run", title="处理流任务", description="确认后创建当前图或全数据集后台任务。", risk="write", requires_confirmation=True),
            AgentCapability(tool="inference.create_job", group="run", title="模型推理任务", description="校验模型与参数后，确认创建推理任务。", risk="write", requires_confirmation=True),
        ]
        if self.cloud_planner is None:
            return AgentStatus(
                state="unconfigured",
                reason_code="invalid_configuration",
                message="Agent 后端尚未接入本地服务。",
                capabilities=capabilities,
            )
        readiness = self.cloud_planner.readiness()
        return AgentStatus(
            state="ready" if readiness.ready else "unconfigured",
            reason_code=readiness.reason_code,
            message=readiness.message,
            model=readiness.model,
            credential_env=readiness.credential_env,
            capabilities=capabilities,
        )

    def require_backend(self) -> AgentStatus:
        status = self.status()
        if status.state != "ready":
            raise AgentBackendUnavailableError(
                status.message,
                details={"reason_code": status.reason_code, "credential_env": status.credential_env or ""},
            )
        return status

    @staticmethod
    def _shape_area(points: list[list[float]]) -> float:
        if len(points) < 3:
            return 0.0
        return abs(sum(
            float(points[index][0]) * float(points[(index + 1) % len(points)][1])
            - float(points[index][1]) * float(points[(index + 1) % len(points)][0])
            for index in range(len(points))
        )) * 0.5

    def _dataset_stats(self, dataset_id: str) -> tuple[dict[str, object], str]:
        dataset = self.datasets.get_dataset(dataset_id)
        summary = dataset.summary
        abnormal = summary.duplicate_match + summary.orphan_annotation + summary.corrupt_image + summary.corrupt_annotation
        data = {
            "dataset_id": dataset_id,
            "name": dataset.name,
            "index_revision": dataset.index_revision,
            "valid": summary.valid,
            "visible_abnormal": abnormal,
            "duplicate_match": summary.duplicate_match,
            "orphan_annotation": summary.orphan_annotation,
            "corrupt_image": summary.corrupt_image,
            "corrupt_annotation": summary.corrupt_annotation,
            "hidden_image_only": summary.hidden_image_only,
        }
        reply = (
            f"数据集当前有 {abnormal} 个可见异常项：重复匹配 {summary.duplicate_match}、"
            f"孤立 JSON {summary.orphan_annotation}、损坏图像 {summary.corrupt_image}、"
            f"损坏标注 {summary.corrupt_annotation}。仅图像未匹配的 {summary.hidden_image_only} 项按规则隐藏。"
        )
        return data, reply

    def _search(self, dataset_id: str, arguments: DatasetSearchArguments) -> tuple[dict[str, object], str]:
        page = self.datasets.search_assets_cursor(
            dataset_id,
            query=arguments.query,
            mode=arguments.mode,
            limit=arguments.limit,
            status=arguments.status,
            annotated=arguments.annotated,
        )
        data = {
            "query": arguments.query,
            "mode": arguments.mode,
            "total": page.total,
            "returned": len(page.items),
            "truncated": page.next_cursor is not None,
            "items": [
                {
                    "asset_id": item.asset_id,
                    "display_path": item.display_path,
                    "status": item.status.value,
                    "selectable": item.selectable,
                    "annotation_count": item.annotation_count,
                    "labels": item.labels,
                    "shape_types": item.shape_types,
                }
                for item in page.items
            ],
        }
        return data, f"服务端查询命中 {page.total} 项，本次按预算返回前 {len(page.items)} 项。"

    def _annotation_qa(
        self,
        dataset_id: str,
        asset_id: str | None,
        arguments: AnnotationQaArguments,
    ) -> tuple[dict[str, object], str]:
        if not asset_id:
            raise InvalidPathError("annotation.qa requires the current asset_id")
        asset = self.datasets.get_asset(dataset_id, asset_id, require_selectable=True)
        envelope = self.annotations.load(dataset_id, asset_id)
        document = envelope.document
        shapes = document.get("shapes", []) if isinstance(document, dict) else []
        if not isinstance(shapes, list):
            shapes = []
        width = int(document.get("imageWidth") or asset.width or 0)
        height = int(document.get("imageHeight") or asset.height or 0)
        labels: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        samples: list[dict[str, object]] = []
        seen: dict[str, int] = {}
        for index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                issue_counts["invalid_shape"] += 1
                continue
            label = str(shape.get("label") or "unlabeled")
            shape_type = str(shape.get("shape_type") or "polygon")
            labels[label] += 1
            raw_points = shape.get("points")
            points = raw_points if isinstance(raw_points, list) else []
            numeric = all(
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)) for value in point)
                for point in points
            )
            issues: list[str] = []
            if not numeric or not points:
                issues.append("invalid_points")
            else:
                if width > 0 and height > 0 and any(
                    float(point[0]) < 0 or float(point[1]) < 0 or float(point[0]) > width or float(point[1]) > height
                    for point in points
                ):
                    issues.append("out_of_bounds")
                unique = {(round(float(point[0]), arguments.duplicate_precision), round(float(point[1]), arguments.duplicate_precision)) for point in points}
                if (shape_type == "point" and len(unique) != 1) or (shape_type != "point" and len(unique) < 2):
                    issues.append("degenerate")
                elif len(points) >= 3 and self._shape_area(points) <= 1e-6:
                    issues.append("degenerate")
                signature = json.dumps(
                    {"label": label, "shape_type": shape_type, "points": sorted(unique)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if signature in seen:
                    issues.append("duplicate")
                else:
                    seen[signature] = index
            for issue in issues:
                issue_counts[issue] += 1
            if issues and len(samples) < 50:
                samples.append({"shape_index": index, "label": label, "issues": issues})
        if not shapes:
            issue_counts["empty_annotation"] = 1
        data = {
            "asset_id": asset_id,
            "display_path": asset.display_path,
            "revision": envelope.revision,
            "shape_count": len(shapes),
            "labels": dict(sorted(labels.items())),
            "issues": dict(sorted(issue_counts.items())),
            "samples": samples,
        }
        label_summary = "、".join(f"{label}×{count}" for label, count in sorted(labels.items())) or "无对象"
        issue_total = sum(issue_counts.values())
        return data, f"当前图 {asset.display_path} 有 {len(shapes)} 个标注：{label_summary}；质检发现 {issue_total} 个问题。本次只读。"

    def _distribution(self, dataset_id: str, arguments: DatasetDistributionArguments) -> tuple[dict[str, object], str]:
        cursor = None
        scanned = 0
        labels: Counter[str] = Counter()
        shape_types: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        total = 0
        while scanned < arguments.max_assets:
            page = self.datasets.list_assets_cursor(
                dataset_id,
                cursor=cursor,
                limit=min(1000, arguments.max_assets - scanned),
            )
            total = page.total
            for item in page.items:
                statuses[item.status.value] += 1
                labels.update(item.labels)
                shape_types.update(item.shape_types)
            scanned += len(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        data = {
            "scanned_assets": scanned,
            "total_assets": total,
            "truncated": scanned < total,
            "labels": dict(labels.most_common(arguments.top_n)),
            "shape_types": dict(shape_types.most_common(arguments.top_n)),
            "statuses": dict(statuses.most_common()),
        }
        return data, f"统计了 {scanned}/{total} 个可见条目；标签和 shape 分布已按 Top {arguments.top_n} 返回。"

    @staticmethod
    def _scope_asset_ids(scope: str, asset_id: str | None) -> list[str]:
        if scope == "current":
            if not asset_id:
                raise InvalidPathError("Current-image proposal requires asset_id")
            return [asset_id]
        return []

    def _validate_model(self, model_id: str):
        if self.models is None:
            raise InvalidPathError("Inference Agent tool is not configured")
        record = self.models.catalog.get(model_id)
        state = self.models.state(model_id)
        if not record.descriptor.capabilities.predict:
            raise InvalidPathError(
                "Selected model does not declare predict capability",
                details={"model_id": model_id, "adapter": record.descriptor.adapter},
            )
        if state.state != "loaded" and record.descriptor.availability.state is not AvailabilityState.AVAILABLE:
            raise InvalidPathError(
                "Selected model is not loaded or locally loadable",
                details={"model_id": model_id, "availability": record.descriptor.availability.state.value},
            )
        return record, state

    @staticmethod
    def _validate_model_parameters(record, parameters: dict[str, object]) -> None:
        validate_json_budget(parameters)
        schema = record.descriptor.capabilities.parameters_schema or {}
        properties = schema.get("properties") if isinstance(schema, dict) else None
        allowed = set(properties) if isinstance(properties, dict) else set()
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise InvalidPathError(
                "Inference parameters are not declared by the model",
                details={"unknown_parameters": unknown, "allowed": sorted(allowed)},
            )
        for name, value in parameters.items():
            property_schema = properties[name]
            expected = property_schema.get("type") if isinstance(property_schema, dict) else None
            if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                raise InvalidPathError("Inference parameter must be integer", details={"parameter": name})
            if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise InvalidPathError("Inference parameter must be numeric", details={"parameter": name})
            if expected == "string" and not isinstance(value, str):
                raise InvalidPathError("Inference parameter must be string", details={"parameter": name})
            if expected == "boolean" and not isinstance(value, bool):
                raise InvalidPathError("Inference parameter must be boolean", details={"parameter": name})
            if isinstance(property_schema, dict) and isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in property_schema and float(value) < float(property_schema["minimum"]):
                    raise InvalidPathError("Inference parameter is below minimum", details={"parameter": name})
                if "maximum" in property_schema and float(value) > float(property_schema["maximum"]):
                    raise InvalidPathError("Inference parameter exceeds maximum", details={"parameter": name})
            if isinstance(property_schema, dict) and "enum" in property_schema and value not in property_schema["enum"]:
                raise InvalidPathError("Inference parameter is not an allowed value", details={"parameter": name})

    def _proposal(
        self,
        request: AgentRunRequest,
        call: AgentToolCall,
        arguments,
        *,
        generated_reply: str | None = None,
    ) -> AgentRun:
        proposal_id = uuid4().hex[:12]
        if call.tool in {"ui.open_dataset", "ui.import_operator", "ui.open_models"}:
            assert isinstance(arguments, UiActionArguments)
            labels = {
                "ui.open_dataset": ("打开数据集", "打开系统文件夹选择器并开始数据集扫描。"),
                "ui.import_operator": ("导入算子", "打开设置中的算子库并调用系统文件选择器。"),
                "ui.open_models": ("打开模型选择", "切换到推理面板并展开模型选择器。"),
            }
            title, description = labels[call.tool]
            payload = {"action": call.tool}
            proposal = AgentProposal(
                id=proposal_id,
                tool=call.tool,
                title=title,
                description=description,
                risk="write",
                requires_confirmation=True,
            )
        elif call.tool == "pipeline.draft":
            assert isinstance(arguments, PipelineDraftArguments)
            normalized = validate_nodes(normalize_legacy_nodes(arguments.nodes), maximum_nodes=34)
            pipeline_nodes = [
                {"id": node.id, "kind": node.kind, "enabled": node.enabled, "parameters": dict(node.parameters)}
                for node in normalized
            ]
            payload = {"action": call.tool, "pipeline_nodes": pipeline_nodes}
            proposal = AgentProposal(
                id=proposal_id,
                tool=call.tool,
                title="生成处理流草案",
                description=f"使用 {len(normalized)} 个已注册节点替换当前处理流草案；不会执行任务或写入源码。",
                risk="write",
                requires_confirmation=True,
            )
        elif call.tool == "pipeline.create_job":
            assert isinstance(arguments, PipelineJobArguments)
            normalized = validate_nodes(normalize_legacy_nodes(arguments.nodes), maximum_nodes=34)
            asset_ids = self._scope_asset_ids(arguments.scope, request.asset_id)
            payload: dict[str, object] = {
                "dataset_id": request.dataset_id,
                "asset_ids": asset_ids,
                "concurrency": arguments.concurrency,
                "pipeline_nodes": [
                    {"id": node.id, "kind": node.kind, "enabled": node.enabled, "parameters": dict(node.parameters)}
                    for node in normalized
                ],
            }
            proposal = AgentProposal(
                id=proposal_id,
                tool=call.tool,
                title="创建处理流任务",
                description=f"范围：{'当前图' if arguments.scope == 'current' else '全部可选图像'}；{len(normalized)} 个内置算子；只写任务 artifact。",
                risk="write",
                requires_confirmation=True,
            )
        else:
            assert isinstance(arguments, InferenceJobArguments)
            record, state = self._validate_model(arguments.model_id)
            self._validate_model_parameters(record, arguments.parameters)
            if len(set(arguments.capture_layers)) != len(arguments.capture_layers):
                raise InvalidPathError("Inference capture layers must be unique")
            if arguments.capture_layers:
                if state.state != "loaded":
                    raise InvalidPathError(
                        "Capture layers require a currently loaded model so adapter outputs can be verified",
                        details={"model_id": arguments.model_id, "state": state.state},
                    )
                available_layers = {layer.id for layer in state.layers}
                unknown_layers = sorted(set(arguments.capture_layers) - available_layers)
                if unknown_layers:
                    raise InvalidPathError(
                        "Inference capture layers are unavailable",
                        details={"unknown_layers": unknown_layers, "available_layers": sorted(available_layers)},
                    )
            asset_ids = self._scope_asset_ids(arguments.scope, request.asset_id)
            payload = {
                "dataset_id": request.dataset_id,
                "asset_ids": asset_ids,
                "model_id": arguments.model_id,
                "capture_layers": arguments.capture_layers,
                "parameters": arguments.parameters,
            }
            remote_notice = (
                "；确认后图片会发送到用户配置的受信 HTTPS endpoint，远程黑盒不提供中间层"
                if record.descriptor.adapter == "trusted_remote_http"
                else ""
            )
            proposal = AgentProposal(
                id=proposal_id,
                tool=call.tool,
                title="创建模型推理任务",
                description=f"模型：{record.descriptor.display_name}；当前状态：{state.state}；范围：{'当前图' if arguments.scope == 'current' else '全部可选图像'}{remote_notice}。",
                risk="write",
                requires_confirmation=True,
            )
        return self.repository.create(
            dataset_id=request.dataset_id,
            asset_id=request.asset_id,
            message=request.message.strip(),
            reply=generated_reply or "已生成受控操作提案。只有再次确认后才会执行。",
            state="proposed",
            proposals=[(proposal, payload)],
        )

    def run(self, request: AgentRunRequest) -> AgentRun:
        self.datasets.get_dataset(request.dataset_id)
        if request.asset_id:
            self.datasets.get_asset(request.dataset_id, request.asset_id, require_selectable=True)
        message = request.message.strip()
        call = request.tool_call or legacy_tool_call(message, has_asset=request.asset_id is not None)
        generated_reply: str | None = None
        if call is None and self.cloud_planner is not None and self.cloud_planner.enabled():
            try:
                plan = self.cloud_planner.plan(
                    message,
                    has_asset=request.asset_id is not None,
                    # Local tool replies can include file display paths and result
                    # summaries. They are intentionally never replayed to the
                    # remote planner.
                    history=[],
                    operator_kinds=[
                        str(item["kind"])
                        for item in operator_catalog()
                        if item.get("node_role") == "transform" or str(item.get("kind")) not in {"source", "output", "visualize"}
                    ],
                )
                call = plan.tool_call
                generated_reply = plan.reply
            except Exception as exc:
                return self.repository.create(
                    dataset_id=request.dataset_id,
                    asset_id=request.asset_id,
                    message=message,
                    reply=f"云端 AI 规划失败：{exc}",
                    state="failed",
                    proposals=[],
                )
        if call is None:
            return self.repository.create(
                dataset_id=request.dataset_id,
                asset_id=request.asset_id,
                message=message,
                reply=generated_reply or "请从允许的工具中选择：数据集统计、服务端搜索、当前标注质检、标签分布、打开/导入应用动作、处理流草案或推理提案。自由文本和数据内容不会被当作源码或命令执行。",
                state="completed",
                proposals=[],
            )
        try:
            validate_json_budget(call.arguments)
            arguments = parse_tool_arguments(call)
            if call.tool in {"ui.open_dataset", "ui.import_operator", "ui.open_models", "pipeline.draft", "pipeline.create_job", "inference.create_job"}:
                return self._proposal(request, call, arguments, generated_reply=generated_reply)
            if call.tool == "dataset.stats":
                data, reply = self._dataset_stats(request.dataset_id)
            elif call.tool == "dataset.search":
                assert isinstance(arguments, DatasetSearchArguments)
                data, reply = self._search(request.dataset_id, arguments)
            elif call.tool == "annotation.qa":
                assert isinstance(arguments, AnnotationQaArguments)
                data, reply = self._annotation_qa(request.dataset_id, request.asset_id, arguments)
            else:
                assert isinstance(arguments, DatasetDistributionArguments)
                data, reply = self._distribution(request.dataset_id, arguments)
        except Exception as exc:
            return self.repository.create(
                dataset_id=request.dataset_id,
                asset_id=request.asset_id,
                message=message,
                reply=f"工具请求被拒绝：{exc}",
                state="failed",
                proposals=[],
                audits=[(call.tool, "read" if call.tool.startswith(("dataset.", "annotation.")) else "write", "failed", call.arguments, {"error": str(exc)})],
            )
        result = AgentToolResult(tool=call.tool, data=data)
        return self.repository.create(
            dataset_id=request.dataset_id,
            asset_id=request.asset_id,
            message=message,
            reply=reply,
            state="completed",
            proposals=[],
            tool_results=[result],
            audits=[(call.tool, "read", "completed", call.arguments, data)],
        )

    def execute(self, run_id: str, proposal_id: str) -> AgentRun:
        run = self.repository.get(run_id)
        tool, payload, executed = self.repository.proposal_payload(run_id, proposal_id)
        if executed:
            self.repository.record_audit(
                run_id,
                tool=tool,
                risk="write",
                status="idempotent",
                arguments=payload,
                result={"proposal_id": proposal_id},
            )
            return run
        try:
            validate_json_budget(payload)
            if tool in {"ui.open_dataset", "ui.import_operator", "ui.open_models", "pipeline.draft"}:
                result: dict[str, object] = {"action": tool}
                if tool == "pipeline.draft":
                    result["pipeline_nodes"] = [dict(node) for node in payload.get("pipeline_nodes", [])]
                    completed_reply = "处理流草案已确认并发送到编辑器；尚未运行任务。"
                else:
                    completed_reply = "应用动作已确认，正在交给当前界面执行。"
                completed = self.repository.complete_proposal(run_id, proposal_id, result, completed_reply)
                self.repository.record_audit(
                    run_id,
                    tool=tool,
                    risk="write",
                    status="executed",
                    arguments=payload,
                    result=result,
                )
                return completed
            if tool == "pipeline.create_job":
                request = BatchJobRequest(
                    kind="pipeline",
                    dataset_id=str(payload["dataset_id"]),
                    asset_ids=[str(item) for item in payload.get("asset_ids", [])],
                    concurrency=int(payload["concurrency"]),
                    pipeline_nodes=[PipelineNode.model_validate(node) for node in payload["pipeline_nodes"]],
                )
            elif tool == "inference.create_job":
                if self.models is None:
                    raise InvalidPathError("Inference Agent tool is not configured")
                model_id = str(payload["model_id"])
                record, state = self._validate_model(model_id)
                parameters = dict(payload.get("parameters", {}))
                self._validate_model_parameters(record, parameters)
                if state.state != "loaded":
                    state = self.models.load(model_id, ["CPUExecutionProvider"])
                capture_layers = [str(item) for item in payload.get("capture_layers", [])]
                available_layers = {layer.id for layer in state.layers}
                unknown_layers = sorted(set(capture_layers) - available_layers)
                if unknown_layers:
                    raise InvalidPathError("Inference capture layers are unavailable", details={"unknown_layers": unknown_layers})
                request = BatchJobRequest(
                    kind="inference",
                    dataset_id=str(payload["dataset_id"]),
                    asset_ids=[str(item) for item in payload.get("asset_ids", [])],
                    concurrency=1,
                    model_id=model_id,
                    capture_layers=capture_layers,
                    parameters=parameters,
                )
            else:
                raise InvalidPathError("Agent tool is not allowlisted", details={"tool": tool})
            job = self.jobs.create(request, idempotency_key=f"agent:{run_id}:{proposal_id}")
            result = {"job_id": job.job_id, "state": job.state, "total": job.total}
            completed = self.repository.complete_proposal(
                run_id,
                proposal_id,
                result,
                f"提案已确认并创建持久任务 {job.job_id[:12]}，共 {job.total} 项。后续进度可在右上角后台任务中查看。",
            )
            self.repository.record_audit(
                run_id,
                tool=tool,
                risk="write",
                status="executed",
                arguments=payload,
                result=result,
            )
            return completed
        except Exception as exc:
            self.repository.record_audit(
                run_id,
                tool=tool,
                risk="write",
                status="failed",
                arguments=payload,
                result={"error": str(exc)},
            )
            if isinstance(exc, (InvalidPathError, ModelRuntimeError)):
                raise
            raise InvalidPathError("Agent proposal execution failed", details={"error": str(exc)}) from exc
