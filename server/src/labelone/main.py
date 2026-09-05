from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
import mimetypes
import os
from threading import Event as ThreadEvent, RLock

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from labelone import __version__
from labelone.api_models import ApplicationSettingsResponse, ApplicationSettingsUpdate, DirectoryPickerRequest, DirectoryPickerResponse, ErrorResponse, HealthResponse, ModelWeightDownloadRequest
from labelone.application_settings import MODEL_DOWNLOAD_SOURCES, ApplicationSettingsStore, apply_network_proxy_environment
from labelone.workspace_settings import DatasetWorkspaceSettings, DatasetWorkspaceSettingsResponse, DatasetWorkspaceSettingsUpdate, ModelUsageRecord, WorkspacePipelineSettings
from labelone.agent import AgentAuditRecord, AgentRepository, AgentRun, AgentRunRequest, AgentService, AgentStatus
from labelone.agent.planner import CloudAgentPlanner
from labelone.annotations import AnnotationEnvelope, AnnotationSaveRequest, AnnotationSaveResponse, AnnotationStore
from labelone.config import Settings, settings as default_settings
from labelone.datasets import (
    AssetCursorPage,
    DatasetScanItemPage,
    DatasetScanRequest,
    DatasetScanResult,
    DatasetScanSession,
    DatasetScanSessionList,
    DatasetScanSessionStore,
    scan_dataset,
)
from labelone.datasets.models import AssetListResponse, AssetStatus, DatasetAsset, DatasetListResponse, RegisteredDataset
from labelone.datasets.repository import DatasetRepository
from labelone.datasets.revalidate import revalidate_asset
from labelone.errors import AnnotationValidationError, InvalidPathError, LabelOneError, ModelRuntimeError, RevisionConflictError
from labelone.images import DeepZoomTileService, ImageService
from labelone.jobs import BatchJobRequest, JobListResponse, JobPriorityRequest, JobRecord, JobRepository, JobService, PipelinePrecomputeEnsureResponse
from labelone.jobs.models import JobItemListResponse, JobItemLookupRequest
from labelone.jobs.sse import event_page_payload, resolve_event_cursor, stream_job_events
from labelone.models import ModelCatalog, ModelManager
from labelone.models.artifacts import ArtifactStore
from labelone.models.sources import ModelSourceStore
from labelone.models.weights import ModelWeightStore
from labelone.models.types import (
    ImportCatalogRequest,
    InferenceRequest,
    InferenceResult,
    ModelCatalogResponse,
    ModelCatalogStatus,
    ModelLoadRequest,
    ModelRuntimeState,
)
from labelone.pipelines import (
    CompositeDefinitionStore,
    CompositeRegistry,
    OperatorPackageManager,
    PipelineEngine,
    PipelinePreviewRequest,
    PipelinePreviewResult,
    PipelineValidationRequest,
    PipelineValidationResult,
    operator_catalog,
    operator_registry_hash,
    register_operator_contracts,
    unregister_operator_contracts,
    validate_pipeline_definition,
)
from labelone.system_picker import pick_directory


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or default_settings
    application_settings = ApplicationSettingsStore(active_settings.data_dir / "application-settings.json")
    effective_network_proxy = application_settings.network_proxy()
    apply_network_proxy_environment(effective_network_proxy)
    persisted_model_weights_dir = application_settings.model_weights_dir()
    configured_model_weights_dir = (
        active_settings.model_weights_dir.expanduser().resolve()
        if active_settings.model_weights_dir is not None
        else persisted_model_weights_dir
        if persisted_model_weights_dir is not None
        else (active_settings.data_dir.expanduser().resolve() / "model-weights")
    )
    catalog = ModelCatalog()
    builtin_model_root = Path(__file__).resolve().parent / "model_library"
    catalog.import_x_anylabeling(builtin_model_root)
    artifact_store = ArtifactStore(active_settings.data_dir / "artifacts")
    weight_store = ModelWeightStore(active_settings.data_dir, root_dir=configured_model_weights_dir)
    manager = ModelManager(
        catalog,
        artifact_store,
        weight_store,
        isolate_processes=True,
        data_dir=active_settings.data_dir,
        model_weights_dir=configured_model_weights_dir,
        worker_startup_timeout=15.0,
        worker_request_timeout=120.0,
        worker_close_timeout=2.0,
        worker_max_request_bytes=2 * 1024 * 1024,
        worker_max_response_bytes=8 * 1024 * 1024,
    )
    model_sources = ModelSourceStore(active_settings.data_dir / "model-sources.json")
    dataset_repository = DatasetRepository(active_settings.data_dir / "index.sqlite3")
    scan_session_store = DatasetScanSessionStore(active_settings.data_dir / "index.sqlite3")
    scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labelone-scan")
    scan_dispatch_lock = RLock()
    scan_futures: dict[str, Future[DatasetScanSession]] = {}
    annotation_store = AnnotationStore(dataset_repository, active_settings.data_dir / "backups" / "annotations")
    image_service = ImageService(dataset_repository, active_settings.data_dir / "cache" / "images")
    tile_service = DeepZoomTileService(dataset_repository, active_settings.data_dir / "cache" / "tiles")
    image_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="labelone-image")
    operator_packages = OperatorPackageManager(active_settings.data_dir / "operators")
    operator_warnings = operator_packages.load_installed()
    registered_package_kinds: set[str] = set()
    for package in operator_packages.list():
        try:
            register_operator_contracts([package.contract])
            registered_package_kinds.add(package.contract.kind)
        except ValueError as exc:
            operator_warnings.append(f"{package.contract.kind}: {exc}")
    pipeline_engine = PipelineEngine(
        dataset_repository,
        annotation_store,
        active_settings.data_dir / "artifacts" / "pipelines",
        operator_packages,
        manager,
        artifact_store,
    )
    composite_store = CompositeDefinitionStore(active_settings.data_dir / "pipeline-composites.json")
    composite_registry = CompositeRegistry()
    composite_registry_lock = RLock()
    composite_warnings: list[str] = []
    try:
        stored_composites = composite_store.load()
    except LabelOneError as exc:
        stored_composites = []
        composite_warnings.append(exc.message)
    for definition in stored_composites:
        try:
            composite_registry.register(definition)
        except LabelOneError as exc:
            composite_warnings.append(f"{definition.get('id', 'unknown')}: {exc.message}")
    job_repository = JobRepository(active_settings.data_dir / "index.sqlite3", dataset_repository)
    job_service = JobService(job_repository, dataset_repository, pipeline_engine, manager, annotation_store)
    agent_repository = AgentRepository(active_settings.data_dir / "index.sqlite3")
    agent_service = AgentService(
        agent_repository,
        dataset_repository,
        annotation_store,
        job_service,
        manager,
        CloudAgentPlanner(application_settings),
    )

    async def run_image(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(image_executor, partial(func, *args, **kwargs))

    def dispatch_scan(session_id: str) -> bool:
        with scan_dispatch_lock:
            current = scan_futures.get(session_id)
            if current is not None and not current.done():
                return False
            future = scan_executor.submit(scan_session_store.run, session_id)
            scan_futures[session_id] = future

        def forget(completed: Future[DatasetScanSession]) -> None:
            with scan_dispatch_lock:
                if scan_futures.get(session_id) is completed:
                    scan_futures.pop(session_id, None)

        future.add_done_callback(forget)
        return True

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        x_anylabeling_root = model_sources.x_anylabeling_root() or active_settings.x_anylabeling_root
        if x_anylabeling_root:
            try:
                await asyncio.to_thread(catalog.merge_x_anylabeling, x_anylabeling_root)
            except LabelOneError:
                pass
        # Recovered model-download jobs resolve their catalog entry as soon as
        # workers start, so restore the persisted catalog before resuming jobs.
        job_service.start()
        yield
        job_service.close()
        manager.close_all()
        image_executor.shutdown(wait=True, cancel_futures=True)
        scan_executor.shutdown(wait=True, cancel_futures=True)
        agent_repository.close()
        job_repository.close()
        scan_session_store.close()
        dataset_repository.close()
        unregister_operator_contracts(registered_package_kinds)

    app = FastAPI(
        title="LabelOne Local API",
        version=__version__,
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 424: {"model": ErrorResponse}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key", "If-Match", "If-None-Match", "If-Range", "Last-Event-ID", "Range"],
        expose_headers=["ETag", "Location", "Accept-Ranges", "Content-Range", "X-LabelOne-Cache", "X-LabelOne-Tile-Backend"],
    )

    @app.exception_handler(LabelOneError)
    async def handle_labelone_error(_: Request, exc: LabelOneError) -> JSONResponse:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = 424 if isinstance(exc, ModelRuntimeError) else 412 if isinstance(exc, RevisionConflictError) else 422 if isinstance(exc, AnnotationValidationError) else 400
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(code=exc.code, message=exc.message, details=exc.details).model_dump(mode="json"),
        )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        try:
            import onnxruntime  # noqa: F401
            onnx_state = "available"
        except ImportError:
            onnx_state = "unavailable"
        response = catalog.list()
        return HealthResponse(
            version=__version__,
            status="ok" if onnx_state == "available" else "degraded",
            model_registry={
                "configs": len(response.models),
                "adapters": len({model.adapter for model in response.models if model.capabilities.predict}),
                "errors": len(response.warnings),
            },
            runtimes={"onnxruntime": onnx_state},
        )

    def current_application_settings() -> ApplicationSettingsResponse:
        persisted = application_settings.model_weights_dir()
        if active_settings.model_weights_dir is not None:
            configured = active_settings.model_weights_dir.expanduser().resolve()
            managed_by = "environment"
        elif persisted is not None:
            configured = persisted
            managed_by = "persisted"
        else:
            configured = active_settings.data_dir.expanduser().resolve() / "model-weights"
            managed_by = "default"
        cloud_ai = application_settings.cloud_ai()
        network_proxy = application_settings.network_proxy()
        credential_configured = bool(os.getenv(str(cloud_ai["api_key_env"])))
        return ApplicationSettingsResponse(
            data_dir=active_settings.data_dir.expanduser().resolve(),
            model_source_dir=model_sources.x_anylabeling_root() or active_settings.x_anylabeling_root,
            model_weights_dir=configured,
            effective_model_weights_dir=weight_store.root,
            model_weights_managed_by=managed_by,
            restart_required=configured != weight_store.root,
            model_download_concurrency=4,
            model_download_source=application_settings.model_download_source(),
            model_download_sources=MODEL_DOWNLOAD_SOURCES,
            network_proxy=network_proxy,
            network_proxy_restart_required=network_proxy != effective_network_proxy,
            cloud_ai={
                **cloud_ai,
                "credential_configured": credential_configured,
                "credential_source": "environment" if credential_configured else "missing",
            },
            workspace=application_settings.workspace(),
            model_usage=application_settings.model_usage(),
        )

    def validate_workspace_pipeline(pipeline: WorkspacePipelineSettings | None) -> None:
        if pipeline is None:
            return
        visualizations_by_tap: dict[str, list[object]] = {}
        node_ids = {node.id for node in pipeline.nodes}
        for visualization in pipeline.visualizations:
            if visualization.tap_after_node_id not in node_ids:
                raise InvalidPathError(
                    "Workspace visualization references an unknown pipeline node",
                    details={"visualization_id": visualization.id, "tap_after_node_id": visualization.tap_after_node_id},
                )
            visualizations_by_tap.setdefault(visualization.tap_after_node_id, []).append(visualization)
        ordered_nodes: list[dict[str, object]] = []
        for node in pipeline.nodes:
            ordered_nodes.append({"id": node.id, "kind": node.kind, "enabled": node.enabled, "parameters": node.parameters})
            ordered_nodes.extend(
                {"id": visualization.id, "kind": visualization.kind, "enabled": visualization.enabled, "parameters": visualization.parameters}
                for visualization in visualizations_by_tap.get(node.id, [])
            )
        validate_pipeline_definition(ordered_nodes)

    @app.get("/api/v1/settings", response_model=ApplicationSettingsResponse)
    async def get_application_settings() -> ApplicationSettingsResponse:
        return current_application_settings()

    @app.patch("/api/v1/settings", response_model=ApplicationSettingsResponse)
    async def update_application_settings(request: ApplicationSettingsUpdate) -> ApplicationSettingsResponse:
        if request.model_weights_dir is None and request.model_download_source is None and request.network_proxy is None and request.cloud_ai is None and request.workspace is None:
            raise HTTPException(status_code=400, detail={"code": "empty_settings_update", "message": "No settings were provided"})
        if request.model_weights_dir is not None and active_settings.model_weights_dir is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "settings_managed_by_environment",
                    "message": "Model download directory is managed by LABELONE_MODEL_WEIGHTS_DIR",
                },
            )
        if request.model_weights_dir is not None:
            await asyncio.to_thread(application_settings.set_model_weights_dir, request.model_weights_dir)
        if request.model_download_source is not None:
            await asyncio.to_thread(application_settings.set_model_download_source, request.model_download_source)
        if request.network_proxy is not None:
            await asyncio.to_thread(application_settings.set_network_proxy, **request.network_proxy.model_dump())
        if request.cloud_ai is not None:
            await asyncio.to_thread(application_settings.set_cloud_ai, **request.cloud_ai.model_dump())
        if request.workspace is not None:
            validate_workspace_pipeline(request.workspace.pipeline)
            await asyncio.to_thread(application_settings.set_workspace, request.workspace)
        return current_application_settings()

    @app.post("/api/v1/models/{model_id}/usage", response_model=ModelUsageRecord)
    async def record_model_usage(model_id: str) -> ModelUsageRecord:
        catalog.get(model_id)
        return await asyncio.to_thread(application_settings.record_model_usage, model_id)

    @app.post("/api/v1/system/pick-directory", response_model=DirectoryPickerResponse)
    async def system_pick_directory(request: DirectoryPickerRequest) -> DirectoryPickerResponse:
        selected = await asyncio.to_thread(pick_directory, request.title, request.initial_dir)
        return DirectoryPickerResponse(path=selected, canceled=selected is None)

    @app.post("/api/v1/datasets/scan", response_model=DatasetScanResult)
    async def scan(request: DatasetScanRequest) -> DatasetScanResult:
        return await asyncio.to_thread(scan_dataset, request)

    @app.post("/api/v1/datasets/register", response_model=DatasetScanResult)
    async def register_dataset(request: DatasetScanRequest, name: str | None = Query(default=None, max_length=160)) -> DatasetScanResult:
        result = await asyncio.to_thread(scan_dataset, request)
        await asyncio.to_thread(dataset_repository.register, result, name=name)
        return result

    @app.post("/api/v1/dataset-scan-sessions", response_model=DatasetScanSession, status_code=202)
    async def create_dataset_scan_session(
        request: DatasetScanRequest,
        response: Response,
    ) -> DatasetScanSession:
        session = await asyncio.to_thread(scan_session_store.create, request)
        dispatch_scan(session.session_id)
        response.headers["Location"] = f"/api/v1/dataset-scan-sessions/{session.session_id}"
        return session

    @app.get("/api/v1/dataset-scan-sessions", response_model=DatasetScanSessionList)
    async def list_dataset_scan_sessions(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> DatasetScanSessionList:
        return await asyncio.to_thread(scan_session_store.list, limit=limit)

    @app.get("/api/v1/dataset-scan-sessions/{session_id}", response_model=DatasetScanSession)
    async def get_dataset_scan_session(session_id: str) -> DatasetScanSession:
        return await asyncio.to_thread(scan_session_store.get, session_id)

    @app.get("/api/v1/dataset-scan-sessions/{session_id}/items", response_model=DatasetScanItemPage)
    async def list_dataset_scan_session_items(
        session_id: str,
        after: int = Query(default=-1, ge=-1),
        limit: int = Query(default=200, ge=1, le=1000),
        status: AssetStatus | None = None,
    ) -> DatasetScanItemPage:
        return await asyncio.to_thread(
            scan_session_store.list_items,
            session_id,
            after_sequence=after,
            limit=limit,
            status=status,
        )

    @app.post("/api/v1/dataset-scan-sessions/{session_id}/run", response_model=DatasetScanSession, status_code=202)
    async def run_dataset_scan_session(
        session_id: str,
        response: Response,
    ) -> DatasetScanSession:
        session = await asyncio.to_thread(scan_session_store.get, session_id)
        if session.state not in {"queued", "failed", "interrupted"}:
            raise HTTPException(
                status_code=409,
                detail={"code": "scan_session_not_runnable", "message": f"Scan session is {session.state}"},
            )
        if not dispatch_scan(session_id):
            raise HTTPException(
                status_code=409,
                detail={"code": "scan_session_already_running", "message": "Scan session is already running"},
            )
        response.headers["Location"] = f"/api/v1/dataset-scan-sessions/{session_id}"
        return session

    @app.post("/api/v1/dataset-scan-sessions/{session_id}/interrupt", response_model=DatasetScanSession)
    async def interrupt_dataset_scan_session(session_id: str) -> DatasetScanSession:
        return await asyncio.to_thread(scan_session_store.interrupt, session_id)

    @app.post("/api/v1/dataset-scan-sessions/{session_id}/register", response_model=RegisteredDataset)
    async def register_dataset_scan_session(
        session_id: str,
        name: str | None = Query(default=None, min_length=1, max_length=160),
    ) -> RegisteredDataset:
        return await asyncio.to_thread(
            scan_session_store.register,
            session_id,
            dataset_repository,
            name=name,
        )

    @app.get("/api/v1/datasets", response_model=DatasetListResponse)
    async def list_datasets() -> DatasetListResponse:
        return dataset_repository.list_datasets()

    @app.delete("/api/v1/datasets/{dataset_id}", status_code=204)
    async def unregister_dataset(dataset_id: str, cancel_active_jobs: bool = False) -> Response:
        active_job_ids = await asyncio.to_thread(job_repository.active_dataset_job_ids, dataset_id)
        if active_job_ids and cancel_active_jobs:
            for job_id in active_job_ids:
                await asyncio.to_thread(job_service.cancel, job_id)
            for _ in range(50):
                active_job_ids = await asyncio.to_thread(job_repository.active_dataset_job_ids, dataset_id)
                if not active_job_ids:
                    break
                await asyncio.sleep(0.1)
        if active_job_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_has_active_jobs",
                    "message": "项目仍有关联的未完成后台任务，请先取消任务后再移除",
                    "details": {"dataset_id": dataset_id, "job_ids": active_job_ids},
                },
            )
        await asyncio.to_thread(dataset_repository.delete_dataset, dataset_id)
        return Response(status_code=204)

    @app.get("/api/v1/datasets/{dataset_id}/assets", response_model=AssetListResponse)
    async def list_dataset_assets(dataset_id: str, offset: int = 0, limit: int = 200) -> AssetListResponse:
        return dataset_repository.list_assets(dataset_id, offset=offset, limit=limit)

    @app.get(
        "/api/v1/datasets/{dataset_id}/settings",
        response_model=DatasetWorkspaceSettingsResponse,
    )
    async def get_dataset_settings(dataset_id: str) -> DatasetWorkspaceSettingsResponse:
        return await asyncio.to_thread(dataset_repository.get_workspace_settings, dataset_id)

    @app.put(
        "/api/v1/datasets/{dataset_id}/settings",
        response_model=DatasetWorkspaceSettingsResponse,
    )
    async def put_dataset_settings(
        dataset_id: str,
        request: DatasetWorkspaceSettingsUpdate,
    ) -> DatasetWorkspaceSettingsResponse:
        settings = DatasetWorkspaceSettings.model_validate(request.model_dump(exclude={"expected_revision"}))
        validate_workspace_pipeline(settings.pipeline)
        return await asyncio.to_thread(
            dataset_repository.set_workspace_settings,
            dataset_id,
            settings,
            expected_revision=request.expected_revision,
        )

    @app.get("/api/v1/datasets/{dataset_id}/assets-cursor", response_model=AssetCursorPage)
    async def list_dataset_assets_cursor(
        dataset_id: str,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> AssetCursorPage:
        return await asyncio.to_thread(
            dataset_repository.list_assets_cursor,
            dataset_id,
            cursor=cursor,
            limit=limit,
        )

    @app.get("/api/v1/datasets/{dataset_id}/search", response_model=AssetListResponse)
    async def search_dataset_assets(
        dataset_id: str,
        q: str = Query(default="", max_length=1024),
        mode: str = Query(default="smart", pattern="^(text|regex|condition|smart)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
        status: str | None = None,
        annotated: bool | None = None,
        has_annotation_file: bool | None = None,
    ) -> AssetListResponse:
        return await asyncio.to_thread(
            dataset_repository.search_assets,
            dataset_id,
            query=q,
            mode=mode,
            offset=offset,
            limit=limit,
            status=status,
            annotated=annotated,
            has_annotation_file=has_annotation_file,
        )

    @app.get("/api/v1/datasets/{dataset_id}/search-cursor", response_model=AssetCursorPage)
    async def search_dataset_assets_cursor(
        dataset_id: str,
        q: str = Query(default="", max_length=1024),
        mode: str = Query(default="smart", pattern="^(text|regex|condition|smart)$"),
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=200, ge=1, le=1000),
        status: str | None = None,
        annotated: bool | None = None,
        has_annotation_file: bool | None = None,
    ) -> AssetCursorPage:
        return await asyncio.to_thread(
            dataset_repository.search_assets_cursor,
            dataset_id,
            query=q,
            mode=mode,
            cursor=cursor,
            limit=limit,
            status=status,
            annotated=annotated,
            has_annotation_file=has_annotation_file,
        )

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}", response_model=DatasetAsset)
    async def get_dataset_asset(dataset_id: str, asset_id: str) -> DatasetAsset:
        return dataset_repository.get_asset(dataset_id, asset_id)

    @app.post("/api/v1/datasets/{dataset_id}/assets/{asset_id}/revalidate", response_model=DatasetAsset)
    async def revalidate_dataset_asset(dataset_id: str, asset_id: str) -> DatasetAsset:
        return await asyncio.to_thread(revalidate_asset, dataset_repository, dataset_id, asset_id)

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/annotation", response_model=AnnotationEnvelope)
    async def get_annotation(dataset_id: str, asset_id: str, response: Response) -> AnnotationEnvelope:
        envelope = await asyncio.to_thread(annotation_store.load, dataset_id, asset_id)
        response.headers["ETag"] = f'"{envelope.revision}"'
        response.headers["Cache-Control"] = "no-cache"
        return envelope

    @app.put("/api/v1/datasets/{dataset_id}/assets/{asset_id}/annotation", response_model=AnnotationSaveResponse)
    async def save_annotation(
        dataset_id: str,
        asset_id: str,
        request: AnnotationSaveRequest,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> AnnotationSaveResponse:
        if if_match is None:
            raise HTTPException(
                status_code=428,
                detail={"code": "if_match_required", "message": "Annotation writes require If-Match"},
            )
        saved = await asyncio.to_thread(annotation_store.save, dataset_id, asset_id, request.document, if_match=if_match)
        response.headers["ETag"] = f'"{saved.revision}"'
        return saved

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/image")
    async def original_image(dataset_id: str, asset_id: str, request: Request):
        path = image_service.image_path(dataset_id, asset_id)
        etag = image_service.source_etag(path)
        if request.headers.get("if-none-match", "").strip('"') == etag:
            return Response(status_code=304, headers={"ETag": f'"{etag}"'})
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers={"ETag": f'"{etag}"', "Cache-Control": "private, max-age=0, must-revalidate"})

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/thumbnail")
    async def thumbnail(
        dataset_id: str,
        asset_id: str,
        request: Request,
        max_size: int = Query(default=256, ge=32, le=2048),
        format_name: str = Query(default="webp", alias="format"),
    ) -> Response:
        rendered = await run_image(image_service.thumbnail, dataset_id, asset_id, max_size=max_size, format_name=format_name)
        if request.headers.get("if-none-match", "").strip('"') == rendered.etag:
            return Response(status_code=304, headers={"ETag": f'"{rendered.etag}"'})
        return Response(
            content=rendered.content,
            media_type=rendered.media_type,
            headers={"ETag": f'"{rendered.etag}"', "Cache-Control": "private, no-cache", "X-LabelOne-Cache": "hit" if rendered.cache_hit else "miss"},
        )

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/region")
    async def image_region(
        dataset_id: str,
        asset_id: str,
        request: Request,
        x: int = Query(ge=0),
        y: int = Query(ge=0),
        width: int = Query(gt=0),
        height: int = Query(gt=0),
        scale: float = Query(default=1.0, gt=0, le=4),
        format_name: str = Query(default="webp", alias="format"),
    ) -> Response:
        rendered = await run_image(
            image_service.region,
            dataset_id,
            asset_id,
            x=x,
            y=y,
            width=width,
            height=height,
            scale=scale,
            format_name=format_name,
        )
        if request.headers.get("if-none-match", "").strip('"') == rendered.etag:
            return Response(status_code=304, headers={"ETag": f'"{rendered.etag}"'})
        return Response(
            content=rendered.content,
            media_type=rendered.media_type,
            headers={"ETag": f'"{rendered.etag}"', "Cache-Control": "private, no-cache", "X-LabelOne-Cache": "hit" if rendered.cache_hit else "miss"},
        )

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/tiles/metadata")
    async def tile_metadata(
        dataset_id: str,
        asset_id: str,
        format_name: str = Query(default="webp", alias="format"),
    ) -> dict[str, object]:
        metadata = await run_image(tile_service.metadata, dataset_id, asset_id, format_name=format_name)
        return {
            "width": metadata.width,
            "height": metadata.height,
            "tile_size": metadata.tile_size,
            "max_level": metadata.max_level,
            "format": metadata.format,
            "source_etag": metadata.source_etag,
            "backend": metadata.backend,
            "source_format": metadata.source_format,
        }

    @app.get("/api/v1/datasets/{dataset_id}/assets/{asset_id}/tiles/{level}/{x}/{y}")
    async def image_tile(
        dataset_id: str,
        asset_id: str,
        level: int,
        x: int,
        y: int,
        request: Request,
        format_name: str = Query(default="webp", alias="format"),
    ) -> Response:
        rendered = await run_image(
            tile_service.tile,
            dataset_id,
            asset_id,
            level=level,
            x=x,
            y=y,
            format_name=format_name,
        )
        headers = {
            "ETag": f'"{rendered.etag}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-LabelOne-Cache": "hit" if rendered.cache_hit else "miss",
            "X-LabelOne-Tile-Backend": rendered.backend,
        }
        if request.headers.get("if-none-match", "").strip('"') == rendered.etag:
            return Response(status_code=304, headers=headers)
        return Response(content=rendered.content, media_type=rendered.media_type, headers=headers)

    @app.post("/api/v1/pipelines/validate", response_model=PipelineValidationResult)
    async def pipeline_validate(request: PipelineValidationRequest) -> PipelineValidationResult:
        validated = validate_pipeline_definition(
            request.nodes,
            mode=request.mode,
            width=request.width,
            height=request.height,
        )
        return PipelineValidationResult(
            registry_hash=operator_registry_hash(),
            normalized_nodes=[node.as_dict() for node in validated.nodes],
            transform_count=validated.transform_count,
            visualization_count=validated.visualization_count,
            output_width=validated.output_width,
            output_height=validated.output_height,
        )

    @app.post("/api/v1/pipelines/preview", response_model=PipelinePreviewResult)
    async def pipeline_preview(payload: PipelinePreviewRequest, http_request: Request) -> PipelinePreviewResult:
        disconnected = ThreadEvent()
        callback = partial(pipeline_engine.preview, payload, canceled=disconnected.is_set)
        run_adhoc = getattr(job_service, "run_adhoc", None)
        if callable(run_adhoc):
            worker = asyncio.create_task(asyncio.to_thread(
                run_adhoc,
                "cpu_pipeline",
                callback,
                priority=payload.priority,
                canceled=disconnected.is_set,
            ))
        else:
            worker = asyncio.create_task(asyncio.to_thread(
                job_service.run_interactive,
                "cpu_pipeline",
                callback,
            ))
        try:
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(worker), timeout=0.05)
                except asyncio.TimeoutError:
                    if await http_request.is_disconnected():
                        disconnected.set()
        finally:
            disconnected.set()

    @app.get("/api/v1/pipelines/operators")
    async def pipeline_operators() -> dict[str, object]:
        installed = operator_packages.list()
        installed_kinds = {package.contract.kind for package in installed}
        with composite_registry_lock:
            composites = [
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "version_hash": item.version_hash,
                }
                for item in composite_registry.list()
            ]
        return {
            "registry_hash": operator_registry_hash(),
            "operators": [
                {
                    **operator,
                    "source": "custom" if operator["kind"] in installed_kinds else "opencv" if str(operator["kind"]).startswith("opencv.") else "builtin",
                }
                for operator in operator_catalog()
            ],
            "installed_packages": [
                {
                    "kind": package.contract.kind,
                    "title": package.contract.title,
                    "version": package.contract.version,
                    "digest": package.digest,
                    "package_dir": str(package.package_dir),
                    "entrypoint": package.entrypoint,
                    "annotation_entrypoint": package.annotation_entrypoint,
                    "annotation_policy": package.annotation_mode,
                    "size_behavior": package.contract.size_behavior,
                    "trusted_local_code": True,
                    "is_os_sandboxed": False,
                }
                for package in installed
            ],
            "composites": composites,
            "warnings": [*operator_warnings, *composite_warnings],
        }

    @app.post("/api/v1/pipelines/operators/inspect")
    async def inspect_pipeline_operator(
        request: Request,
        filename: str = Query(default="operator.zip", min_length=5, max_length=255),
    ) -> dict[str, object]:
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().casefold()
        if content_type not in {"application/zip", "application/octet-stream"}:
            raise HTTPException(
                status_code=415,
                detail={"code": "unsupported_media_type", "message": "Operator inspection requires application/zip"},
            )
        inspected = await asyncio.to_thread(operator_packages.inspect_zip, await request.body(), filename=filename)
        return {
            "operator": inspected.contract.as_dict(),
            "digest": inspected.digest,
            "entrypoint": inspected.entrypoint,
            "annotation_entrypoint": inspected.annotation_entrypoint,
            "filename": inspected.filename,
            "annotation_policy": inspected.annotation_mode,
            "will_execute_local_code": True,
            "is_os_sandboxed": False,
        }

    @app.post("/api/v1/pipelines/operators/import", status_code=201)
    async def import_pipeline_operator(
        request: Request,
        filename: str = Query(default="operator.zip", min_length=5, max_length=255),
    ) -> dict[str, object]:
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().casefold()
        if content_type not in {"application/zip", "application/octet-stream"}:
            raise HTTPException(
                status_code=415,
                detail={"code": "unsupported_media_type", "message": "Operator import requires application/zip"},
            )
        content = await request.body()
        package = await asyncio.to_thread(operator_packages.install_zip, content, filename=filename)
        try:
            register_operator_contracts([package.contract])
            registered_package_kinds.add(package.contract.kind)
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "operator_conflict", "message": str(exc)},
            ) from exc
        return {
            "operator": package.contract.as_dict(),
            "digest": package.digest,
            "package_dir": str(package.package_dir),
            "trusted_local_code": True,
            "is_os_sandboxed": False,
        }

    @app.post("/api/v1/pipelines/composites", status_code=201)
    async def create_pipeline_composite(definition: dict[str, object]) -> dict[str, object]:
        with composite_registry_lock:
            composite = composite_registry.register(definition)
            try:
                composite_store.append(definition)
            except Exception:
                composite_registry.remove(composite.id)
                raise
        return {
            "id": composite.id,
            "name": composite.name,
            "description": composite.description,
            "version_hash": composite.version_hash,
        }

    @app.get("/api/v1/pipelines/composites/{composite_id}/expand")
    async def expand_pipeline_composite(
        composite_id: str,
        width: int = Query(gt=0),
        height: int = Query(gt=0),
    ) -> dict[str, object]:
        with composite_registry_lock:
            expanded = composite_registry.expand(composite_id, input_width=width, input_height=height)
        return {
            "id": expanded.composite_id,
            "version_hash": expanded.version_hash,
            "output_width": expanded.output_width,
            "output_height": expanded.output_height,
            "nodes": [node.as_dict() for node in expanded.nodes],
        }

    @app.get("/api/v1/pipeline-artifacts/{artifact_id}")
    async def pipeline_artifact(artifact_id: str) -> FileResponse:
        path, media_type = pipeline_engine.artifact_path(artifact_id)
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.post("/api/v1/jobs", response_model=JobRecord, status_code=202)
    async def create_job(
        request: BatchJobRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobRecord:
        if not idempotency_key:
            raise HTTPException(status_code=428, detail={"code": "idempotency_key_required", "message": "Job creation requires Idempotency-Key"})
        job = await asyncio.to_thread(job_service.create, request, idempotency_key=idempotency_key)
        response.headers["Location"] = f"/api/v1/jobs/{job.job_id}"
        return job

    @app.post("/api/v1/pipelines/precompute/ensure", response_model=PipelinePrecomputeEnsureResponse)
    async def ensure_pipeline_precompute(request: BatchJobRequest) -> PipelinePrecomputeEnsureResponse:
        return await asyncio.to_thread(job_service.ensure_pipeline_precompute, request)

    @app.get("/api/v1/jobs", response_model=JobListResponse)
    async def list_jobs(limit: int = Query(default=100, ge=1, le=500)) -> JobListResponse:
        return await asyncio.to_thread(job_repository.list, limit)

    @app.get("/api/v1/jobs-scheduler")
    async def jobs_scheduler() -> dict[str, object]:
        snapshot = await asyncio.to_thread(job_service.scheduler_snapshot)
        return {
            "queued": snapshot.queued,
            "inflight": snapshot.inflight,
            "global_capacity": snapshot.global_capacity,
            "lane_inflight": snapshot.lane_inflight,
            "lane_capacities": snapshot.lane_capacities,
            "lane_interactive_reserves": snapshot.lane_interactive_reserves,
            "job_queued": snapshot.job_queued,
            "job_inflight": snapshot.job_inflight,
            "closed": snapshot.closed,
        }

    @app.get("/api/v1/jobs/{job_id}", response_model=JobRecord)
    async def get_job(job_id: str, include_items: bool = False) -> JobRecord:
        return await asyncio.to_thread(job_repository.get, job_id, include_items=include_items)

    @app.post("/api/v1/jobs/{job_id}/prioritize", response_model=JobRecord)
    async def prioritize_job_items(job_id: str, request: JobPriorityRequest) -> JobRecord:
        return await asyncio.to_thread(
            job_repository.prioritize_queued_items,
            job_id,
            request.asset_ids,
        )

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        after: str | None = Query(default=None),
        format_name: str = Query(default="sse", alias="format"),
        limit: int = Query(default=200, ge=1, le=1000),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        try:
            cursor = resolve_event_cursor(after, last_event_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_event_cursor", "message": str(exc)},
            ) from exc
        await asyncio.to_thread(job_repository.get, job_id, include_items=False)
        normalized_format = format_name.casefold()
        if normalized_format == "json":
            page = await asyncio.to_thread(job_repository.list_events, job_id, after=cursor, limit=limit)
            return event_page_payload(job_id, cursor, page)
        if normalized_format != "sse":
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_event_format", "message": "format must be sse or json"},
            )
        return StreamingResponse(
            stream_job_events(request, job_repository, job_id=job_id, after=cursor, limit=limit),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/jobs/{job_id}/items", response_model=JobItemListResponse)
    async def list_job_items(
        job_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
        state: str | None = None,
    ) -> JobItemListResponse:
        return await asyncio.to_thread(
            job_repository.list_items,
            job_id,
            offset=offset,
            limit=limit,
            state=state,
        )

    @app.post("/api/v1/jobs/{job_id}/items/lookup", response_model=JobItemListResponse)
    async def lookup_job_items(job_id: str, request: JobItemLookupRequest) -> JobItemListResponse:
        return await asyncio.to_thread(job_repository.lookup_items, job_id, request.asset_ids)

    @app.post("/api/v1/jobs/{job_id}/pause", response_model=JobRecord)
    async def pause_job(job_id: str) -> JobRecord:
        return await asyncio.to_thread(job_service.pause, job_id)

    @app.post("/api/v1/jobs/{job_id}/resume", response_model=JobRecord)
    async def resume_job(job_id: str) -> JobRecord:
        return await asyncio.to_thread(job_service.resume, job_id)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobRecord)
    async def cancel_job(job_id: str) -> JobRecord:
        return await asyncio.to_thread(job_service.cancel, job_id)

    @app.get("/api/v1/agent/status", response_model=AgentStatus)
    async def agent_status() -> AgentStatus:
        return await asyncio.to_thread(agent_service.status)

    @app.post("/api/v1/agent/runs", response_model=AgentRun)
    async def create_agent_run(request: AgentRunRequest) -> AgentRun:
        await asyncio.to_thread(agent_service.require_backend)
        return await asyncio.to_thread(agent_service.run, request)

    @app.post("/api/v1/agent/runs/{run_id}/proposals/{proposal_id}/execute", response_model=AgentRun)
    async def execute_agent_proposal(run_id: str, proposal_id: str) -> AgentRun:
        await asyncio.to_thread(agent_service.require_backend)
        return await asyncio.to_thread(agent_service.execute, run_id, proposal_id)

    @app.get("/api/v1/agent/runs/{run_id}/audit", response_model=list[AgentAuditRecord])
    async def agent_run_audit(
        run_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[AgentAuditRecord]:
        await asyncio.to_thread(agent_repository.get, run_id)
        return await asyncio.to_thread(agent_repository.list_audit, run_id, after=after, limit=limit)

    @app.post("/api/v1/model-sources/x-anylabeling/import", response_model=ModelCatalogResponse)
    async def import_x_anylabeling(request: ImportCatalogRequest) -> ModelCatalogResponse:
        # Catalog replacement must not retain supervisors created from the
        # previous source tree; unloading also releases model memory/processes.
        for descriptor in catalog.list().models:
            await asyncio.to_thread(manager.unload, descriptor.id)
        await asyncio.to_thread(catalog.import_x_anylabeling, builtin_model_root)
        await asyncio.to_thread(catalog.merge_x_anylabeling, request.root_dir)
        await asyncio.to_thread(model_sources.set_x_anylabeling_root, request.root_dir)
        return effective_catalog_response()

    def effective_catalog_response() -> ModelCatalogResponse:
        response = catalog.list()
        usage = application_settings.model_usage()
        descriptors = [weight_store.effective_record(catalog.get(model.id)).descriptor for model in response.models]
        status_by_model = {
            model.id: ModelCatalogStatus(
                runtime_state=manager.state(model.id).state,
                usage_count=usage.get(model.id, ModelUsageRecord()).count,
                last_used_at=usage.get(model.id, ModelUsageRecord()).last_used_at,
            )
            for model in descriptors
        }

        def readiness_rank(model) -> int:  # noqa: ANN001
            runtime_state = status_by_model[model.id].runtime_state
            if runtime_state == "loaded":
                return 0
            if runtime_state == "loading":
                return 1
            if runtime_state == "failed":
                return 4
            return {"available": 2, "missing_weights": 3}.get(model.availability.state, 4)

        descriptors.sort(key=lambda model: (model.task.casefold(), model.display_name.casefold(), model.id.casefold()))
        descriptors.sort(key=lambda model: status_by_model[model.id].last_used_at or "", reverse=True)
        descriptors.sort(key=lambda model: status_by_model[model.id].usage_count, reverse=True)
        descriptors.sort(key=readiness_rank)
        return ModelCatalogResponse(
            models=descriptors,
            warnings=response.warnings,
            status_by_model=status_by_model,
        )

    @app.get("/api/v1/models", response_model=ModelCatalogResponse)
    async def models() -> ModelCatalogResponse:
        return effective_catalog_response()

    @app.get("/api/v1/models/{model_id}/weights")
    async def model_weights(model_id: str) -> list[dict[str, object]]:
        record = catalog.get(model_id)
        weights = await asyncio.to_thread(
            weight_store.list_remote,
            model_id,
            record,
            application_settings.model_download_source(),
        )
        return [
            {
                "url_index": item.url_index,
                "url": item.url,
                "filename": item.filename,
                "downloaded": item.downloaded,
                "local_path": str(item.local_path) if item.local_path else None,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "source_id": item.source_id,
                "preferred": item.preferred,
            }
            for item in weights
        ]

    @app.post(
        "/api/v1/models/{model_id}/weights/download",
        response_model=JobRecord,
        status_code=202,
    )
    async def download_model_weights(
        model_id: str,
        request: ModelWeightDownloadRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobRecord:
        if not idempotency_key:
            raise HTTPException(
                status_code=428,
                detail={"code": "idempotency_key_required", "message": "Model weight download requires Idempotency-Key"},
            )
        job = await asyncio.to_thread(
            job_service.create_model_download,
            model_id,
            request.url_indices,
            expected_sha256=request.expected_sha256,
            idempotency_key=idempotency_key,
        )
        response.headers["Location"] = f"/api/v1/jobs/{job.job_id}"
        return job

    @app.post(
        "/api/v1/models/{model_id}/weights/{url_index}/download",
        response_model=JobRecord,
        status_code=202,
    )
    async def download_model_weight(
        model_id: str,
        url_index: int,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobRecord:
        if not idempotency_key:
            raise HTTPException(
                status_code=428,
                detail={
                    "code": "idempotency_key_required",
                    "message": "Model weight download requires Idempotency-Key",
                },
            )
        job = await asyncio.to_thread(
            job_service.create_model_download,
            model_id,
            [url_index],
            idempotency_key=idempotency_key,
        )
        response.headers["Location"] = f"/api/v1/jobs/{job.job_id}"
        return job

    @app.post("/api/v1/models/{model_id}/load", response_model=ModelRuntimeState)
    async def load_model(model_id: str, request: ModelLoadRequest) -> ModelRuntimeState:
        return await asyncio.to_thread(manager.load, model_id, request.providers)

    @app.delete("/api/v1/models/{model_id}", response_model=ModelRuntimeState)
    async def unload_model(model_id: str) -> ModelRuntimeState:
        return await asyncio.to_thread(manager.unload, model_id)

    @app.get("/api/v1/models/{model_id}/layers", response_model=ModelRuntimeState)
    async def model_layers(model_id: str) -> ModelRuntimeState:
        return await asyncio.to_thread(manager.layers, model_id)

    @app.post("/api/v1/inference-runs", response_model=InferenceResult)
    async def inference(request: InferenceRequest) -> InferenceResult:
        return await asyncio.to_thread(
            job_service.run_interactive,
            f"model:{request.model_id}",
            partial(
                manager.predict,
                request.model_id,
                request.image_path,
                request.capture_layers,
                request.parameters,
            ),
        )

    @app.get("/api/v1/artifacts/{artifact_id}")
    async def artifact_manifest(artifact_id: str) -> dict[str, object]:
        try:
            return artifact_store.get_manifest(artifact_id)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"code": "artifact_not_found", "message": str(exc)})

    @app.get("/api/v1/artifacts/{artifact_id}/content")
    async def artifact_content(artifact_id: str) -> Response:
        try:
            path, media_type = artifact_store.content_path(artifact_id)
            return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"code": "artifact_not_found", "message": str(exc)})

    @app.get("/api/v1/artifacts/{artifact_id}/preview")
    async def artifact_preview(artifact_id: str) -> Response:
        try:
            path, media_type = artifact_store.preview_path(artifact_id)
            return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"code": "artifact_preview_not_found", "message": str(exc)})

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "labelone.main:app",
        host=default_settings.host,
        port=default_settings.port,
        reload=False,
        access_log=os.getenv("LABELONE_ACCESS_LOG", "0") == "1",
    )


if __name__ == "__main__":
    run()
