'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import type {
  AgentRunInput,
  AgentRunResult,
  AgentStatus,
  ApiError,
  AnnotationEnvelope,
  AnnotationSaveResponse,
  ApplicationSettings,
  ApplicationSettingsUpdate,
  BackendMode,
  DatasetAssetSearchInput,
  DatasetScanInput,
  DatasetScanItem,
  DatasetScanItemPage,
  DatasetScanSession,
  DatasetScanSessionView,
  DatasetScanResult,
  DatasetListResponse,
  DatasetWorkspaceSettingsResponse,
  DatasetWorkspaceSettingsUpdate,
  DirectoryPickerInput,
  DirectoryPickerResult,
  AssetCursorPage,
  HealthResponse,
  InferenceResult,
  BatchJobRequest,
  JobItem,
  JobItemListResponse,
  JobListResponse,
  JobRecord,
  JobItemProgress,
  ModelCatalogResponse,
  ModelRuntimeState,
  ModelWeightItem,
  PipelinePreviewResult,
  PipelinePrecomputeEnsureResponse,
  PipelineValidationResult,
  PipelineOperatorImportResponse,
  PipelineOperatorInspection,
  PipelineRegistryResponse,
  RemoteState,
  RegisteredDataset,
  TileMetadata,
} from '../lib/contracts';
import { mergeAssetCursorPage, summarizeScanItems } from '../lib/dataset-stream';
import { LocalApiError, localApiBase, localRequest } from '../lib/local-api';

const emptyModels: ModelCatalogResponse = { models: [], warnings: [], status_by_model: {} };
const ANNOTATION_CACHE_LIMIT = 24;
const activeJobStates = new Set(['queued', 'running', 'pausing', 'canceling']);
const jobItemStateValues = new Set(['queued', 'running', 'succeeded', 'failed', 'canceled']);
const jobEventTypes = ['snapshot', 'job.created', 'job.state', 'item.state', 'item.progress', 'job.progress', 'job.recovered', 'job.terminal'] as const;

function apiError(error: unknown): ApiError {
  if (error instanceof LocalApiError) return { code: error.code, message: error.message, details: error.details };
  return { code: 'unknown_error', message: error instanceof Error ? error.message : 'Unknown local API error' };
}

export function useLocalBackend() {
  const [mode, setMode] = useState<BackendMode>('probing');
  const [health, setHealth] = useState<RemoteState<HealthResponse | null>>({ phase: 'idle', data: null });
  const [applicationSettings, setApplicationSettings] = useState<RemoteState<ApplicationSettings | null>>({ phase: 'idle', data: null });
  const [agentStatus, setAgentStatus] = useState<RemoteState<AgentStatus | null>>({ phase: 'idle', data: null });
  const [models, setModels] = useState<RemoteState<ModelCatalogResponse>>({ phase: 'idle', data: emptyModels });
  const [runtime, setRuntime] = useState<RemoteState<ModelRuntimeState | null>>({ phase: 'idle', data: null });
  const [inference, setInference] = useState<RemoteState<InferenceResult | null>>({ phase: 'idle', data: null });
  const [scan, setScan] = useState<RemoteState<DatasetScanSessionView | null>>({ phase: 'idle', data: null });
  const [datasets, setDatasets] = useState<RemoteState<DatasetListResponse>>({ phase: 'idle', data: { datasets: [] } });
  const [annotation, setAnnotation] = useState<RemoteState<AnnotationEnvelope | null>>({ phase: 'idle', data: null });
  const [pipeline, setPipeline] = useState<RemoteState<PipelinePreviewResult | null>>({ phase: 'idle', data: null });
  const [pipelineValidation, setPipelineValidation] = useState<RemoteState<PipelineValidationResult | null>>({ phase: 'idle', data: null });
  const [pipelineRegistry, setPipelineRegistry] = useState<RemoteState<PipelineRegistryResponse | null>>({ phase: 'idle', data: null });
  const [jobs, setJobs] = useState<RemoteState<JobListResponse>>({ phase: 'idle', data: { jobs: [] } });
  const [jobItems, setJobItems] = useState<RemoteState<JobItemListResponse | null>>({ phase: 'idle', data: null });
  const [assetSearch, setAssetSearch] = useState<RemoteState<AssetCursorPage>>({ phase: 'idle', data: { items: [], total: 0, next_cursor: null, index_revision: 0 } });
  const [assetSearchLoadingMore, setAssetSearchLoadingMore] = useState(false);
  const [assetSearchDatasetId, setAssetSearchDatasetId] = useState<string | null>(null);
  const [preferredJobEventId, setPreferredJobEventId] = useState<string | null>(null);
  const [pageVisible, setPageVisible] = useState(() => typeof document === 'undefined' || !document.hidden);
  const [jobEvents, setJobEvents] = useState<{ mode: 'realtime' | 'connecting' | 'polling' | 'hidden' | 'offline'; job_id: string | null; last_event_id: string | null }>({ mode: 'offline', job_id: null, last_event_id: null });
  const [jobItemProgress, setJobItemProgress] = useState<Record<string, JobItemProgress>>({});
  const [jobItemSnapshots, setJobItemSnapshots] = useState<Record<string, JobItem>>({});
  const baseRef = useRef<string | null>(null);
  const healthController = useRef<AbortController | null>(null);
  const scanController = useRef<AbortController | null>(null);
  const scanItemsRef = useRef<DatasetScanItem[]>([]);
  const scanSummaryRef = useRef<DatasetScanResult['summary']>(summarizeScanItems([]));
  const scanAfterRef = useRef(-1);
  const scanGenerationRef = useRef(-1);
  const assetSearchController = useRef<AbortController | null>(null);
  const assetSearchRequestId = useRef(0);
  const assetSearchKey = useRef('');
  const annotationCache = useRef(new Map<string, AnnotationEnvelope>());
  const annotationInflight = useRef(new Map<string, Promise<AnnotationEnvelope>>());
  const pipelineController = useRef<AbortController | null>(null);
  const pipelineRequestId = useRef(0);
  const runtimeController = useRef<AbortController | null>(null);
  const runtimeRequestId = useRef(0);
  const inferenceController = useRef<AbortController | null>(null);
  const inferenceRequestId = useRef(0);
  const pipelineValidationController = useRef<AbortController | null>(null);
  const pipelineValidationRequestId = useRef(0);
  const pipelinePrefetchController = useRef<AbortController | null>(null);
  const pipelinePrefetchRequestId = useRef(0);
  const jobLastEventIds = useRef(new Map<string, string>());

  const refreshModels = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return;
    setModels((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<ModelCatalogResponse>(base, '/models', {}, 6000);
      setModels({ phase: 'ready', data: response });
    } catch (error) {
      setModels((old) => ({ phase: 'error', data: old.data, stale: old.data.models.length > 0, error: apiError(error) }));
    }
  }, []);

  const refreshAgentStatus = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return null;
    setAgentStatus((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<AgentStatus>(base, '/agent/status', {}, 6000);
      setAgentStatus({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      setAgentStatus((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    }
  }, []);

  const refreshApplicationSettings = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return null;
    setApplicationSettings((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<ApplicationSettings>(base, '/settings', {}, 6000);
      setApplicationSettings({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      setApplicationSettings((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    }
  }, []);

  const updateApplicationSettings = useCallback(async (input: ApplicationSettingsUpdate) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Application settings require the local service' });
    setApplicationSettings((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<ApplicationSettings>(base, '/settings', {
        method: 'PATCH',
        body: JSON.stringify(input),
      }, 15000);
      setApplicationSettings({ phase: 'ready', data: response });
      await refreshAgentStatus();
      return response;
    } catch (error) {
      setApplicationSettings((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    }
  }, [refreshAgentStatus]);

  const listModelWeights = useCallback(async (modelId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Model weight inspection requires the local service' });
    return localRequest<ModelWeightItem[]>(base, `/models/${encodeURIComponent(modelId)}/weights`, {}, 15000);
  }, []);

  const downloadModelWeight = useCallback(async (modelId: string, urlIndex: number) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Model weight download requires the local service' });
    const response = await localRequest<JobRecord>(base, `/models/${encodeURIComponent(modelId)}/weights/${urlIndex}/download`, {
      method: 'POST',
      headers: { 'Idempotency-Key': crypto.randomUUID() },
    }, 15000);
    setJobs((old) => ({ phase: 'ready', data: { jobs: [response, ...old.data.jobs.filter((job) => job.job_id !== response.job_id)] } }));
    return response;
  }, []);

  const refreshDatasets = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return;
    setDatasets((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<DatasetListResponse>(base, '/datasets', {}, 6000);
      setDatasets({ phase: 'ready', data: response });
    } catch (error) {
      setDatasets((old) => ({ phase: 'error', data: old.data, stale: old.data.datasets.length > 0, error: apiError(error) }));
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return;
    healthController.current?.abort();
    const controller = new AbortController();
    healthController.current = controller;
    setMode('probing');
    setHealth((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<HealthResponse>(base, '/health', { signal: controller.signal }, 1500);
      setMode('online');
      setHealth({ phase: 'ready', data: response });
      await Promise.all([refreshModels(), refreshDatasets(), refreshApplicationSettings(), refreshAgentStatus()]);
    } catch (error) {
      if (controller.signal.aborted) return;
      setMode('offline');
      setHealth((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
    }
  }, [refreshAgentStatus, refreshApplicationSettings, refreshDatasets, refreshModels]);

  const pollScanSession = useCallback(async (sessionId: string, controller: AbortController) => {
    const base = baseRef.current;
    if (!base) return;
    try {
      while (!controller.signal.aborted) {
        const session = await localRequest<DatasetScanSession>(base, `/dataset-scan-sessions/${encodeURIComponent(sessionId)}`, { signal: controller.signal }, 10000);
        if (controller.signal.aborted) return;
        if (session.run_generation !== scanGenerationRef.current) {
          scanGenerationRef.current = session.run_generation;
          scanItemsRef.current = [];
          scanSummaryRef.current = summarizeScanItems([]);
          scanAfterRef.current = -1;
        }
        const page = await localRequest<DatasetScanItemPage>(base, `/dataset-scan-sessions/${encodeURIComponent(sessionId)}/items?after=${scanAfterRef.current}&limit=500`, { signal: controller.signal }, 15000);
        if (controller.signal.aborted) return;
        if (page.items.length) {
          const seen = new Set(scanItemsRef.current.map((item) => item.asset_id));
          const fresh = page.items.filter((item) => !seen.has(item.asset_id));
          const pageSummary = summarizeScanItems(fresh);
          scanSummaryRef.current = {
            valid: scanSummaryRef.current.valid + pageSummary.valid,
            duplicate_match: scanSummaryRef.current.duplicate_match + pageSummary.duplicate_match,
            orphan_annotation: scanSummaryRef.current.orphan_annotation + pageSummary.orphan_annotation,
            corrupt_image: scanSummaryRef.current.corrupt_image + pageSummary.corrupt_image,
            corrupt_annotation: scanSummaryRef.current.corrupt_annotation + pageSummary.corrupt_annotation,
            hidden_image_only: 0,
          };
          scanItemsRef.current = [...scanItemsRef.current, ...fresh].slice(-200);
        }
        if (page.next_after !== null) scanAfterRef.current = page.next_after;
        else if (page.items.length) scanAfterRef.current += page.items.length;
        const view: DatasetScanSessionView = {
          ...session,
          items: scanItemsRef.current,
          next_after: page.next_after,
          streamed_summary: session.summary ?? scanSummaryRef.current,
        };
        const active = session.state === 'queued' || session.state === 'running';
        setScan({ phase: active ? 'loading' : 'ready', data: view });
        // A completed session can be registered immediately from SQLite; the
        // browser does not need to drain every persisted preview item first.
        if (!active) return;
        await new Promise<void>((resolve) => {
          const timer = window.setTimeout(resolve, 400);
          controller.signal.addEventListener('abort', () => { window.clearTimeout(timer); resolve(); }, { once: true });
        });
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      setScan((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
    }
  }, []);

  const beginScanPolling = useCallback((session: DatasetScanSession, reset: boolean) => {
    scanController.current?.abort();
    const controller = new AbortController();
    scanController.current = controller;
    if (reset) {
      scanItemsRef.current = [];
      scanSummaryRef.current = summarizeScanItems([]);
      scanAfterRef.current = -1;
      scanGenerationRef.current = session.run_generation;
    }
    setScan({
      phase: ['queued', 'running'].includes(session.state) ? 'loading' : 'ready',
      data: {
        ...session,
        items: [...scanItemsRef.current],
        next_after: scanAfterRef.current >= 0 ? scanAfterRef.current : null,
        streamed_summary: session.summary ?? scanSummaryRef.current,
      },
    });
    void pollScanSession(session.session_id, controller);
  }, [pollScanSession]);

  const startScan = useCallback(async (input: DatasetScanInput) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset scanning requires the local service' });
    setScan((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<DatasetScanSession>(base, '/dataset-scan-sessions', {
        method: 'POST',
        body: JSON.stringify(input),
      }, 15000);
      beginScanPolling(response, true);
      return response;
    } catch (error) {
      setScan((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    }
  }, [beginScanPolling]);

  const pickDirectory = useCallback(async (input: DirectoryPickerInput) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Directory picking requires the local service' });
    return localRequest<DirectoryPickerResult>(base, '/system/pick-directory', {
      method: 'POST',
      body: JSON.stringify(input),
    }, 10 * 60 * 1000);
  }, []);

  const interruptScan = useCallback(async () => {
    const base = baseRef.current;
    const sessionId = scan.data?.session_id;
    if (!base || !sessionId) throw new LocalApiError({ code: 'scan_session_missing', message: 'No scan session is selected' });
    const response = await localRequest<DatasetScanSession>(base, `/dataset-scan-sessions/${encodeURIComponent(sessionId)}/interrupt`, { method: 'POST' }, 10000);
    beginScanPolling(response, false);
    return response;
  }, [beginScanPolling, scan.data?.session_id]);

  const resumeScan = useCallback(async () => {
    const base = baseRef.current;
    const sessionId = scan.data?.session_id;
    if (!base || !sessionId) throw new LocalApiError({ code: 'scan_session_missing', message: 'No scan session is selected' });
    const response = await localRequest<DatasetScanSession>(base, `/dataset-scan-sessions/${encodeURIComponent(sessionId)}/run`, { method: 'POST' }, 10000);
    beginScanPolling(response, true);
    return response;
  }, [beginScanPolling, scan.data?.session_id]);

  const registerScan = useCallback(async (sessionId: string, name?: string) => {
    const base = baseRef.current;
    if (!base || !sessionId) throw new LocalApiError({ code: 'scan_session_missing', message: 'No scan session is selected' });
    const parameters = name?.trim() ? `?name=${encodeURIComponent(name.trim())}` : '';
    const response = await localRequest<RegisteredDataset>(base, `/dataset-scan-sessions/${encodeURIComponent(sessionId)}/register${parameters}`, { method: 'POST' }, 30000);
    setScan((old) => old.data ? {
      phase: ['queued', 'running'].includes(old.data.state) ? 'loading' : 'ready',
      data: {
        ...old.data,
        registration_name: response.name,
        registered_dataset_id: response.dataset_id,
        registered_index_revision: response.index_revision,
        registered_items: old.data.persisted_items,
        registered_at: response.updated_at,
      },
    } : old);
    await refreshDatasets();
    return response;
  }, [refreshDatasets]);

  const openRegisteredDataset = useCallback(async (datasetId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset loading requires the local service' });
    return localRequest<AssetCursorPage>(base, `/datasets/${encodeURIComponent(datasetId)}/assets-cursor?limit=100`, {}, 15000);
  }, []);

  const getDatasetSettings = useCallback(async (datasetId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset settings require the local service' });
    return localRequest<DatasetWorkspaceSettingsResponse>(base, `/datasets/${encodeURIComponent(datasetId)}/settings`, {}, 10000);
  }, []);

  const saveDatasetSettings = useCallback(async (datasetId: string, settings: DatasetWorkspaceSettingsUpdate) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset settings require the local service' });
    return localRequest<DatasetWorkspaceSettingsResponse>(base, `/datasets/${encodeURIComponent(datasetId)}/settings`, {
      method: 'PUT',
      body: JSON.stringify(settings),
    }, 15000);
  }, []);

  const getDatasetAsset = useCallback(async (datasetId: string, assetId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset asset loading requires the local service' });
    return localRequest<DatasetScanResult['items'][number]>(base, `/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}`, {}, 10000);
  }, []);

  const searchDatasetAssets = useCallback(async (datasetId: string, input: DatasetAssetSearchInput, append = false) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset search requires the local service' });
    assetSearchController.current?.abort();
    const controller = new AbortController();
    const requestId = ++assetSearchRequestId.current;
    assetSearchController.current = controller;
    const queryKey = JSON.stringify({ datasetId, q: input.q, mode: input.mode, status: input.status ?? null, annotated: input.annotated ?? null, hasAnnotationFile: input.has_annotation_file ?? null });
    if (append) setAssetSearchLoadingMore(true);
    else {
      assetSearchKey.current = queryKey;
      setAssetSearchDatasetId(datasetId);
      setAssetSearchLoadingMore(false);
      setAssetSearch((old) => ({ ...old, phase: 'loading', error: undefined, stale: old.data.items.length > 0 }));
    }
    const parameters = new URLSearchParams({
      q: input.q,
      mode: input.mode,
      limit: String(input.limit ?? 100),
    });
    if (append && input.cursor) parameters.set('cursor', input.cursor);
    if (input.status) parameters.set('status', input.status);
    if (input.annotated !== undefined) parameters.set('annotated', String(input.annotated));
    if (input.has_annotation_file !== undefined) parameters.set('has_annotation_file', String(input.has_annotation_file));
    try {
      let response: AssetCursorPage;
      try {
        response = await localRequest<AssetCursorPage>(base, `/datasets/${encodeURIComponent(datasetId)}/search-cursor?${parameters.toString()}`, { signal: controller.signal }, 30000);
      } catch (error) {
        if (!(error instanceof LocalApiError) || error.code !== 'stale_dataset_cursor' || controller.signal.aborted || requestId !== assetSearchRequestId.current) throw error;
        parameters.delete('cursor');
        setAssetSearch({ phase: 'loading', data: { items: [], total: 0, next_cursor: null, index_revision: typeof error.details?.index_revision === 'number' ? error.details.index_revision : 0 } });
        response = await localRequest<AssetCursorPage>(base, `/datasets/${encodeURIComponent(datasetId)}/search-cursor?${parameters.toString()}`, { signal: controller.signal }, 30000);
        append = false;
      }
      if (requestId !== assetSearchRequestId.current || controller.signal.aborted || queryKey !== assetSearchKey.current) return null;
      setAssetSearch((old) => ({ phase: 'ready', data: mergeAssetCursorPage(old.data, response, append) }));
      return response;
    } catch (error) {
      if (controller.signal.aborted || requestId !== assetSearchRequestId.current) return null;
      setAssetSearch((old) => ({ phase: 'error', data: old.data, stale: old.data.items.length > 0, error: apiError(error) }));
      throw error;
    } finally {
      if (requestId === assetSearchRequestId.current) setAssetSearchLoadingMore(false);
    }
  }, []);

  const revalidateDatasetAsset = useCallback(async (datasetId: string, assetId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Dataset asset revalidation requires the local service' });
    return localRequest<DatasetScanResult['items'][number]>(base, `/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/revalidate`, { method: 'POST' }, 30000);
  }, []);

  const cancelAssetSearch = useCallback(() => {
    assetSearchController.current?.abort();
    assetSearchRequestId.current += 1;
    setAssetSearchLoadingMore(false);
    setAssetSearch((old) => ({ ...old, phase: 'loading', stale: old.data.items.length > 0, error: undefined }));
  }, []);

  const fetchAnnotation = useCallback(async (datasetId: string, assetId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Annotation loading requires the local service' });
    const key = `${datasetId}:${assetId}`;
    const cached = annotationCache.current.get(key);
    if (cached) {
      annotationCache.current.delete(key);
      annotationCache.current.set(key, cached);
      return structuredClone(cached);
    }
    const active = annotationInflight.current.get(key);
    if (active) return structuredClone(await active);
    const request = localRequest<AnnotationEnvelope>(base, `/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/annotation`, {}, 10000)
      .then((response) => {
        annotationCache.current.delete(key);
        annotationCache.current.set(key, structuredClone(response));
        while (annotationCache.current.size > ANNOTATION_CACHE_LIMIT) {
          annotationCache.current.delete(annotationCache.current.keys().next().value!);
        }
        return response;
      })
      .finally(() => annotationInflight.current.delete(key));
    annotationInflight.current.set(key, request);
    return structuredClone(await request);
  }, []);

  const loadAnnotation = useCallback(async (datasetId: string, assetId: string) => {
    setAnnotation((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await fetchAnnotation(datasetId, assetId);
      setAnnotation({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      setAnnotation({ phase: 'error', data: null, error: apiError(error) });
      throw error;
    }
  }, [fetchAnnotation]);

  const prefetchAnnotation = useCallback(async (datasetId: string, assetId: string) => {
    await fetchAnnotation(datasetId, assetId);
  }, [fetchAnnotation]);

  const saveAnnotation = useCallback(async (datasetId: string, assetId: string, revision: string, document: AnnotationEnvelope['document']) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Annotation saving requires the local service' });
    const response = await localRequest<AnnotationSaveResponse>(base, `/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/annotation`, {
      method: 'PUT',
      headers: { 'If-Match': `"${revision}"` },
      body: JSON.stringify({ document }),
    }, 15000);
    setAnnotation((old) => {
      if (!old.data) return old;
      const next = { ...old.data, revision: response.revision, document };
      annotationCache.current.set(`${datasetId}:${assetId}`, structuredClone(next));
      return { phase: 'ready', data: next };
    });
    return response;
  }, []);

  const assetUrl = useCallback((datasetId: string, assetId: string, kind: 'image' | 'thumbnail' | 'region' = 'image', query = '') => {
    const base = baseRef.current;
    if (!base) return null;
    return `${base}/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/${kind}${query}`;
  }, []);

  const loadTileMetadata = useCallback(async (datasetId: string, assetId: string, signal?: AbortSignal) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Image tiles require the local service' });
    return localRequest<TileMetadata>(
      base,
      `/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/tiles/metadata`,
      { signal },
      10000,
    );
  }, []);

  const tileUrl = useCallback((datasetId: string, assetId: string, level: number, x: number, y: number, format = 'webp') => {
    const base = baseRef.current;
    if (!base) return null;
    return `${base}/datasets/${encodeURIComponent(datasetId)}/assets/${encodeURIComponent(assetId)}/tiles/${level}/${x}/${y}?format=${encodeURIComponent(format)}`;
  }, []);

  const artifactContentUrl = useCallback((artifactId: string) => {
    const base = baseRef.current;
    if (!base) return null;
    return `${base}/artifacts/${encodeURIComponent(artifactId)}/content`;
  }, []);

  const artifactPreviewUrl = useCallback((artifactId: string) => {
    const base = baseRef.current;
    if (!base) return null;
    return `${base}/artifacts/${encodeURIComponent(artifactId)}/preview`;
  }, []);

  const invalidatePipelinePreview = useCallback(() => {
    pipelineRequestId.current += 1;
    pipelineController.current?.abort();
    pipelineController.current = null;
    setPipeline((old) => old.phase === 'loading' ? { ...old, phase: 'idle', stale: Boolean(old.data) } : old);
  }, []);

  const previewPipeline = useCallback(async (input: { dataset_id: string; asset_id: string; priority?: 'interactive' | 'background'; nodes: Array<{ id: string; kind: string; enabled: boolean; parameters: Record<string, unknown> }> }) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Pipeline preview requires the local service' });
    pipelineController.current?.abort();
    const controller = new AbortController();
    const requestId = ++pipelineRequestId.current;
    pipelineController.current = controller;
    setPipeline((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<PipelinePreviewResult>(base, '/pipelines/preview', {
        method: 'POST',
        body: JSON.stringify(input),
        signal: controller.signal,
      }, 120000);
      if (requestId !== pipelineRequestId.current) throw new LocalApiError({ code: 'request_superseded', message: 'Pipeline preview was superseded by a newer graph' });
      setPipeline({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      if (requestId === pipelineRequestId.current) setPipeline((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    } finally {
      if (pipelineController.current === controller) pipelineController.current = null;
    }
  }, []);

  const restorePipelinePreview = useCallback((result: PipelinePreviewResult) => {
    pipelineRequestId.current += 1;
    pipelineController.current?.abort();
    pipelineController.current = null;
    setPipeline({ phase: 'ready', data: result });
  }, []);

  const validatePipeline = useCallback(async (input: { nodes: Array<{ id: string; kind: string; enabled: boolean; parameters: Record<string, unknown> }>; width?: number; height?: number }) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Pipeline validation requires the local service' });
    pipelineValidationController.current?.abort();
    const controller = new AbortController();
    const requestId = ++pipelineValidationRequestId.current;
    pipelineValidationController.current = controller;
    setPipelineValidation((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<PipelineValidationResult>(base, '/pipelines/validate', { method: 'POST', body: JSON.stringify(input), signal: controller.signal }, 30000);
      if (requestId !== pipelineValidationRequestId.current) throw new LocalApiError({ code: 'request_superseded', message: 'Pipeline validation was superseded' });
      setPipelineValidation({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      if (requestId === pipelineValidationRequestId.current) setPipelineValidation({ phase: 'error', data: null, error: apiError(error) });
      throw error;
    } finally {
      if (pipelineValidationController.current === controller) pipelineValidationController.current = null;
    }
  }, []);

  const invalidatePipelineValidation = useCallback(() => {
    pipelineValidationRequestId.current += 1;
    pipelineValidationController.current?.abort();
    pipelineValidationController.current = null;
    setPipelineValidation({ phase: 'idle', data: null });
  }, []);

  const prefetchPipelinePreview = useCallback(async (input: { dataset_id: string; asset_id: string; priority?: 'interactive' | 'background'; nodes: Array<{ id: string; kind: string; enabled: boolean; parameters: Record<string, unknown> }> }) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Pipeline prefetch requires the local service' });
    pipelinePrefetchController.current?.abort();
    const controller = new AbortController();
    const requestId = ++pipelinePrefetchRequestId.current;
    pipelinePrefetchController.current = controller;
    try {
      const response = await localRequest<PipelinePreviewResult>(base, '/pipelines/preview', { method: 'POST', body: JSON.stringify({ ...input, priority: 'background' }), signal: controller.signal }, 120000);
      if (requestId !== pipelinePrefetchRequestId.current) throw new LocalApiError({ code: 'request_superseded', message: 'Pipeline prefetch was superseded' });
      return response;
    } finally {
      if (pipelinePrefetchController.current === controller) pipelinePrefetchController.current = null;
    }
  }, []);

  const cancelPipelinePrefetch = useCallback(() => {
    pipelinePrefetchRequestId.current += 1;
    pipelinePrefetchController.current?.abort();
    pipelinePrefetchController.current = null;
  }, []);

  const refreshPipelineRegistry = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return null;
    setPipelineRegistry((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<PipelineRegistryResponse>(base, '/pipelines/operators', {}, 10000);
      setPipelineRegistry({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      setPipelineRegistry((old) => ({ phase: 'error', data: old.data, stale: Boolean(old.data), error: apiError(error) }));
      throw error;
    }
  }, []);

  const importPipelineOperator = useCallback(async (file: File) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Operator import requires the local service' });
    if (!file.name.toLowerCase().endsWith('.zip')) throw new LocalApiError({ code: 'invalid_operator_package', message: 'Operator package must be a .zip file' });
    const response = await localRequest<PipelineOperatorImportResponse>(
      base,
      `/pipelines/operators/import?filename=${encodeURIComponent(file.name)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/zip' },
        body: file,
      },
      120000,
    );
    await refreshPipelineRegistry();
    return response;
  }, [refreshPipelineRegistry]);

  const inspectPipelineOperator = useCallback(async (file: File) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Operator inspection requires the local service' });
    if (!file.name.toLowerCase().endsWith('.zip')) throw new LocalApiError({ code: 'invalid_operator_package', message: 'Operator package must be a .zip file' });
    return localRequest<PipelineOperatorInspection>(
      base,
      `/pipelines/operators/inspect?filename=${encodeURIComponent(file.name)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/zip' },
        body: file,
      },
      30000,
    );
  }, []);

  const pipelineArtifactUrl = useCallback((artifactId: string) => {
    const base = baseRef.current;
    return base ? `${base}/pipeline-artifacts/${encodeURIComponent(artifactId)}` : null;
  }, []);

  const refreshJobs = useCallback(async () => {
    const base = baseRef.current;
    if (!base) return null;
    try {
      const response = await localRequest<JobListResponse>(base, '/jobs?limit=100', {}, 8000);
      setJobs({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      setJobs((old) => ({ phase: 'error', data: old.data, stale: old.data.jobs.length > 0, error: apiError(error) }));
      return null;
    }
  }, []);

  const createJob = useCallback(async (input: BatchJobRequest, idempotencyKey = crypto.randomUUID()) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Batch jobs require the local service' });
    const response = await localRequest<JobRecord>(base, '/jobs', {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify(input),
    }, 15000);
    await refreshJobs();
    return response;
  }, [refreshJobs]);

  const ensurePipelinePrecompute = useCallback(async (input: BatchJobRequest) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Pipeline precompute requires the local service' });
    const response = await localRequest<PipelinePrecomputeEnsureResponse>(base, '/pipelines/precompute/ensure', {
      method: 'POST',
      body: JSON.stringify(input),
    }, 15000);
    await refreshJobs();
    return response;
  }, [refreshJobs]);

  const controlJob = useCallback(async (jobId: string, action: 'pause' | 'resume' | 'cancel') => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Job control requires the local service' });
    const response = await localRequest<JobRecord>(base, `/jobs/${encodeURIComponent(jobId)}/${action}`, { method: 'POST' }, 10000);
    if (action === 'resume') {
      setJobItemProgress((old) => Object.fromEntries(Object.entries(old).filter(([key]) => !key.startsWith(`${jobId}:`))));
      setJobItemSnapshots((old) => Object.fromEntries(Object.entries(old).filter(([key]) => !key.startsWith(`${jobId}:`))));
    }
    await refreshJobs();
    return response;
  }, [refreshJobs]);

  const prioritizeJobItems = useCallback(async (jobId: string, assetIds: string[]) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Job priority requires the local service' });
    const response = await localRequest<JobRecord>(base, `/jobs/${encodeURIComponent(jobId)}/prioritize`, {
      method: 'POST',
      body: JSON.stringify({ asset_ids: assetIds }),
    }, 10000);
    setJobs((old) => ({ phase: 'ready', data: { jobs: old.data.jobs.map((job) => job.job_id === jobId ? response : job) } }));
    return response;
  }, []);

  const loadJobItems = useCallback(async (jobId: string, offset = 0) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Job items require the local service' });
    setJobItems((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<JobItemListResponse>(base, `/jobs/${encodeURIComponent(jobId)}/items?offset=${offset}&limit=200`, {}, 10000);
      setJobItems({ phase: 'ready', data: response });
      setJobItemSnapshots((old) => ({
        ...old,
        ...Object.fromEntries(response.items.map((item) => [`${jobId}:${item.asset_id}`, item])),
      }));
      return response;
    } catch (error) {
      setJobItems({ phase: 'error', data: null, error: apiError(error) });
      throw error;
    }
  }, []);

  const lookupJobItems = useCallback(async (jobId: string, assetIds: string[]) => {
    const base = baseRef.current;
    if (!base || assetIds.length === 0) return null;
    const response = await localRequest<JobItemListResponse>(base, `/jobs/${encodeURIComponent(jobId)}/items/lookup`, {
      method: 'POST',
      body: JSON.stringify({ asset_ids: assetIds.slice(0, 200) }),
    }, 10000);
    setJobItemSnapshots((old) => ({
      ...old,
      ...Object.fromEntries(response.items.map((item) => [`${jobId}:${item.asset_id}`, item])),
    }));
    return response;
  }, []);

  const loadModel = useCallback(async (modelId: string, providers: string[] = ['CPUExecutionProvider']) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Model loading requires the local service' });
    runtimeController.current?.abort();
    const controller = new AbortController();
    const requestId = ++runtimeRequestId.current;
    runtimeController.current = controller;
    setRuntime((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<ModelRuntimeState>(base, `/models/${encodeURIComponent(modelId)}/load`, {
        method: 'POST',
        body: JSON.stringify({ providers }),
        signal: controller.signal,
      }, 120000);
      if (requestId !== runtimeRequestId.current || controller.signal.aborted) return null;
      setRuntime({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      if (requestId !== runtimeRequestId.current || controller.signal.aborted) return null;
      setRuntime({ phase: 'error', data: null, error: apiError(error) });
      throw error;
    } finally {
      if (runtimeController.current === controller) runtimeController.current = null;
    }
  }, []);

  const recordModelUsage = useCallback(async (modelId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Model usage requires the local service' });
    const response = await localRequest<{ count: number; last_used_at: string | null }>(base, `/models/${encodeURIComponent(modelId)}/usage`, {
      method: 'POST',
    }, 10000);
    setApplicationSettings((old) => old.data ? {
      ...old,
      data: { ...old.data, model_usage: { ...old.data.model_usage, [modelId]: response } },
    } : old);
    await refreshModels();
    return response;
  }, [refreshModels]);

  const loadModelLayers = useCallback(async (modelId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Model layers require the local service' });
    runtimeController.current?.abort();
    const controller = new AbortController();
    const requestId = ++runtimeRequestId.current;
    runtimeController.current = controller;
    try {
      const response = await localRequest<ModelRuntimeState>(base, `/models/${encodeURIComponent(modelId)}/layers`, { signal: controller.signal }, 30000);
      if (requestId !== runtimeRequestId.current || controller.signal.aborted) return null;
      setRuntime({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      if (requestId !== runtimeRequestId.current || controller.signal.aborted) return null;
      throw error;
    } finally {
      if (runtimeController.current === controller) runtimeController.current = null;
    }
  }, []);

  const runInference = useCallback(async (input: { model_id: string; image_path: string; capture_layers: string[]; parameters: Record<string, unknown> }) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Inference requires the local service' });
    inferenceController.current?.abort();
    const controller = new AbortController();
    const requestId = ++inferenceRequestId.current;
    inferenceController.current = controller;
    setInference((old) => ({ ...old, phase: 'loading', error: undefined }));
    try {
      const response = await localRequest<InferenceResult>(base, '/inference-runs', {
        method: 'POST',
        body: JSON.stringify(input),
        signal: controller.signal,
      }, 120000);
      if (requestId !== inferenceRequestId.current || controller.signal.aborted) return null;
      setInference({ phase: 'ready', data: response });
      return response;
    } catch (error) {
      if (requestId !== inferenceRequestId.current || controller.signal.aborted) return null;
      setInference({ phase: 'error', data: null, error: apiError(error) });
      throw error;
    } finally {
      if (inferenceController.current === controller) inferenceController.current = null;
    }
  }, []);

  const runAgent = useCallback(async (input: AgentRunInput) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Agent runs require the local service' });
    return localRequest<AgentRunResult>(base, '/agent/runs', {
      method: 'POST',
      body: JSON.stringify(input),
    }, 120000);
  }, []);

  const executeAgentProposal = useCallback(async (runId: string, proposalId: string) => {
    const base = baseRef.current;
    if (!base) throw new LocalApiError({ code: 'local_service_unavailable', message: 'Agent proposal execution requires the local service' });
    return localRequest<AgentRunResult>(base, `/agent/runs/${encodeURIComponent(runId)}/proposals/${encodeURIComponent(proposalId)}/execute`, {
      method: 'POST',
    }, 120000);
  }, []);

  const watchJobEvents = useCallback((jobId: string | null) => setPreferredJobEventId(jobId), []);
  const preferredActiveJob = preferredJobEventId
    ? jobs.data.jobs.find((job) => job.job_id === preferredJobEventId && activeJobStates.has(job.state))
    : undefined;
  const eventJobId = preferredActiveJob?.job_id ?? jobs.data.jobs.find((job) => activeJobStates.has(job.state))?.job_id ?? null;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const base = localApiBase();
      baseRef.current = base;
      if (!base) {
        setMode('offline');
        setHealth({ phase: 'error', data: null, error: { code: 'local_service_unavailable', message: 'Local service is unavailable' } });
        return;
      }
      void refreshHealth();
    }, 0);
    return () => {
      window.clearTimeout(timer);
      healthController.current?.abort();
      scanController.current?.abort();
      assetSearchController.current?.abort();
      pipelineController.current?.abort();
      runtimeController.current?.abort();
      inferenceController.current?.abort();
      pipelineValidationController.current?.abort();
      pipelinePrefetchController.current?.abort();
    };
  }, [refreshHealth]);

  useEffect(() => {
    const onVisibilityChange = () => setPageVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  useEffect(() => {
    let disposed = false;
    let terminalEvent = false;
    let reconnectAttempt = 0;
    let source: EventSource | null = null;
    let reconnectTimer: number | undefined;
    let connectionTimer: number | undefined;
    let refreshing = false;
    let refreshQueued = false;
    const base = baseRef.current;

    const publish = (nextMode: 'realtime' | 'connecting' | 'polling' | 'hidden' | 'offline', jobId: string | null) => {
      if (disposed) return;
      setJobEvents({ mode: nextMode, job_id: jobId, last_event_id: jobId ? jobLastEventIds.current.get(jobId) ?? null : null });
    };

    const refreshFromEvent = async () => {
      if (refreshing) {
        refreshQueued = true;
        return;
      }
      refreshing = true;
      do {
        refreshQueued = false;
        await refreshJobs();
        if (!disposed && preferredJobEventId === eventJobId && eventJobId) {
          await loadJobItems(eventJobId).catch(() => undefined);
        }
      } while (!disposed && refreshQueued);
      refreshing = false;
    };

    const closeSource = () => {
      if (connectionTimer) window.clearTimeout(connectionTimer);
      connectionTimer = undefined;
      source?.close();
      source = null;
    };

    const connect = () => {
      if (disposed || terminalEvent || mode !== 'online' || !pageVisible || !eventJobId || !base || typeof EventSource === 'undefined') return;
      closeSource();
      publish('connecting', eventJobId);
      const after = jobLastEventIds.current.get(eventJobId) ?? '0';
      source = new EventSource(`${base}/jobs/${encodeURIComponent(eventJobId)}/events?after=${encodeURIComponent(after)}`);
      connectionTimer = window.setTimeout(() => {
        if (disposed || !source) return;
        closeSource();
        publish('polling', eventJobId);
        scheduleReconnect();
      }, 8000);
      source.onopen = () => {
        if (disposed) return;
        if (connectionTimer) window.clearTimeout(connectionTimer);
        connectionTimer = undefined;
        reconnectAttempt = 0;
        publish('realtime', eventJobId);
      };
      const handleEvent = (event: MessageEvent) => {
        if (disposed) return;
        let identifier = event.lastEventId;
        let payload: unknown = null;
        if (event.data) {
          try {
            payload = JSON.parse(event.data);
            if (!identifier && payload && typeof payload === 'object') {
              const eventPayload = payload as Record<string, unknown>;
              identifier = String(eventPayload.event_id ?? eventPayload.id ?? '');
            }
          } catch { /* Event payload remains opaque to the frontend. */ }
        }
        if (event.type === 'item.progress' && payload && typeof payload === 'object') {
          const progress = payload as Record<string, unknown>;
          if (typeof progress.asset_id === 'string' && (typeof progress.progress === 'number' || progress.progress === null)) {
            const progressSnapshot = { ...progress };
            delete progressSnapshot.asset_id;
            const key = `${eventJobId}:${progress.asset_id}`;
            setJobItemSnapshots((old) => ({
              ...old,
              [key]: {
                ...(old[key] ?? { asset_id: progress.asset_id as string, position: -1, state: 'running' as const, attempts: 1 }),
                progress: progressSnapshot,
              },
            }));
          }
          if (typeof progress.asset_id === 'string' && typeof progress.received_bytes === 'number' && (typeof progress.total_bytes === 'number' || progress.total_bytes === null) && (typeof progress.progress === 'number' || progress.progress === null)) {
            const value: JobItemProgress = { job_id: eventJobId, asset_id: progress.asset_id, received_bytes: progress.received_bytes, total_bytes: progress.total_bytes, progress: progress.progress };
            setJobItemProgress((old) => ({ ...old, [`${eventJobId}:${progress.asset_id}`]: value }));
          }
        }
        if (event.type === 'item.state' && payload && typeof payload === 'object') {
          const item = payload as Record<string, unknown>;
          if (typeof item.asset_id === 'string' && typeof item.state === 'string' && jobItemStateValues.has(item.state)) {
            const key = `${eventJobId}:${item.asset_id}`;
            setJobItemSnapshots((old) => ({
              ...old,
              [key]: {
                ...(old[key] ?? { asset_id: item.asset_id as string, position: -1, state: 'queued' as const, attempts: 0 }),
                state: item.state as JobItem['state'],
                attempts: typeof item.attempts === 'number' ? item.attempts : old[key]?.attempts ?? 0,
                error: typeof item.error === 'string' ? item.error : undefined,
              },
            }));
          }
        }
        if (identifier) jobLastEventIds.current.set(eventJobId, identifier);
        publish('realtime', eventJobId);
        void refreshFromEvent();
        if (event.type === 'job.terminal') {
          terminalEvent = true;
          closeSource();
          publish('polling', eventJobId);
        }
      };
      source.onmessage = handleEvent;
      for (const eventType of jobEventTypes) source.addEventListener(eventType, handleEvent as EventListener);
      source.onerror = () => {
        if (disposed || terminalEvent) return;
        closeSource();
        publish('polling', eventJobId);
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (disposed || terminalEvent) return;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      const delay = [1000, 2000, 5000][Math.min(reconnectAttempt, 2)];
      reconnectAttempt += 1;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = undefined;
        connect();
      }, delay);
    };

    const statusTimer = window.setTimeout(() => {
      if (mode !== 'online') publish('offline', null);
      else if (!pageVisible) publish('hidden', eventJobId);
      else if (!eventJobId || !base || typeof EventSource === 'undefined') publish('polling', eventJobId);
      else connect();
    }, 0);

    return () => {
      disposed = true;
      closeSource();
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (statusTimer) window.clearTimeout(statusTimer);
    };
  }, [eventJobId, loadJobItems, mode, pageVisible, preferredJobEventId, refreshJobs]);

  useEffect(() => {
    if (mode !== 'online') return;
    if (pageVisible && eventJobId && jobEvents.mode === 'realtime' && jobEvents.job_id === eventJobId) return;
    let stopped = false;
    let timer: number | undefined;
    const poll = async () => {
      const response = await refreshJobs();
      if (stopped) return;
      const active = response?.jobs.some((job) => activeJobStates.has(job.state));
      const delay = !pageVisible ? 10000 : active ? 1000 : 5000;
      timer = window.setTimeout(() => void poll(), delay);
    };
    timer = window.setTimeout(() => void poll(), 0);
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [eventJobId, jobEvents.job_id, jobEvents.mode, mode, pageVisible, refreshJobs]);

  return {
    mode,
    health,
    applicationSettings,
    agentStatus,
    models,
    runtime,
    inference,
    scan,
    datasets,
    annotation,
    pipeline,
    pipelineValidation,
    pipelineRegistry,
    jobs,
    jobItems,
    jobEvents,
    jobItemProgress,
    jobItemSnapshots,
    assetSearch,
    assetSearchLoadingMore,
    assetSearchDatasetId,
    refreshHealth,
    refreshApplicationSettings,
    refreshAgentStatus,
    updateApplicationSettings,
    refreshModels,
    listModelWeights,
    downloadModelWeight,
    refreshDatasets,
    startScan,
    pickDirectory,
    interruptScan,
    resumeScan,
    registerScan,
    openRegisteredDataset,
    getDatasetSettings,
    saveDatasetSettings,
    getDatasetAsset,
    searchDatasetAssets,
    revalidateDatasetAsset,
    cancelAssetSearch,
    loadAnnotation,
    prefetchAnnotation,
    saveAnnotation,
    assetUrl,
    loadTileMetadata,
    tileUrl,
    artifactContentUrl,
    artifactPreviewUrl,
    previewPipeline,
    restorePipelinePreview,
    invalidatePipelinePreview,
    validatePipeline,
    invalidatePipelineValidation,
    prefetchPipelinePreview,
    cancelPipelinePrefetch,
    refreshPipelineRegistry,
    importPipelineOperator,
    inspectPipelineOperator,
    pipelineArtifactUrl,
    refreshJobs,
    createJob,
    ensurePipelinePrecompute,
    controlJob,
    prioritizeJobItems,
    loadJobItems,
    lookupJobItems,
    watchJobEvents,
    loadModel,
    recordModelUsage,
    loadModelLayers,
    runInference,
    runAgent,
    executeAgentProposal,
  };
}
