export type BackendMode = 'probing' | 'online' | 'offline';

export type ApiError = {
  code: string;
  message: string;
  details?: Record<string, unknown>;
};

export type HealthResponse = {
  status: 'ok' | 'degraded';
  service: string;
  version: string;
  api_version: 'v1';
  model_registry: { configs: number; adapters: number; errors: number };
  runtimes: Record<string, 'available' | 'unavailable'>;
};

export type ApplicationSettings = {
  data_dir: string;
  model_source_dir: string | null;
  model_weights_dir: string;
  effective_model_weights_dir: string;
  model_weights_managed_by: 'default' | 'persisted' | 'environment';
  restart_required: boolean;
  model_download_concurrency: number;
  checksum_verification: boolean;
  model_download_source: string;
  model_download_sources: Array<{ id: string; label: string }>;
  network_proxy: NetworkProxySettings;
  network_proxy_restart_required: boolean;
  cloud_ai: CloudAiSettings;
  workspace: GlobalWorkspaceSettings;
  model_usage: Record<string, ModelUsageRecord>;
};

export type ModelUsageRecord = {
  count: number;
  last_used_at: string | null;
};

export type WorkspacePipelineNode = {
  id: string;
  kind: string;
  name: string;
  enabled: boolean;
  parameters: Record<string, unknown>;
  operator_version?: string | null;
};

export type WorkspaceVisualizationNode = WorkspacePipelineNode & {
  kind: 'visualize';
  tap_after_node_id: string;
};

export type WorkspacePipelineSettings = {
  enabled: boolean;
  scope: 'current' | 'all';
  nodes: WorkspacePipelineNode[];
  visualizations: WorkspaceVisualizationNode[];
  display_mode: 'source' | 'split' | 'overlay';
  single_source: string;
  layer_state: Record<string, { visible: boolean; opacity: number }>;
};

export type GlobalWorkspaceSettings = {
  schema_version: 1;
  pipeline: WorkspacePipelineSettings | null;
  inference: {
    model_id: string | null;
    provider: string;
    parameters: Record<string, unknown>;
  };
};

export type DatasetWorkspaceSettings = {
  schema_version: 1;
  last_asset_id: string | null;
  pipeline: WorkspacePipelineSettings | null;
};

export type DatasetWorkspaceSettingsResponse = DatasetWorkspaceSettings & {
  revision: number;
  updated_at: string | null;
};

export type DatasetWorkspaceSettingsUpdate = DatasetWorkspaceSettings & {
  expected_revision: number;
};

export type NetworkProxySettings = {
  mode: 'system' | 'direct' | 'manual';
  url: string;
  bypass: string;
};

export type CloudAiSettings = {
  enabled: boolean;
  provider: 'openai_compatible';
  endpoint: string;
  model: string;
  api_key_env: string;
  timeout_seconds: number;
  max_output_tokens: number;
  credential_configured: boolean;
  credential_source: 'environment' | 'missing';
};

export type ApplicationSettingsUpdate = {
  model_weights_dir?: string;
  model_download_source?: string;
  network_proxy?: NetworkProxySettings;
  cloud_ai?: Omit<CloudAiSettings, 'credential_configured' | 'credential_source'>;
  workspace?: GlobalWorkspaceSettings;
};

export type FeatureLayer = {
  id: string;
  group: string;
  name: string;
  shape: Array<number | string | null>;
  axes: string[];
  dtype?: string;
  spatial: boolean;
  captureable: boolean;
  reason?: string;
};

export type ModelCatalogItem = {
  id: string;
  name: string;
  display_name: string;
  model_type: string;
  provider: string;
  task: string;
  family: string;
  adapter: string;
  runtime: string[];
  config_path: string;
  weight_locations: string[];
  availability: { state: 'available' | 'missing_weights' | 'unsupported' | 'invalid'; reason?: string };
  capabilities: {
    predict: boolean;
    unload: boolean;
    result_kinds: Array<'annotations' | 'classifications' | 'tensors' | 'rasters'>;
    feature_capture: {
      mode: 'none' | 'exported_outputs' | 'eager_hooks' | 'graph_rewrite' | 'remote';
      enumerable: boolean;
      layers: FeatureLayer[];
    };
    parameters_schema: {
      type?: 'object';
      additionalProperties?: boolean;
      properties?: Record<string, PipelineParameterSchema>;
    };
  };
};

export type ModelCatalogResponse = {
  models: ModelCatalogItem[];
  warnings: Array<{ path: string; code: string; message: string }>;
  status_by_model: Record<string, {
    runtime_state: 'unloaded' | 'loading' | 'loaded' | 'failed';
    usage_count: number;
    last_used_at: string | null;
  }>;
};

export type ModelWeightItem = {
  url_index: number;
  url: string;
  filename: string;
  downloaded: boolean;
  local_path: string | null;
  size_bytes: number | null;
  sha256: string | null;
  source_id: string;
  preferred: boolean;
};

export type ModelWeightDownloadResult = {
  model_id: string;
  url_index: number;
  source_url: string;
  final_url: string;
  local_path: string;
  size_bytes: number;
  sha256: string;
  cache_hit: boolean;
};

export type ModelRuntimeState = {
  model_id: string;
  state: 'unloaded' | 'loading' | 'loaded' | 'failed';
  layers: FeatureLayer[];
  capture_mode: 'none' | 'exported_outputs' | 'eager_hooks' | 'graph_rewrite' | 'remote';
  capture_warning?: string;
  error?: string;
};

export type InferenceResult = {
  model_id: string;
  image_path: string;
  annotations: Array<{ label: string; score: number; shape_type: string; points: number[][] }>;
  classifications: Array<{ label: string; score: number; rank: number }>;
  artifacts: Array<{ id: string; layer_id: string; path: string; shape: number[]; source_shape: number[]; transform: Record<string, unknown>; dtype: string; size_bytes: number; statistics: Record<string, number>; preview_available: boolean; preview_width?: number; preview_height?: number }>;
  rasters: Array<{ id: string; role: string; path: string; media_type: string; width: number; height: number; size_bytes: number; metadata: Record<string, unknown> }>;
  timings_ms: Record<string, number>;
};

export type AgentToolName =
  | 'dataset.stats'
  | 'dataset.search'
  | 'annotation.qa'
  | 'dataset.distribution'
  | 'ui.open_dataset'
  | 'ui.import_operator'
  | 'ui.open_models'
  | 'pipeline.draft'
  | 'pipeline.create_job'
  | 'inference.create_job';

export type AgentCapability = {
  tool: AgentToolName;
  group: 'inspect' | 'prepare' | 'run';
  title: string;
  description: string;
  risk: 'read' | 'write';
  requires_confirmation: boolean;
  requires_dataset: boolean;
  requires_asset: boolean;
};

export type AgentStatus = {
  state: 'ready' | 'unconfigured';
  reason_code: 'ready' | 'disabled' | 'missing_credential' | 'invalid_configuration';
  message: string;
  provider: 'openai_compatible';
  model?: string | null;
  credential_env?: string | null;
  capabilities: AgentCapability[];
};

export type AgentRunInput = {
  dataset_id: string;
  asset_id?: string | null;
  message: string;
  tool_call?: {
    tool: AgentToolName;
    arguments: Record<string, unknown>;
  };
};

export type AgentToolResult = {
  tool: AgentToolName;
  data: Record<string, unknown>;
};

export type AgentProposal = {
  id: string;
  tool: string;
  title: string;
  description: string;
  risk: 'read' | 'write';
  requires_confirmation: boolean;
  executed?: boolean;
  result?: Record<string, unknown> | null;
};

export type AgentRunResult = {
  run_id: string;
  dataset_id: string;
  asset_id?: string | null;
  reply: string;
  state: 'proposed' | 'completed' | 'failed';
  proposals: AgentProposal[];
  tool_results: AgentToolResult[];
};

export type DatasetScanInput = {
  dataset_id?: string;
  root_dir: string;
  image_dir?: string;
  annotation_dir?: string;
  annotation_storage_root?: string;
  layout: 'auto' | 'same_directory' | 'parallel' | 'custom';
  match_strategy: 'relative_stem' | 'same_directory' | 'image_path' | 'basename';
  recursive: boolean;
  validate_images: boolean;
  validate_annotations: boolean;
  image_extensions?: string[];
  annotation_extension?: string;
};

export type DirectoryPickerInput = { title: string; initial_dir?: string };
export type DirectoryPickerResult = { path: string | null; canceled: boolean };

export type DatasetScanItem = {
  asset_id: string;
  match_key: string;
  display_path: string;
  image_path?: string;
  annotation_paths: string[];
  status: 'valid' | 'duplicate_match' | 'orphan_annotation' | 'corrupt_image' | 'corrupt_annotation';
  selectable: boolean;
  reason?: string;
  issues: string[];
  width?: number;
  height?: number;
  annotation_count?: number;
  annotation_file_exists: boolean;
  labels?: string[];
  shape_types?: string[];
};

export type DatasetAssetSearchInput = {
  q: string;
  mode: 'smart' | 'text' | 'regex' | 'condition';
  cursor?: string | null;
  limit?: number;
  status?: string;
  annotated?: boolean;
  has_annotation_file?: boolean;
};

export type DatasetScanResult = {
  dataset_id: string;
  root_dir: string;
  image_root: string;
  annotation_roots: string[];
  items: DatasetScanItem[];
  summary: {
    valid: number;
    duplicate_match: number;
    orphan_annotation: number;
    corrupt_image: number;
    corrupt_annotation: number;
    hidden_image_only: number;
  };
};

export type RegisteredDataset = {
  dataset_id: string;
  name: string;
  root_dir: string;
  image_root: string;
  summary: DatasetScanResult['summary'];
  created_at: string;
  updated_at: string;
  index_revision: number;
};

export type DatasetListResponse = { datasets: RegisteredDataset[] };
export type AssetListResponse = { items: DatasetScanItem[]; total: number; next_offset?: number | null };
export type AssetCursorPage = { items: DatasetScanItem[]; total: number; next_cursor: string | null; index_revision: number };

export type DatasetScanSessionState = 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted';
export type DatasetScanSession = {
  session_id: string;
  state: DatasetScanSessionState;
  request: DatasetScanInput;
  dataset_id: string | null;
  root_dir: string | null;
  image_root: string | null;
  annotation_roots: string[];
  summary: DatasetScanResult['summary'] | null;
  persisted_items: number;
  run_generation: number;
  error: string | null;
  registration_name: string | null;
  registered_dataset_id: string | null;
  registered_index_revision: number | null;
  registered_items: number;
  registered_at: string | null;
  interrupted_at: string | null;
  interruption_reason: string | null;
  created_at: string;
  updated_at: string;
};
export type DatasetScanItemPage = { items: DatasetScanItem[]; total: number; next_after: number | null; state: DatasetScanSessionState };
export type DatasetScanSessionView = DatasetScanSession & {
  items: DatasetScanItem[];
  next_after: number | null;
  streamed_summary: DatasetScanResult['summary'];
};

export type TileMetadata = {
  width: number;
  height: number;
  tile_size: number;
  max_level: number;
  format: string;
  source_etag: string;
  backend: 'pillow' | 'pyvips';
  source_format?: string;
};

export type AnnotationEnvelope = {
  dataset_id: string;
  asset_id: string;
  path: string;
  revision: string;
  document: Record<string, unknown> & { shapes?: AnnotationShape[] };
};

export type AnnotationShape = {
  label: string;
  shape_type: string;
  points: number[][];
  direction?: number;
  score?: number | null;
  [key: string]: unknown;
};

export type AnnotationSaveResponse = {
  dataset_id: string;
  asset_id: string;
  path: string;
  previous_revision: string;
  revision: string;
  backup_path: string;
};

export type PipelineVisualizationResult = {
  visualization_id: string;
  label: string;
  artifact_id: string;
  width: number;
  height: number;
  media_type: string;
  annotation_document: AnnotationEnvelope['document'];
  operator_timings_ms: Record<string, number>;
  content_kind?: 'image' | 'model_feature' | 'frequency_spectrum' | 'wavelet_coefficients';
  overlay_compatible?: boolean;
  coordinate_mapping?: {
    kind: 'identity' | 'affine' | 'unavailable';
    source_width: number;
    source_height: number;
    output_width: number;
    output_height: number;
    source_to_output?: [number, number, number, number, number, number] | null;
    output_to_source?: [number, number, number, number, number, number] | null;
    coordinate_space_id: string;
    topology_safe: boolean;
    reason?: string | null;
  };
};

export type PipelinePreviewResult = {
  dataset_id: string;
  asset_id: string;
  artifact_id: string;
  width: number;
  height: number;
  media_type: string;
  annotation_document: AnnotationEnvelope['document'];
  operator_timings_ms: Record<string, number>;
  operator_average_timings_ms?: Record<string, number>;
  timing_sample_count?: Record<string, number> | number;
  /** New services return one entry per visualize tap; absent on legacy services. */
  visualizations?: PipelineVisualizationResult[];
  cache_hit?: boolean;
};

export type PipelineValidationResult = {
  valid: boolean;
  registry_hash: string;
  nodes: PipelineFlowNode[];
  message?: string;
  errors?: Array<{ node_id?: string; message: string }>;
};

export type PipelineOperatorSizeBehavior = 'preserve' | 'deterministic' | 'dynamic';
export type PipelineOperatorNodeRole = 'source' | 'transform' | 'visualization' | 'batch_export';

export type PipelineParameterSchema = {
  type?: 'integer' | 'number' | 'string' | 'boolean';
  default?: unknown;
  enum?: Array<string | number | boolean>;
  minimum?: number;
  maximum?: number;
  multipleOf?: number;
  title?: string;
  description?: string;
  'x-ui'?: {
    control?: 'auto' | 'slider' | 'number';
    role?: 'region-x' | 'region-y' | 'region-width' | 'region-height' | 'target-width' | 'target-height' | 'scale-factor' | 'ratio';
    unit?: string;
  };
};

export type PipelineOperatorContract = {
  kind: string;
  title: string;
  description: string;
  version: string;
  input_type: string;
  output_type: string;
  annotation_policy: Record<string, unknown>;
  /** Optional while older local services are still supported. */
  size_behavior?: PipelineOperatorSizeBehavior;
  /** Optional while older local services are still supported. */
  node_role?: PipelineOperatorNodeRole;
  source?: 'builtin' | 'opencv' | 'custom';
  parameters_schema: {
    type: 'object';
    default?: Record<string, unknown>;
    properties: Record<string, PipelineParameterSchema>;
    additionalProperties?: boolean;
    [key: string]: unknown;
  };
};

export type InstalledOperatorPackage = {
  kind: string;
  title: string;
  version: string;
  digest: string;
  package_dir: string;
  entrypoint: string;
  annotation_entrypoint?: string | null;
  annotation_policy: 'preserve' | 'scale' | 'transform';
  size_behavior: PipelineOperatorSizeBehavior;
  trusted_local_code: true;
  is_os_sandboxed: false;
};

export type PipelineOperatorInspection = {
  operator: PipelineOperatorContract;
  digest: string;
  entrypoint: string;
  annotation_entrypoint?: string | null;
  filename: string;
  annotation_policy: 'preserve' | 'scale' | 'transform';
  will_execute_local_code: true;
  is_os_sandboxed: false;
};

export type PipelineCompositeSummary = {
  id: string;
  name: string;
  description: string;
  version_hash: string;
};

export type PipelineRegistryResponse = {
  registry_hash: string;
  operators: PipelineOperatorContract[];
  installed_packages?: InstalledOperatorPackage[];
  composites: PipelineCompositeSummary[];
  warnings?: string[];
};

export type PipelineOperatorImportResponse = {
  operator: PipelineOperatorContract;
  digest: string;
  package_dir: string;
  trusted_local_code: boolean;
  is_os_sandboxed: boolean;
};

export type PipelineFlowNode = {
  id: string;
  kind: string;
  enabled: boolean;
  parameters: Record<string, unknown>;
  operator_version: string;
};

export type PipelineCompositeDefinition = {
  id: string;
  name: string;
  description?: string;
  steps: Array<Record<string, unknown>>;
};

export type PipelineExpandedComposite = {
  id: string;
  version_hash: string;
  output_width: number;
  output_height: number;
  nodes: PipelineFlowNode[];
};

export type JobState = 'queued' | 'running' | 'pausing' | 'paused' | 'canceling' | 'canceled' | 'succeeded' | 'succeeded_with_errors' | 'failed' | 'interrupted';
export type JobItemState = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';

export type JobItem = {
  asset_id: string;
  position: number;
  state: JobItemState;
  attempts: number;
  result?: Record<string, unknown>;
  error?: string;
  started_at?: string;
  finished_at?: string;
  progress?: JobItemProgressSnapshot;
};

export type JobItemProgressSnapshot = {
  kind?: 'pipeline' | 'model_download' | string;
  progress?: number | null;
  phase?: string;
  completed_steps?: number;
  total_steps?: number;
  node_id?: string | null;
  node_kind?: string | null;
  received_bytes?: number;
  total_bytes?: number | null;
  [key: string]: unknown;
};

export type BatchJobRequest = {
  kind: 'pipeline' | 'inference' | 'model_download' | 'category_rename';
  dataset_id: string;
  asset_ids?: string[];
  preferred_asset_ids?: string[];
  concurrency?: number;
  priority?: 'user_batch' | 'background';
  pipeline_nodes?: Array<{ id: string; kind: string; enabled: boolean; parameters: Record<string, unknown> }>;
  output_policy?: {
    mode: 'preview' | 'derived_dataset';
    output_root?: string;
    image_format: 'png' | 'webp' | 'jpeg';
    conflict: 'reuse' | 'error';
  };
  model_id?: string;
  capture_layers?: string[];
  parameters?: Record<string, unknown>;
  weight_url_indices?: number[];
  expected_sha256?: Record<number, string>;
  source_category?: string;
  target_category?: string;
  pipeline_context?: {
    signature: string;
    dataset_index_revision: number;
    registry_hash: string;
    output_format: 'png' | 'webp' | 'jpeg';
  } | null;
};

export type JobRecord = {
  job_id: string;
  kind: 'pipeline' | 'inference' | 'model_download' | 'category_rename';
  dataset_id: string;
  state: JobState;
  desired_state: 'run' | 'pause' | 'cancel';
  generation: number;
  request: BatchJobRequest;
  total: number;
  completed: number;
  failed: number;
  canceled: number;
  created_at: string;
  updated_at: string;
  error?: string;
  items: JobItem[];
};

export type JobListResponse = { jobs: JobRecord[] };
export type PipelinePrecomputeEnsureResponse = { job: JobRecord; reused: boolean; resumed: boolean; canceled_job_ids: string[] };
export type JobItemListResponse = { items: JobItem[]; total: number; next_offset?: number };
export type JobItemProgress = { job_id: string; asset_id: string; received_bytes: number; total_bytes: number | null; progress: number | null };

export type RemoteState<T> = {
  phase: 'idle' | 'loading' | 'ready' | 'error';
  data: T;
  error?: ApiError;
  stale?: boolean;
};
