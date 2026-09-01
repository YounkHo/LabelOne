'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { TiledImage } from './components/tiled-image';
import { VirtualFileList } from './components/virtual-file-list';
import { CustomSelect } from './components/custom-select';
import { GlobalTooltip } from './components/global-tooltip';
import { GlobalSettingsPage, type CloudAiDraft, type GlobalSettingsSection } from './components/global-settings-page';
import { PipelineInsertPopover } from './components/pipeline-insert-popover';
import { PipelineParameterControl } from './components/pipeline-parameter-control';
import { ModelPickerDialog } from './components/model-picker-dialog';
import { UiLanguageBridge } from './components/ui-language-bridge';
import { WelcomeScreen } from './components/welcome-screen';
import { useLocalBackend } from './hooks/use-local-backend';
import { deleteAnnotationDraft, getAnnotationDraft, putAnnotationDraft, type PersistedAnnotationDraft } from './lib/annotation-drafts';
import { annotationFingerprint, commitAnnotationHistory, createAnnotationHistory, redoAnnotationHistory, undoAnnotationHistory, type AnnotationHistory } from './lib/annotation-history';
import { annotationCategoryColors, normalizeAnnotationCategoryColor } from './lib/annotation-category-colors';
import { annotationIndexesForCategory, normalizeAnnotationCategory, setAnnotationIndexesVisible } from './lib/annotation-category-filter';
import { DEFAULT_ANNOTATION_AUTO_SAVE, shouldWriteAnnotationFile } from './lib/annotation-save-policy';
import { buildAnnotationLabelChoices, normalizeAnnotationLabel, positionFloatingLabelMenu, type ScreenPoint } from './lib/annotation-labels';
import { annotationHitCandidates, annotationShapeClass, canClosePolygonAtPoint, compactFreehandPoints, createDragShape, createFreehandLine, editableControlPointIndexes, moveShapeControlPoint, polygonVertexControlPath, rotateRotationShape, rotationCenter, rotationCornerHandle, rotationDirection, selectAnnotationHitIndex, translateShapeWithinImage, type AnnotationPoint } from './lib/annotation-tools';
import { remapHiddenShapesAfterDeletion, remapHiddenShapesAfterDeletions, remapSelectedShapeAfterDeletion, remapSelectedShapeAfterDeletions } from './lib/annotation-visibility';
import { BACKGROUND_TASK_ACTIVE_STATES, BACKGROUND_TASK_ATTENTION_STATES, BACKGROUND_TASK_HISTORY_HOURS, clearableCompletedTaskIds, filterBackgroundTaskHistory, type BackgroundTaskHistoryHours } from './lib/background-task-history';
import { applyCanvasWheel, CANVAS_ZOOM_STEP, canvasScaleForSourcePixelSize, canvasScaleFromPercent, isBrowserZoomKeyboardShortcut, isSourcePixelGridVisible, maximumCanvasScaleForPixelInspection, MAX_SOURCE_PIXEL_INSPECTION_SIZE, MIN_SOURCE_PIXEL_GRID_SIZE, sourcePixelScreenSize, zoomCanvasView } from './lib/canvas-viewport';
import { resolveCanvasCursorMode, resolveResizeCursor } from './lib/canvas-cursor';
import { inverseTransformCanvasDelta, inverseTransformCanvasPoint, inverseTransformCanvasShape, resolveCanvasPresentation, resolvePipelineCoordinateTransform, shouldSwitchToSelectAfterBlankClick, transformCanvasShape, type CanvasCoordinateTransform } from './lib/canvas-interaction';
import { CANVAS_CONTROL_POINT_RADIUS_PX, CANVAS_VERTEX_CONTROL_DIAMETER_PX, canvasAnnotationOpticalScale, canvasLabelLayouts, canvasLabelOpacity, canvasLabelTopVertexAnchor, canvasVisibleImageBounds, fitCanvasLabelText, type CanvasLabelLayout } from './lib/canvas-label';
import { navigatorPointToView, navigatorViewport, type CanvasNavigatorMetrics } from './lib/canvas-navigator';
import { pixelSampleFromRgba, sourcePixelAtDisplayPoint, type PixelSample } from './lib/pixel-sampling';
import { insertPipelineNode, insertPipelineNodeAtGap, MAX_PIPELINE_VISUALIZATIONS, normalizeVisualizationTaps, pipelineInsertionGaps, pipelineLinearItems, pipelineSignature, removePipelineNode, serializePipelineNodes, visualizationOverlayCompatibility, type PipelineInsertionGap, type PipelineVisualizationNode } from './lib/pipeline-flow';
import { formatPipelineTiming, neighboringAssetIds, pipelineArtifactDisplayUrl, pipelineImageRetryExhausted, pipelinePrecomputeKey, pipelinePreviewCacheKey, pipelinePreviewResultFromJobItem, pipelineRequestNodesEqual, pipelineValidationKey, storeCachedPipelinePreview, takeCachedPipelinePreview } from './lib/pipeline-preview';
import { canHidePipelineLayer, containedPipelineImageRect, createPipelineSharedCursor, pipelineCoordinateMappingFromTransform, pipelinePaneMetrics, pipelinePaneTransform, pipelinePaneVectorToReference, pipelineWheelInputToReference, pipelineSharedCursorPointForPane, resolvePipelineDisplayMode, snapPipelineGridCoordinate, stablePipelineDisplaySlots, updatePipelineLayerOpacity, updatePipelineLayerVisibility, type PipelineCoordinateMappingLike, type PipelineDisplayMode, type PipelineSharedCursor } from './lib/pipeline-multiview';
import type { AgentCapability, AgentRunResult, AgentToolName, AgentToolResult, AnnotationEnvelope, AnnotationShape, AssetCursorPage, DatasetAssetSearchInput, DatasetScanItem, DatasetScanResult, DatasetWorkspaceSettingsResponse, FeatureLayer, JobRecord, NetworkProxySettings, PipelineFlowNode, PipelineOperatorContract, PipelineOperatorInspection, PipelineVisualizationResult, RegisteredDataset, TileMetadata, WorkspacePipelineSettings } from './lib/contracts';
import { directoryBasename, formatDownloadProgress } from './lib/dataset-stream';
import { fileAnnotationFilterLabels, fileAnnotationFilters, type FileAnnotationFilter } from './lib/file-annotation-filter';
import { defaultFeatureProjection, featurePreviewDescription, featureProjectionOptions, featureTensorKind, featureTensorKindLabel } from './lib/feature-presentation';
import { inferenceParameterDefaults, inferenceRequestSignature, normalizeInferenceParameterSchema, type InferenceParameterSchema } from './lib/inference-parameters';
import { normalizeUiLanguage, UI_LANGUAGE_STORAGE_KEY, type UiLanguage } from './lib/i18n';
import { resolveFileStatusIndicator, selectFileProgressJob } from './lib/file-status-indicator';
import { resolveCurrentFilePath } from './lib/current-file-path';
import { defaultShortcutMap, displayShortcut, findShortcutConflict, isForbiddenShortcut, resolveShortcutAction, resolvedShortcutMap, sanitizeShortcutOverrides, shortcutAriaLabel, shortcutDefinitions, shortcutFromKeyboardEvent, type ShortcutActionId, type ShortcutOverrides } from './lib/keyboard-shortcuts';
import { remoteInferenceConsentMatches, requiresRemoteInferenceConfirmation, type RemoteInferenceConsentContext } from './lib/real-workflows';
import { groupInferencePredictionsByCategory, inferenceAnnotationsAreSegmentation, inferencePredictionKey, inferenceRasterMatchesSource, inferenceResultFromJobItem, latestInferenceJobForModel } from './lib/inference-results';
import { globalWorkspaceSettings, snapshotPipelineSettings, usablePipelineSettings } from './lib/workspace-settings';

type FileStatus = 'valid' | 'duplicate' | 'corrupt' | 'orphan';
type RightTab = 'layers' | 'pipeline' | 'inference' | 'agent';
type SearchMode = 'smart' | 'text' | 'regex' | 'query';
type PipelineScope = 'all' | 'current';
type AnnotationTool = 'select' | 'rect' | 'rotation' | 'polygon' | 'point' | 'line' | 'circle' | 'brush' | 'pan';
const shortcutToolActions: Partial<Record<ShortcutActionId, AnnotationTool>> = {
  'tool.select': 'select',
  'tool.pan': 'pan',
  'tool.rect': 'rect',
  'tool.rotation': 'rotation',
  'tool.polygon': 'polygon',
  'tool.point': 'point',
  'tool.line': 'line',
  'tool.circle': 'circle',
  'tool.brush': 'brush',
};
const toolShortcutActions: Record<AnnotationTool, ShortcutActionId | null> = {
  select: 'tool.select', pan: 'tool.pan', rect: 'tool.rect', rotation: 'tool.rotation', polygon: 'tool.polygon',
  point: 'tool.point', line: 'tool.line', circle: 'tool.circle', brush: 'tool.brush',
};
type SamPromptTool = 'positive' | 'negative' | 'box';
const ACTIVE_TASK_STATES = BACKGROUND_TASK_ACTIVE_STATES;
const DISMISSED_BACKGROUND_TASKS_KEY = 'labelone-dismissed-background-tasks-v1';
const BACKGROUND_TASK_HISTORY_HOURS_KEY = 'labelone-background-task-history-hours-v1';
const LAST_SELECTED_MODEL_KEY = 'labelone-last-selected-model-v1';
type SamPromptPoint = { x: number; y: number; label: 0 | 1 };
type SamPromptBox = [number, number, number, number];
type AgentMessageView = {
  id: string;
  role: 'agent' | 'user';
  text: string;
  source?: 'system' | 'live' | 'error';
  context?: {
    label: string;
    datasetId: string | null;
    assetId: string | null;
  };
  run?: AgentRunResult;
};

type FakeFile = {
  id: string;
  name: string;
  meta: string;
  status: FileStatus;
  annotations: number;
  annotationFileExists: boolean;
  variant: 'image';
  imagePath?: string;
  width?: number;
  height?: number;
  displayPath?: string;
  selectable?: boolean;
  rawStatus?: DatasetScanItem['status'];
  labels?: string[];
  shapeTypes?: string[];
};

type DatasetView = { id?: string; name: string; path: string; total: number; files: FakeFile[]; summary?: DatasetScanResult['summary'] };
const fileItemKey = (file: FakeFile) => file.id;
type TileMetadataState = { assetKey: string | null; phase: 'idle' | 'loading' | 'ready' | 'error'; data: TileMetadata | null };
type AnnotationPersistenceState = { phase: 'idle' | 'pending' | 'local' | 'saving' | 'saved' | 'offline' | 'error'; message?: string };
type AnnotationPersistOutcome = 'unchanged' | 'remote' | 'local' | 'busy' | 'failed';
type AnnotationRecoveryState = { kind: 'recoverable' | 'conflict'; local: PersistedAnnotationDraft; server: AnnotationEnvelope };
type PendingManualShape = { shape: AnnotationShape; datasetId: string; assetId: string; anchor: ScreenPoint };
type PendingShapeLabelEdit = { index: number; label: string; shapeType: string; datasetId: string; assetId: string };
type PendingCategoryLabelEdit = { category: string; count: number; datasetId: string; assetId: string; idempotencyKey: string };
type PendingAnnotationNavigation = {
  file: FakeFile;
  datasetId?: string;
  targetLabel: string;
  sourceDatasetId: string;
  sourceAssetId: string;
  sourceFingerprint: string;
};
type RemoteInferenceConfirmation = RemoteInferenceConsentContext & { fileName?: string; selectableCount?: number };
type FlowNode = PipelineFlowNode & { name: string };
type VisualizationNode = PipelineVisualizationNode & { name: string };
const CANVAS_CONTROL_SELECTOR = 'button:not(.annotation-box),input,textarea,select,[contenteditable="true"],.navigator,.pipeline-summary-popover,.raster-overlay-controls,.canvas-hint,.service-banner';
type ModelView = {
  id: string;
  name: string;
  task: string;
  runtime: string;
  badge: string;
  family: string;
  adapter?: string;
  predict?: boolean;
  capture: boolean;
  real?: boolean;
  availability?: string;
  availabilityReason?: string;
  captureMode?: 'none' | 'exported_outputs' | 'eager_hooks' | 'graph_rewrite' | 'remote';
  layers?: FeatureLayer[];
  parametersSchema: InferenceParameterSchema;
  runtimeState: 'unloaded' | 'loading' | 'loaded' | 'failed';
  usageCount: number;
  lastUsedAt: string | null;
};

const EMPTY_DATASET: DatasetView = { name: '未打开项目', path: '', total: 0, files: [] };
const EMPTY_MODEL: ModelView = { id: '', name: '未选择模型', task: '—', runtime: '—', badge: '—', family: '', capture: false, availability: 'unavailable', parametersSchema: {}, runtimeState: 'unloaded', usageCount: 0, lastUsedAt: null };
const LARGE_IMAGE_TILE_THRESHOLD_PIXELS = 4096 * 4096;
const PIPELINE_PREVIEW_FORMAT = 'webp' as const;

function nextGeneratedSequence(ids: string[], minimum = 2): number {
  return Math.max(minimum, ...ids.map((id) => Number(id.match(/-(\d+)$/)?.[1] ?? 0) + 1));
}

function canvasLabelPointsForShape(shape: AnnotationShape): number[][] {
  if (shape.shape_type === 'rectangle' && shape.points.length === 2) {
    return [[shape.points[0][0], shape.points[0][1]], [shape.points[1][0], shape.points[0][1]], [shape.points[1][0], shape.points[1][1]], [shape.points[0][0], shape.points[1][1]]];
  }
  if (shape.shape_type === 'circle' && shape.points.length >= 2) {
    const [center, edge] = shape.points;
    const radius = Math.hypot(edge[0] - center[0], edge[1] - center[1]);
    return [[center[0] - radius, center[1] - radius], [center[0] + radius, center[1] + radius]];
  }
  return shape.points;
}

function canvasLabelAnchorForShape(shape: AnnotationShape): { x: number; y: number; align: 'center' } | undefined {
  return shape.shape_type === 'rotation' ? canvasLabelTopVertexAnchor(shape.points) as { x: number; y: number; align: 'center' } | undefined : undefined;
}
const shapeTypeLabels: Record<string, string> = {
  rectangle: '矩形框',
  rotation: '旋转框',
  polygon: '多边形',
  point: '点',
  line: '直线',
  linestrip: '连续线',
  circle: '圆',
};

const operatorPalette = [
  { kind: 'crop', icon: '⌗', name: '智能裁剪', color: 'mint' },
  { kind: 'resize', icon: '↔', name: '缩放', color: 'blue' },
  { kind: 'flip', icon: '⇋', name: '翻转', color: 'violet' },
  { kind: 'rotate', icon: '↻', name: '直角旋转', color: 'amber' },
  { kind: 'color', icon: '◐', name: '颜色增强', color: 'blue' },
  { kind: 'noise', icon: '◇', name: '去噪锐化', color: 'violet' },
];

const initialNodes: FlowNode[] = [
  { id: 'source', kind: 'source', name: '原图像', enabled: true, parameters: {}, operator_version: 'pending' },
];

const initialVisualizations: VisualizationNode[] = [
  { id: 'visualize-1', kind: 'visualize', name: '显示', enabled: true, parameters: { label: '显示' }, operator_version: 'pending', tap_after_node_id: 'source' },
];

const fileStatusMap: Record<DatasetScanItem['status'], FileStatus> = {
  valid: 'valid',
  duplicate_match: 'duplicate',
  orphan_annotation: 'orphan',
  corrupt_image: 'corrupt',
  corrupt_annotation: 'corrupt',
};

function fileFromAsset(item: DatasetScanItem): FakeFile {
  return {
    id: item.asset_id,
    name: item.display_path.split('/').at(-1) ?? item.display_path,
    displayPath: item.display_path,
    meta: item.selectable
      ? `${item.annotation_count ?? 0} 标注 · ${item.width ?? '?'} × ${item.height ?? '?'}`
      : item.reason ?? item.issues.join(', '),
    status: fileStatusMap[item.status] ?? 'corrupt',
    rawStatus: item.status,
    selectable: item.selectable,
    annotations: item.annotation_count ?? 0,
    annotationFileExists: item.annotation_file_exists,
    variant: 'image',
    imagePath: item.image_path,
    width: item.width,
    height: item.height,
    labels: item.labels ?? [],
    shapeTypes: item.shape_types ?? [],
  };
}

function datasetFromScan(result: DatasetScanResult): DatasetView {
  return {
    id: result.dataset_id,
    name: result.root_dir.split('/').filter(Boolean).at(-1) ?? '本地数据集',
    path: result.root_dir,
    total: result.items.length,
    files: result.items.map(fileFromAsset),
    summary: result.summary,
  };
}

function datasetFromRegistered(dataset: RegisteredDataset, assets: AssetCursorPage): DatasetView {
  const view = datasetFromScan({
    dataset_id: dataset.dataset_id,
    root_dir: dataset.root_dir,
    image_root: dataset.image_root,
    annotation_roots: [],
    items: assets.items,
    summary: dataset.summary,
  });
  return { ...view, name: dataset.name, total: assets.total, summary: dataset.summary };
}

function IconButton({ children, label, description, shortcut, ariaShortcut, onClick, active = false, disabled = false }: { children: React.ReactNode; label: string; description?: string; shortcut?: string; ariaShortcut?: string; onClick?: () => void; active?: boolean; disabled?: boolean }) {
  return <button className={`icon-button ${active ? 'active' : ''}`} aria-label={label} aria-keyshortcuts={ariaShortcut} title={label} data-tooltip-title={description ? label : undefined} data-tooltip={description} data-shortcut={shortcut} disabled={disabled} onClick={onClick}>{children}</button>;
}

function ShapeTypeIcon({ shapeType }: { shapeType: string }) {
  const normalizedShapeType = shapeType === 'rect' ? 'rectangle' : shapeType === 'brush' ? 'linestrip' : shapeType;
  return <svg className={`annotation-shape-type-icon type-${normalizedShapeType}`} viewBox="0 0 20 20" aria-hidden="true">
    {normalizedShapeType === 'rectangle' ? <rect x="3.5" y="5" width="13" height="10" rx="1.2" />
      : normalizedShapeType === 'rotation' ? <rect x="4" y="5" width="12" height="10" rx="1.2" transform="rotate(-14 10 10)" />
        : normalizedShapeType === 'polygon' ? <path d="M10 3.2 16 6.6 15 13.6 8.8 16.4 3.8 12 4.7 5.8Z" />
          : normalizedShapeType === 'point' ? <circle className="solid" cx="10" cy="10" r="2.8" />
            : normalizedShapeType === 'line' ? <path d="M4 15.2 16 4.8" />
              : normalizedShapeType === 'linestrip' ? <path d="m3.5 14.5 4.2-6.2 3.5 3.8 5.3-6.6" />
                : normalizedShapeType === 'circle' ? <circle cx="10" cy="10" r="6.2" />
                  : <path d="M4 15 16 5" />}
  </svg>;
}

function VisibilityEyeIcon({ visible }: { visible: boolean }) {
  return <svg className="visibility-eye-icon" viewBox="0 0 20 20" aria-hidden="true">
    <path d="M2.5 10s2.7-4.2 7.5-4.2 7.5 4.2 7.5 4.2-2.7 4.2-7.5 4.2S2.5 10 2.5 10Z" />
    <circle cx="10" cy="10" r="2.1" />
    {!visible && <path className="visibility-eye-slash" d="M4 4l12 12" />}
  </svg>;
}

function SettingsIcon() {
  return <svg className="top-action-icon" viewBox="0 0 24 24" aria-hidden="true">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6h.08A1.65 1.65 0 0 0 10 3.09V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9v.08A1.65 1.65 0 0 0 20.91 10H21a2 2 0 1 1 0 4h-.09A1.65 1.65 0 0 0 19.4 15Z" />
  </svg>;
}

function FullscreenIcon({ active }: { active: boolean }) {
  return <svg className="top-action-icon" viewBox="0 0 20 20" aria-hidden="true">
    {active
      ? <path d="M8 3v5H3M12 3v5h5M8 17v-5H3M12 17v-5h5" />
      : <path d="M7.5 3H3v4.5M12.5 3H17v4.5M7.5 17H3v-4.5M12.5 17H17v-4.5" />}
  </svg>;
}

function BackgroundTasksIcon({ attention, complete }: { attention: boolean; complete: boolean }) {
  return <svg className={`top-action-icon background-tasks-icon ${attention ? 'attention' : complete ? 'complete' : ''}`} viewBox="0 0 20 20" aria-hidden="true">
    {complete && !attention
      ? <path className="task-complete-glyph" d="m4.2 10.2 3.1 3.1 8.5-8.2M4 16.2h12" />
      : <path className="task-download-glyph" d="M10 3.8v8m0 0 2.8-2.8M10 11.8 7.2 9M4.3 15.4h11.4" />}
  </svg>;
}

function PixelReadout({ cursor }: { cursor: { x: number; y: number; insideImage: boolean; pixel: PixelSample | null } }) {
  const pixel = cursor.insideImage ? cursor.pixel : null;
  const value = (channel: number | undefined) => channel === undefined ? '—' : String(channel);
  const ariaLabel = !cursor.insideImage
    ? '移入图像查看像素'
    : pixel
      ? `坐标 X ${cursor.x}，Y ${cursor.y}；红色 ${pixel.r}，绿色 ${pixel.g}，蓝色 ${pixel.b}，Alpha ${pixel.a}，明度 V ${pixel.v}`
      : `坐标 X ${cursor.x}，Y ${cursor.y}；当前显示像素不可读取`;
  return <output className={`pixel-readout ${pixel ? 'ready' : 'unavailable'}`} aria-label={ariaLabel}>
    <span className="pixel-readout-visual" aria-hidden="true">
      <span className="pixel-group pixel-coordinates"><span className="pixel-field"><i>X</i><data>{cursor.insideImage ? cursor.x.toLocaleString() : '—'}</data></span><span className="pixel-field"><i>Y</i><data>{cursor.insideImage ? cursor.y.toLocaleString() : '—'}</data></span></span>
      <span className="pixel-group pixel-channels"><span className="pixel-field"><i className="channel-r">R</i><data>{value(pixel?.r)}</data></span><span className="pixel-field"><i className="channel-g">G</i><data>{value(pixel?.g)}</data></span><span className="pixel-field"><i className="channel-b">B</i><data>{value(pixel?.b)}</data></span><span className="pixel-field"><i className="channel-a">A</i><data>{value(pixel?.a)}</data></span><span className="pixel-field"><i className="channel-v">V</i><data>{value(pixel?.v)}</data></span></span>
    </span>
  </output>;
}

function annotationCategoryStyle(label: string, colorOverrides: Record<string, string> = {}): React.CSSProperties {
  const category = normalizeAnnotationCategory(label);
  const customStroke = Object.hasOwn(colorOverrides, category) ? colorOverrides[category] : undefined;
  const colors = annotationCategoryColors(label, customStroke);
  return {
    '--shape-color': colors.stroke,
    '--shape-fill': colors.fill,
    '--shape-label': colors.labelBackground,
  } as React.CSSProperties;
}

function sanitizeAnnotationCategoryColorOverrides(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value).flatMap(([label, rawColor]) => {
    const color = typeof rawColor === 'string' ? normalizeAnnotationCategoryColor(rawColor) : null;
    return color ? [[normalizeAnnotationCategory(label), color]] : [];
  }));
}

function formatTensorShape(shape: Array<number | string | null>) {
  return `[${shape.map((dimension) => dimension ?? '?').join(' × ')}]`;
}

function formatArtifactBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function parameterDefaults(contract: PipelineOperatorContract | undefined) {
  if (!contract) return {};
  return Object.fromEntries(Object.entries(contract.parameters_schema.properties).flatMap(([name, schema]) => 'default' in schema ? [[name, schema.default]] : []));
}

const featureProjectionValue = { 'PCA-1': 'pca1', Mean: 'mean', Max: 'max', 'Single Channel': 'channel', 'Token Grid': 'token_grid', None: 'none' } as const;
function PipelineFeatureConfiguration({ node, models, featureLayers, featureRuntimeModelId, onOpenInference }: {
  node: FlowNode;
  models: ModelView[];
  featureLayers: FeatureLayer[];
  featureRuntimeModelId?: string;
  onOpenInference: () => void;
}) {
  const modelId = typeof node.parameters.model_id === 'string' ? node.parameters.model_id : '';
  const layerId = typeof node.parameters.layer_id === 'string' ? node.parameters.layer_id : '';
  const model = models.find((item) => item.id === modelId);
  const layer = featureRuntimeModelId === modelId ? featureLayers.find((item) => item.id === layerId) : undefined;
  return <div className="pipeline-feature-node-controls linked">
    <div className="pipeline-feature-linked-summary"><span>由推理选项卡统一配置</span><strong>{(model?.name ?? modelId) || '尚未选择模型'}</strong><small>{layer ? `${layer.group} / ${layer.name} · ${featureTensorKindLabel(featureTensorKind(layer))} · ${formatTensorShape(layer.shape)}` : layerId || '尚未选择中间层'}</small></div>
    <button type="button" className="pipeline-feature-open-inference" onClick={onOpenInference}>前往推理配置 <span aria-hidden="true">→</span></button>
  </div>;
}

function PipelineParameterEditor({
  node,
  contract,
  registryHash,
  onChange,
  onEnabledChange,
  models = [],
  featureLayers = [],
  featureRuntimeModelId,
  onOpenInference,
  inputWidth,
  inputHeight,
}: {
  node: FlowNode | VisualizationNode | undefined;
  contract: PipelineOperatorContract | undefined;
  registryHash?: string;
  onChange: (name: string, value: unknown) => void;
  onEnabledChange?: (enabled: boolean) => void;
  models?: ModelView[];
  featureLayers?: FeatureLayer[];
  featureRuntimeModelId?: string;
  onOpenInference?: () => void;
  inputWidth?: number;
  inputHeight?: number;
}) {
  if (!node) return null;
  const properties = contract?.parameters_schema.properties ?? {};
  const modelFeature = node.kind === 'model_feature';
  const crop = node.kind === 'crop';
  const featureParameterNames = ['model_id', 'layer_id', 'projection', 'normalization', 'interpolation', 'channel', 'gain', 'gamma', 'spatial_scale', 'clip'];
  const cropRegionParameterNames = ['x', 'y', 'width', 'height'];
  const cropUsesExactRegion = cropRegionParameterNames.some((name) => node.parameters[name] !== undefined);
  const parameterEntries = Object.entries(properties).filter(([name]) => {
    if (modelFeature && featureParameterNames.includes(name)) return false;
    if (!crop) return true;
    return cropUsesExactRegion ? cropRegionParameterNames.includes(name) : name === 'margin_ratio';
  });
  const setCropMode = (exact: boolean) => {
    if (!crop || exact === cropUsesExactRegion) return;
    if (exact) {
      onChange('margin_ratio', undefined);
      onChange('x', 0);
      onChange('y', 0);
      onChange('width', Math.max(1, Math.floor(inputWidth ?? 1)));
      onChange('height', Math.max(1, Math.floor(inputHeight ?? 1)));
      return;
    }
    for (const name of cropRegionParameterNames) onChange(name, undefined);
    onChange('margin_ratio', typeof properties.margin_ratio?.default === 'number' ? properties.margin_ratio.default : 0.05);
  };
  const parameterContext = { inputWidth, inputHeight, parameters: node.parameters, schemas: properties };
  return <div className="node-params"><div className="section-title"><div><h2>{contract?.title ?? node.name}</h2><small>{contract?.input_type ?? 'image'} → {contract?.output_type ?? 'image'} · v{node.operator_version}</small></div>{onEnabledChange ? <label className="selected-node-enabled"><input type="checkbox" checked={node.enabled} onChange={(event) => onEnabledChange(event.target.checked)} /><span>启用</span></label> : <span>{Object.keys(properties).length} 个参数</span>}</div><p className="operator-description">{contract?.description ?? '该算子暂未提供说明。'}</p>{crop && <div className="crop-parameter-mode" role="group" aria-label="裁剪配置方式"><button type="button" className={!cropUsesExactRegion ? 'active' : ''} aria-pressed={!cropUsesExactRegion} onClick={() => setCropMode(false)}>边缘比例</button><button type="button" className={cropUsesExactRegion ? 'active' : ''} aria-pressed={cropUsesExactRegion} onClick={() => setCropMode(true)}>精确区域</button><small>{cropUsesExactRegion ? `输入 ${inputWidth ?? '?'} × ${inputHeight ?? '?'} px` : '按四周相同比例裁剪'}</small></div>}{modelFeature && onOpenInference && <PipelineFeatureConfiguration node={node as FlowNode} models={models} featureLayers={featureLayers} featureRuntimeModelId={featureRuntimeModelId} onOpenInference={onOpenInference} />}{parameterEntries.map(([name, schema]) => <PipelineParameterControl key={name} name={name} label={schema.title ?? name} schema={schema} value={node.parameters[name]} context={parameterContext} onChange={onChange} />)}{Object.keys(properties).length === 0 && <div className="parameter-empty">该算子没有可配置参数。</div>}<div className="operator-contract-summary"><span>{contract?.input_type ?? 'image'} → {contract?.output_type ?? 'image'}</span><span>角色：{contract?.node_role ?? (node.kind === 'source' ? 'source' : node.kind === 'visualize' ? 'visualization' : 'transform')}</span><span>尺寸：{contract?.size_behavior ?? 'unknown'}</span><span>标注：{String(contract?.annotation_policy?.mode ?? 'unknown')}</span><span>空间：{String(contract?.annotation_policy?.spatial_behavior ?? 'unknown')} · {contract?.annotation_policy?.synchronized === true ? '标签同步' : '未验证'}</span><code title={registryHash}>registry {registryHash?.slice(0, 12) ?? '未连接'}</code></div></div>;
}

function formatArtifactNumber(value: number) {
  if (!Number.isFinite(value)) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude > 0 && (magnitude < 0.001 || magnitude >= 10000)) return value.toExponential(3);
  return Number(value.toFixed(4)).toLocaleString();
}

function summarizeFeatureTransform(transform: Record<string, unknown>) {
  const hiddenDisplayOnly = new Set(['gain', 'gamma', 'spatial_scale', 'interpolation', 'max_output_elements']);
  const entries = Object.entries(transform).filter(([key, value]) => !hiddenDisplayOnly.has(key) && value !== null && value !== undefined);
  if (!entries.length) return '未声明 transform';
  return entries.map(([key, value]) => {
    const rendered = Array.isArray(value) ? `[${value.join(', ')}]` : typeof value === 'object' ? JSON.stringify(value) : String(value);
    return `${key}=${rendered}`;
  }).join(' · ');
}

function summarizeRasterMetadata(metadata: Record<string, unknown>) {
  const entries = Object.entries(metadata).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) return '无附加 metadata';
  return entries.slice(0, 6).map(([key, value]) => {
    const rendered = Array.isArray(value) ? `[${value.join(', ')}]` : typeof value === 'object' ? JSON.stringify(value) : String(value);
    return `${key}=${rendered.length > 72 ? `${rendered.slice(0, 69)}…` : rendered}`;
  }).join(' · ');
}

function backgroundTaskTitle(job: JobRecord) {
  if (job.kind === 'pipeline') return '批量处理流';
  if (job.kind === 'model_download') return `模型权重 · ${job.request.model_id ?? '未命名模型'}`;
  if (job.kind === 'category_rename') return `类别重命名 · ${job.request.source_category ?? '未命名'} → ${job.request.target_category ?? '未命名'}`;
  return `批量推理 · ${job.request.model_id ?? '未命名模型'}`;
}

function backgroundTaskStateLabel(state: JobRecord['state']) {
  return ({
    queued: '排队中', running: '运行中', pausing: '正在暂停', paused: '已暂停', canceling: '正在取消', canceled: '已取消',
    succeeded: '已完成', succeeded_with_errors: '部分完成', failed: '失败', interrupted: '已中断',
  } as Record<JobRecord['state'], string>)[state];
}

function backgroundTaskPercent(job: JobRecord) {
  const terminal = job.completed + job.failed + job.canceled;
  return job.total ? Math.round(terminal / job.total * 100) : 0;
}

function AgentToolResultView({ result }: { result: AgentToolResult }) {
  const data = result.data;
  if (result.tool === 'dataset.stats') {
    const rows: Array<[string, unknown]> = [
      ['有效', data.valid],
      ['异常', data.visible_abnormal],
      ['孤立 JSON', data.orphan_annotation],
      ['损坏文件', Number(data.corrupt_image ?? 0) + Number(data.corrupt_annotation ?? 0)],
    ];
    return <dl className="agent-result-metrics">{rows.map(([label, value]) => <div key={String(label)}><dt>{label}</dt><dd>{String(value ?? 0)}</dd></div>)}</dl>;
  }
  if (result.tool === 'dataset.search') {
    const items = Array.isArray(data.items) ? data.items.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object')).slice(0, 4) : [];
    return <div className="agent-search-result"><strong>命中 {String(data.total ?? 0)} 项</strong>{items.map((item, index) => <span key={`${String(item.asset_id ?? index)}`}>{String(item.display_path ?? item.asset_id ?? '未命名图片')}</span>)}</div>;
  }
  if (result.tool === 'annotation.qa') {
    const issues = data.issues && typeof data.issues === 'object' && !Array.isArray(data.issues) ? data.issues as Record<string, unknown> : {};
    return <div className="agent-qa-result"><strong>{String(data.shape_count ?? 0)} 个对象</strong><div>{Object.entries(issues).map(([name, count]) => <span key={name}>{name} · {String(count)}</span>)}{Object.keys(issues).length === 0 && <span>未发现问题</span>}</div></div>;
  }
  if (result.tool === 'dataset.distribution') {
    const labels = data.labels && typeof data.labels === 'object' && !Array.isArray(data.labels) ? data.labels as Record<string, unknown> : {};
    return <div className="agent-distribution-result"><strong>标签 Top {Math.min(5, Object.keys(labels).length)}</strong><div>{Object.entries(labels).slice(0, 5).map(([label, count]) => <span key={label}>{label}<b>{String(count)}</b></span>)}</div></div>;
  }
  return <code className="agent-result-fallback">{JSON.stringify(data)}</code>;
}

function pipelinePreviewContainedRect(pane: HTMLElement) {
  const content = pane.querySelector<HTMLElement>('.pipeline-preview-pane-content') ?? pane;
  return containedPipelineImageRect(
    content.getBoundingClientRect(),
    Number(pane.dataset.pipelineWidth),
    Number(pane.dataset.pipelineHeight),
  );
}

function PipelinePixelGridCanvas({ imageWidth, imageHeight, view, referenceWidth, referenceHeight, enabled }: { imageWidth: number; imageHeight: number; view: { scale: number; x: number; y: number }; referenceWidth: number; referenceHeight: number; enabled: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const pane = canvas?.parentElement;
    if (!canvas || !pane) return;
    let frame = 0;
    const draw = () => {
      frame = 0;
      const width = pane.clientWidth;
      const height = pane.clientHeight;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const backingWidth = Math.max(1, Math.round(width * dpr));
      const backingHeight = Math.max(1, Math.round(height * dpr));
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      const context = canvas.getContext('2d');
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      const metrics = pipelinePaneMetrics(width, height, imageWidth, imageHeight, view, referenceWidth, referenceHeight, enabled);
      if (!metrics.pixelGridVisible || metrics.display.width <= 0 || metrics.display.height <= 0) return;
      const stepX = metrics.display.width / imageWidth;
      const stepY = metrics.display.height / imageHeight;
      const startX = Math.max(0, Math.ceil(-metrics.display.left / stepX));
      const endX = Math.min(imageWidth, Math.floor((width - metrics.display.left) / stepX));
      const startY = Math.max(0, Math.ceil(-metrics.display.top / stepY));
      const endY = Math.min(imageHeight, Math.floor((height - metrics.display.top) / stepY));
      context.save();
      context.beginPath();
      context.rect(Math.max(0, metrics.display.left), Math.max(0, metrics.display.top), Math.min(width, metrics.display.left + metrics.display.width) - Math.max(0, metrics.display.left), Math.min(height, metrics.display.top + metrics.display.height) - Math.max(0, metrics.display.top));
      context.clip();
      context.beginPath();
      for (let index = startX; index <= endX; index += 1) {
        const x = snapPipelineGridCoordinate(metrics.display.left + index * stepX, dpr);
        context.moveTo(x, Math.max(0, metrics.display.top));
        context.lineTo(x, Math.min(height, metrics.display.top + metrics.display.height));
      }
      for (let index = startY; index <= endY; index += 1) {
        const y = snapPipelineGridCoordinate(metrics.display.top + index * stepY, dpr);
        context.moveTo(Math.max(0, metrics.display.left), y);
        context.lineTo(Math.min(width, metrics.display.left + metrics.display.width), y);
      }
      context.strokeStyle = 'rgba(215,230,240,.38)';
      context.lineWidth = 1 / dpr;
      context.stroke();
      context.restore();
    };
    const schedule = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(pane);
    schedule();
    return () => {
      observer.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [enabled, imageHeight, imageWidth, referenceHeight, referenceWidth, view]);
  return <canvas ref={canvasRef} className="pipeline-pixel-grid-canvas" aria-hidden="true" />;
}

export default function Home() {
  const backend = useLocalBackend();
  const loadTileMetadata = backend.loadTileMetadata;
  const buildTileUrl = backend.tileUrl;
  const searchDatasetAssets = backend.searchDatasetAssets;
  const cancelAssetSearch = backend.cancelAssetSearch;
  const lookupJobItems = backend.lookupJobItems;
  const ensurePipelinePrecompute = backend.ensurePipelinePrecompute;
  const watchPipelineJobEvents = backend.watchJobEvents;
  const controlPipelineJob = backend.controlJob;
  const prioritizeJobItems = backend.prioritizeJobItems;
  const listModelWeights = backend.listModelWeights;
  const loadModelLayers = backend.loadModelLayers;
  const refreshPipelineRegistry = backend.refreshPipelineRegistry;
  const importPipelineOperator = backend.importPipelineOperator;
  const inspectPipelineOperator = backend.inspectPipelineOperator;
  const invalidatePipelinePreview = backend.invalidatePipelinePreview;
  const validatePipeline = backend.validatePipeline;
  const invalidatePipelineValidation = backend.invalidatePipelineValidation;
  const prefetchPipelinePreview = backend.prefetchPipelinePreview;
  const prefetchAnnotation = backend.prefetchAnnotation;
  const buildAssetUrl = backend.assetUrl;
  const cancelPipelinePrefetch = backend.cancelPipelinePrefetch;
  const restorePipelinePreview = backend.restorePipelinePreview;
  const previewPipeline = backend.previewPipeline;
  const pipelineArtifactUrl = backend.pipelineArtifactUrl;
  const pipelineValidationState = backend.pipelineValidation;
  const refreshBackendModels = backend.refreshModels;
  const refreshBackendHealth = backend.refreshHealth;
  const refreshBackendDatasets = backend.refreshDatasets;
  const refreshApplicationSettings = backend.refreshApplicationSettings;
  const revalidateDatasetAsset = backend.revalidateDatasetAsset;
  const getDatasetSettings = backend.getDatasetSettings;
  const saveDatasetSettings = backend.saveDatasetSettings;
  const getDatasetAsset = backend.getDatasetAsset;
  const updateApplicationSettings = backend.updateApplicationSettings;
  const recordModelUsage = backend.recordModelUsage;
  const backendJobs = backend.jobs.data.jobs;
  const [openedDataset, setOpenedDataset] = useState<DatasetView | null>(null);
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [currentFileId, setCurrentFileId] = useState('');
  const [openingRecentDatasetId, setOpeningRecentDatasetId] = useState<string | null>(null);
  const [welcomeError, setWelcomeError] = useState('');
  const [recentProjectIds, setRecentProjectIds] = useState<string[]>([]);
  const [filter, setFilter] = useState<FileAnnotationFilter>('all');
  const [search, setSearch] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>('smart');
  const [searchHelp, setSearchHelp] = useState(false);
  const [rightTab, setRightTab] = useState<RightTab>('layers');
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>('zh-CN');
  const [languageAnnouncement, setLanguageAnnouncement] = useState('');
  const [tool, setTool] = useState<AnnotationTool>('select');
  const [canvasPanning, setCanvasPanning] = useState(false);
  const [, setNavigatorLayoutVersion] = useState(0);
  const [selectedShape, setSelectedShape] = useState('rectangle');
  const [selectedPredictionIndex, setSelectedPredictionIndex] = useState<number | null>(null);
  const [objectSourceTab, setObjectSourceTab] = useState<'manual' | 'ai'>('manual');
  const [showGT, setShowGT] = useState(true);
  const [hiddenPredictionCategories, setHiddenPredictionCategories] = useState<Set<string>>(() => new Set());
  const [hiddenPredictionKeys, setHiddenPredictionKeys] = useState<Set<string>>(() => new Set());
  const [showMasks, setShowMasks] = useState(true);
  const [showClassifications, setShowClassifications] = useState(true);
  const [showPixel, setShowPixel] = useState(true);
  const [hiddenShapeIndexes, setHiddenShapeIndexes] = useState<Set<number>>(() => new Set());
  const [annotationCategoryColorOverrides, setAnnotationCategoryColorOverrides] = useState<Record<string, string>>({});
  const [opacity, setOpacity] = useState(88);
  const [view, setView] = useState({ scale: 1.15, x: 0, y: 0 });
  const [zoomEditing, setZoomEditing] = useState(false);
  const [zoomDraft, setZoomDraft] = useState('');
  const [spaceDown, setSpaceDown] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number; insideImage: boolean; pixel: PixelSample | null }>({ x: 3842, y: 2156, insideImage: false, pixel: null });
  const [pipelineSharedCursor, setPipelineSharedCursor] = useState<PipelineSharedCursor | null>(null);
  const [toast, setToast] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState<GlobalSettingsSection>('models');
  const [shortcutOverrides, setShortcutOverrides] = useState<ShortcutOverrides>({});
  const [recordingShortcut, setRecordingShortcut] = useState<ShortcutActionId | null>(null);
  const [shortcutFeedback, setShortcutFeedback] = useState('');
  const [useMacShortcutSymbols, setUseMacShortcutSymbols] = useState(true);
  const [modelWeightsPathInput, setModelWeightsPathInput] = useState('');
  const [modelSettingsSaving, setModelSettingsSaving] = useState(false);
  const [modelSettingsStatus, setModelSettingsStatus] = useState('');
  const [modelDownloadSource, setModelDownloadSource] = useState('auto');
  const [modelDirectoryPicking, setModelDirectoryPicking] = useState<'weights' | null>(null);
  const [cloudAiDraft, setCloudAiDraft] = useState<CloudAiDraft>({ enabled: false, provider: 'openai_compatible', endpoint: '', model: '', api_key_env: 'OPENAI_API_KEY', timeout_seconds: 30, max_output_tokens: 800 });
  const [cloudAiSaving, setCloudAiSaving] = useState(false);
  const [cloudAiStatus, setCloudAiStatus] = useState('');
  const [networkProxyDraft, setNetworkProxyDraft] = useState<NetworkProxySettings>({ mode: 'system', url: '', bypass: 'localhost,127.0.0.1,::1' });
  const [networkProxySaving, setNetworkProxySaving] = useState(false);
  const [networkProxyStatus, setNetworkProxyStatus] = useState('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [nodes, setNodes] = useState(initialNodes);
  const [visualizations, setVisualizations] = useState(initialVisualizations);
  const [selectedNode, setSelectedNode] = useState('source');
  const [pipelineInsertGapKey, setPipelineInsertGapKey] = useState<string | null>(null);
  const [operatorSearch, setOperatorSearch] = useState('');
  const [operatorImporting, setOperatorImporting] = useState(false);
  const [operatorImportStatus, setOperatorImportStatus] = useState('');
  const [operatorCandidateFile, setOperatorCandidateFile] = useState<File | null>(null);
  const [operatorInspection, setOperatorInspection] = useState<PipelineOperatorInspection | null>(null);
  const [completedPipelineSignature, setCompletedPipelineSignature] = useState<string | null>(null);
  const [validatedPipelineKey, setValidatedPipelineKey] = useState<string | null>(null);
  const [pipelineTimingByNode, setPipelineTimingByNode] = useState<Record<string, { milliseconds: number; samples?: number }>>({});
  const [pipelineConstraintMessage, setPipelineConstraintMessage] = useState('');
  const [pipelinePrecomputeJob, setPipelinePrecomputeJob] = useState<{ key: string; jobId: string } | null>(null);
  const [automaticallyPausedPipelineJobs, setAutomaticallyPausedPipelineJobs] = useState<Set<string>>(() => new Set());
  const [visualizationDisplayMode, setVisualizationDisplayMode] = useState<PipelineDisplayMode>('split');
  const [singlePipelineSource, setSinglePipelineSource] = useState('source');
  const [visualizationLayerState, setVisualizationLayerState] = useState<Record<string, { visible: boolean; opacity: number }>>({});
  const [pipelineEnabled, setPipelineEnabled] = useState(true);
  const [pipelineSummaryOpen, setPipelineSummaryOpen] = useState(false);
  const [pipelineScope, setPipelineScope] = useState<PipelineScope>('current');
  const [pipelineImageEpoch, setPipelineImageEpoch] = useState(0);
  const [pipelineImageAttempts, setPipelineImageAttempts] = useState<Record<string, number>>({});
  const [modelLoaded, setModelLoaded] = useState(false);
  const [inferenceProvider, setInferenceProvider] = useState('CPUExecutionProvider');
  const [inferenceParameters, setInferenceParameters] = useState<Record<string, unknown>>({});
  const [remoteInferenceConfirmation, setRemoteInferenceConfirmation] = useState<RemoteInferenceConfirmation | null>(null);
  const [modelDownloadActionPending, setModelDownloadActionPending] = useState(false);
  const [modelLoadError, setModelLoadError] = useState('');
  const [modelTask, setModelTask] = useState('全部');
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelStatusRefreshing, setModelStatusRefreshing] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [globalWorkspaceHydrated, setGlobalWorkspaceHydrated] = useState(false);
  const [datasetWorkspaceHydratedId, setDatasetWorkspaceHydratedId] = useState<string | null>(null);
  const [workspaceSettingsSaving, setWorkspaceSettingsSaving] = useState(false);
  const [workspaceSettingsStatus, setWorkspaceSettingsStatus] = useState('');
  const [selectedLayerId, setSelectedLayerId] = useState('');
  const [projection, setProjection] = useState('PCA-1');
  const [normalization, setNormalization] = useState('Min-Max');
  const [featureChannel, setFeatureChannel] = useState(0);
  const [featureClip, setFeatureClip] = useState('p1p99');
  const [selectedRasterId, setSelectedRasterId] = useState<string | null>(null);
  const [completedInferenceSignature, setCompletedInferenceSignature] = useState('');
  const [rasterOpacity, setRasterOpacity] = useState(58);
  const [samPromptMode, setSamPromptMode] = useState(false);
  const [samPromptTool, setSamPromptTool] = useState<SamPromptTool>('positive');
  const [samPoints, setSamPoints] = useState<SamPromptPoint[]>([]);
  const [samBoxes, setSamBoxes] = useState<SamPromptBox[]>([]);
  const [samBoxPreview, setSamBoxPreview] = useState<SamPromptBox | null>(null);
  const [scanRegistering, setScanRegistering] = useState(false);
  const [directoryPickerPending, setDirectoryPickerPending] = useState<'image' | null>(null);
  const [autoScanIntent, setAutoScanIntent] = useState<{ sessionId: string; name: string; operationId: number } | null>(null);
  const [autoOpenError, setAutoOpenError] = useState('');
  const [rootPath, setRootPath] = useState('');
  const [annotationDraft, setAnnotationDraft] = useState<AnnotationEnvelope | null>(null);
  const [selectedShapeIndex, setSelectedShapeIndex] = useState<number | null>(null);
  const [annotationSaving, setAnnotationSaving] = useState(false);
  const [annotationDirty, setAnnotationDirty] = useState(false);
  const [annotationHistoryVersion, setAnnotationHistoryVersion] = useState(0);
  const [annotationEditActive, setAnnotationEditActive] = useState(false);
  const [annotationAutoSave, setAnnotationAutoSave] = useState(DEFAULT_ANNOTATION_AUTO_SAVE);
  const [annotationPersistence, setAnnotationPersistence] = useState<AnnotationPersistenceState>({ phase: 'idle' });
  const [annotationRecovery, setAnnotationRecovery] = useState<AnnotationRecoveryState | null>(null);
  const [pendingAnnotationNavigation, setPendingAnnotationNavigation] = useState<PendingAnnotationNavigation | null>(null);
  const [annotationNavigationDecision, setAnnotationNavigationDecision] = useState<'keep' | 'discard' | null>(null);
  const [annotationNavigationError, setAnnotationNavigationError] = useState('');
  const [drawPreview, setDrawPreview] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [polygonDraft, setPolygonDraft] = useState<AnnotationPoint[]>([]);
  const [polygonPointer, setPolygonPointer] = useState<AnnotationPoint | null>(null);
  const [polygonCloseReady, setPolygonCloseReady] = useState(false);
  const [brushPreview, setBrushPreview] = useState<AnnotationPoint[]>([]);
  const [pendingManualShape, setPendingManualShape] = useState<PendingManualShape | null>(null);
  const [pendingShapeLabelEdit, setPendingShapeLabelEdit] = useState<PendingShapeLabelEdit | null>(null);
  const [pendingCategoryLabelEdit, setPendingCategoryLabelEdit] = useState<PendingCategoryLabelEdit | null>(null);
  const [categoryRenameCreating, setCategoryRenameCreating] = useState(false);
  const [manualShapeLabel, setManualShapeLabel] = useState('');
  const [lastManualLabel, setLastManualLabel] = useState('');
  const [manualLabelMenuGeometry, setManualLabelMenuGeometry] = useState({ width: 300, height: 320, viewportWidth: 0, viewportHeight: 0 });
  const [tileMetadataState, setTileMetadataState] = useState<TileMetadataState>({ assetKey: null, phase: 'idle', data: null });
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [taskActionPending, setTaskActionPending] = useState<string | null>(null);
  const [taskStreamOpen, setTaskStreamOpen] = useState(false);
  const [taskStreamHovered, setTaskStreamHovered] = useState(false);
  const [taskStreamFocused, setTaskStreamFocused] = useState(false);
  const taskStreamVisible = taskStreamOpen || taskStreamHovered || taskStreamFocused;
  const [dismissedTaskVersions, setDismissedTaskVersions] = useState<Record<string, string>>({});
  const [taskHistoryHours, setTaskHistoryHours] = useState<BackgroundTaskHistoryHours>(24);
  const taskStreamJobs = useMemo(() => filterBackgroundTaskHistory(backendJobs, dismissedTaskVersions, taskHistoryHours)
    .sort((left, right) => {
      const priority = (job: JobRecord) => ACTIVE_TASK_STATES.has(job.state) ? 0 : BACKGROUND_TASK_ATTENTION_STATES.has(job.state) ? 1 : 2;
      return priority(left) - priority(right) || Date.parse(right.updated_at) - Date.parse(left.updated_at);
    }).slice(0, 8), [backendJobs, dismissedTaskVersions, taskHistoryHours]);
  const activeTaskCount = taskStreamJobs.filter((job) => ACTIVE_TASK_STATES.has(job.state)).length;
  const attentionTaskCount = taskStreamJobs.filter((job) => BACKGROUND_TASK_ATTENTION_STATES.has(job.state)).length;
  const latestBackgroundTask = taskStreamJobs[0] ?? null;
  const clearableTaskIds = clearableCompletedTaskIds(taskStreamJobs);
  const [agentInput, setAgentInput] = useState('');
  const [agentSending, setAgentSending] = useState(false);
  const [agentProposalPending, setAgentProposalPending] = useState<string | null>(null);
  const [agentConfirmingProposal, setAgentConfirmingProposal] = useState<string | null>(null);
  const [agentExecutionErrors, setAgentExecutionErrors] = useState<Record<string, string>>({});
  const [agentMessages, setAgentMessages] = useState<AgentMessageView[]>([]);

  const appShellRef = useRef<HTMLElement>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const settingsCloseRef = useRef<HTMLButtonElement>(null);
  const modelPickerTriggerRef = useRef<HTMLButtonElement>(null);
  const modelSelectionOperationRef = useRef(0);
  const currentInferenceSignatureRef = useRef('');
  const lastAutoInferenceSignatureRef = useRef('');
  const autoInferenceRunnerRef = useRef<() => void>(() => undefined);
  const zoomValueInputRef = useRef<HTMLInputElement>(null);
  const pipelineInsertAnchorRef = useRef<HTMLButtonElement>(null);
  const globalSettingsHydratedRef = useRef(false);
  const remoteWorkspaceHydratedRef = useRef(false);
  const datasetWorkspaceFingerprintRef = useRef<string | null>(null);
  const datasetWorkspaceRevisionRef = useRef(0);
  const datasetWorkspaceSaveChainRef = useRef<Promise<unknown>>(Promise.resolve());
  const datasetWorkspaceSaveTimerRef = useRef<number | null>(null);
  const globalInferenceFingerprintRef = useRef<string | null>(null);
  const globalInferenceSaveTimerRef = useRef<number | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const rightSidebarRef = useRef<HTMLElement>(null);
  const inferenceFeatureCardRef = useRef<HTMLElement>(null);
  const annotationObjectListRef = useRef<HTMLDivElement>(null);
  const taskActivityRef = useRef<HTMLElement>(null);
  const taskActivityButtonRef = useRef<HTMLButtonElement>(null);
  const imageRef = useRef<HTMLDivElement>(null);
  const navigatorImageRef = useRef<HTMLDivElement>(null);
  const navigatorDragRef = useRef<{ pointerId: number } | null>(null);
  const canvasCrosshairRef = useRef<HTMLDivElement>(null);
  const pixelSampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pixelSampleFrameRef = useRef<number | null>(null);
  const pixelSampleGenerationRef = useRef(0);
  const lastPointerMoveAtRef = useRef(0);
  const pendingPixelSampleRef = useRef<{ clientX: number; clientY: number; x: number; y: number; pane: HTMLElement | null } | null>(null);
  const manualLabelMenuRef = useRef<HTMLFormElement>(null);
  const lastCanvasPointerRef = useRef<ScreenPoint>({ x: 0, y: 0 });
  const pendingManualShapeRef = useRef<PendingManualShape | null>(null);
  const selectOnNextCanvasBlankRef = useRef(false);
  const dragRef = useRef<{ startX: number; startY: number; viewX: number; viewY: number } | null>(null);
  const gestureZoomRef = useRef<{ lastScale: number } | null>(null);
  const toastTimer = useRef<number | null>(null);
  const nextNodeId = useRef(2);
  const nextVisualizationId = useRef(2);
  const shapeDragRef = useRef<{ index: number; startX: number; startY: number; points: number[][] } | null>(null);
  const controlPointDragRef = useRef<{ shapeIndex: number; pointIndex: number } | null>(null);
  const realRotationDragRef = useRef<{ shapeIndex: number; center: AnnotationPoint; startAngle: number; startDirection: number; shape: AnnotationShape } | null>(null);
  const drawRef = useRef<{ startX: number; startY: number } | null>(null);
  const polygonDraftRef = useRef<AnnotationPoint[]>([]);
  const brushRef = useRef<{ pointerId: number; points: AnnotationPoint[] } | null>(null);
  const samBoxDragRef = useRef<{ startX: number; startY: number } | null>(null);
  const weightDownloadPendingRef = useRef(new Set<string>());
  const completedWeightJobsRef = useRef(new Set<string>());
  const autoLoadAfterDownloadRef = useRef<string | null>(null);
  const completedCategoryRenameJobsRef = useRef(new Set<string>());
  const autoOpenedScanSessionsRef = useRef(new Set<string>());
  const progressivelyOpenedScanSessionsRef = useRef(new Set<string>());
  const scanFinalizeAttemptsRef = useRef(new Map<string, number>());
  const finishAutoScanRef = useRef<(sessionId: string, name: string, operationId: number, progressive?: boolean) => Promise<void>>(async () => undefined);
  const directoryPickerInFlightRef = useRef(false);
  const datasetOperationRef = useRef(0);
  const sessionHydratedRef = useRef(false);
  const annotationLoadRequestId = useRef(0);
  const annotationLoaderRef = useRef<(file: FakeFile, datasetId?: string) => Promise<void>>(async () => undefined);
  const annotationDraftRef = useRef<AnnotationEnvelope | null>(null);
  const annotationDirtyRef = useRef(false);
  const annotationBaseDocumentRef = useRef<AnnotationEnvelope['document'] | null>(null);
  const annotationHistoryRef = useRef<AnnotationHistory<AnnotationEnvelope['document']> | null>(null);
  const annotationSavePendingRef = useRef(false);
  const annotationAutoSaveTimerRef = useRef<number | null>(null);
  const annotationAutoSaveRef = useRef(DEFAULT_ANNOTATION_AUTO_SAVE);
  const annotationEditActiveRef = useRef(false);
  const backendModeRef = useRef(backend.mode);
  const saveAnnotationRef = useRef(backend.saveAnnotation);
  const operatorPackageInputRef = useRef<HTMLInputElement | null>(null);
  const pipelinePreviewCacheRef = useRef(new Map<string, PipelinePreviewResult>());
  const pipelinePrefetchGenerationRef = useRef(0);
  const pipelinePrecomputeGenerationRef = useRef(0);
  const pipelinePrecomputePausePendingRef = useRef(new Set<string>());
  const pipelineImageRetryTimersRef = useRef(new Map<string, number>());
  const neighborImageCacheRef = useRef(new Map<string, HTMLImageElement>());
  const shortcuts = useMemo(() => resolvedShortcutMap(shortcutOverrides), [shortcutOverrides]);
  annotationDraftRef.current = annotationDraft;
  annotationDirtyRef.current = annotationDirty;
  pendingManualShapeRef.current = pendingManualShape;
  polygonDraftRef.current = polygonDraft;
  annotationAutoSaveRef.current = annotationAutoSave;
  backendModeRef.current = backend.mode;
  saveAnnotationRef.current = backend.saveAnnotation;
  useEffect(() => {
    const stored = normalizeUiLanguage(window.localStorage.getItem(UI_LANGUAGE_STORAGE_KEY));
    setUiLanguage(stored);
  }, []);
  const toggleUiLanguage = () => {
    const next: UiLanguage = uiLanguage === 'zh-CN' ? 'en' : 'zh-CN';
    window.localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, next);
    setUiLanguage(next);
    setLanguageAnnouncement(next === 'en' ? 'Language changed to English' : '语言已切换为中文');
  };
  const dataset = openedDataset ?? EMPTY_DATASET;
  const datasetWorkspaceReady = !dataset.id || datasetWorkspaceHydratedId === dataset.id;
  const recentProjects = useMemo(() => {
    const rank = new Map(recentProjectIds.map((id, index) => [id, index]));
    return [...backend.datasets.data.datasets].sort((left, right) => {
      const leftRank = rank.get(left.dataset_id);
      const rightRank = rank.get(right.dataset_id);
      if (leftRank !== undefined || rightRank !== undefined) return (leftRank ?? Number.MAX_SAFE_INTEGER) - (rightRank ?? Number.MAX_SAFE_INTEGER);
      return right.updated_at.localeCompare(left.updated_at);
    }).slice(0, 8);
  }, [backend.datasets.data.datasets, recentProjectIds]);
  const isRealDataset = Boolean(dataset.id);
  const activeCategoryRenameJob = backendJobs.find((job) => job.kind === 'category_rename' && job.dataset_id === dataset.id && ACTIVE_TASK_STATES.has(job.state));
  const realFiles = useMemo(() => backend.assetSearchDatasetId === dataset.id ? backend.assetSearch.data.items.map(fileFromAsset) : [], [backend.assetSearch.data.items, backend.assetSearchDatasetId, dataset.id]);
  const validFiles = (isRealDataset ? realFiles : dataset.files).filter((file) => file.status === 'valid' && (!isRealDataset || file.selectable === true));
  const requestedCurrentFile = isRealDataset
    ? realFiles.find((file) => file.id === currentFileId) ?? dataset.files.find((file) => file.id === currentFileId)
    : dataset.files.find((file) => file.id === currentFileId);
  const currentFile = requestedCurrentFile?.status === 'valid' && (!isRealDataset || requestedCurrentFile.selectable === true) ? requestedCurrentFile : validFiles[0];
  const currentFilePath = resolveCurrentFilePath({ datasetPath: dataset.path, fileName: currentFile?.name, displayPath: currentFile?.displayPath, imagePath: currentFile?.imagePath });
  const canCopyAbsoluteImagePath = Boolean(currentFile?.imagePath && currentFilePath.isAbsolute);
  const validTotal = isRealDataset && dataset.summary ? dataset.summary.valid : dataset.total;
  const liveModels = useMemo<ModelView[]>(() => backend.models.data.models.map((model) => ({
    id: model.id,
    name: model.display_name,
    task: model.task,
    runtime: model.runtime.join(' / '),
    badge: model.adapter === 'yolo_detection_onnx' ? 'YOLO' : model.adapter === 'onnx_raw' ? 'ONNX' : '计划',
    family: model.family,
    adapter: model.adapter,
    predict: model.capabilities.predict,
    capture: model.capabilities.feature_capture.mode !== 'none',
    real: true,
    availability: model.availability.state,
    availabilityReason: model.availability.reason,
    captureMode: model.capabilities.feature_capture.mode,
    layers: model.capabilities.feature_capture.layers,
    parametersSchema: normalizeInferenceParameterSchema(model.capabilities.parameters_schema),
    runtimeState: backend.models.data.status_by_model?.[model.id]?.runtime_state ?? 'unloaded',
    usageCount: backend.models.data.status_by_model?.[model.id]?.usage_count ?? 0,
    lastUsedAt: backend.models.data.status_by_model?.[model.id]?.last_used_at ?? null,
  })), [backend.models.data.models, backend.models.data.status_by_model]);
  const displayedModelCatalog = useMemo(() => backend.mode === 'online' && backend.models.phase === 'ready' ? liveModels : [], [backend.mode, backend.models.phase, liveModels]);
  const modelTasks = useMemo(() => [...new Set(displayedModelCatalog.map((model) => model.task))].sort((left, right) => left.localeCompare(right, 'zh-Hans-CN')), [displayedModelCatalog]);
  const selectedModel = displayedModelCatalog.find((model) => model.id === selectedModelId) ?? displayedModelCatalog[0] ?? EMPTY_MODEL;
  const isSamModel = backend.mode === 'online' && selectedModel.adapter === 'segment_anything_onnx';
  const isTrustedRemoteModel = backend.mode === 'online' && requiresRemoteInferenceConfirmation(selectedModel.adapter);
  const samPromptCount = samPoints.length + samBoxes.length;
  const samPromptParameters = { points: samPoints, boxes: samBoxes };
  const runtimeLayers = backend.runtime.data?.model_id === selectedModel.id
    ? backend.runtime.data.layers
    : [];
  const availableLayers = runtimeLayers.filter((layer) => layer.captureable !== false);
  const selectedLayer = availableLayers.find((layer) => layer.id === selectedLayerId);
  const captureFeatures = Boolean(selectedLayer);
  const selectedFeatureKind = featureTensorKind(selectedLayer);
  const selectedFeatureProjectionOptions = featureProjectionOptions(selectedLayer);
  const selectedFeatureChannelCount = Number(selectedLayer?.shape[selectedFeatureKind === 'spatial-map' ? 1 : selectedFeatureKind === 'token-sequence' ? 2 : 1] ?? 1);
  useEffect(() => {
    if (backend.models.phase !== 'ready' || !globalWorkspaceHydrated) return;
    if (displayedModelCatalog.some((model) => model.id === selectedModelId)) return;
    const next = displayedModelCatalog.find((model) => model.availability === 'available' && model.predict)
      ?? displayedModelCatalog[0];
    setSelectedModelId(next?.id ?? '');
    lastAutoInferenceSignatureRef.current = '';
    setCompletedInferenceSignature('');
    setInferenceParameters(next ? inferenceParameterDefaults(next.parametersSchema) : {});
    setSelectedLayerId('');
  }, [backend.models.phase, displayedModelCatalog, globalWorkspaceHydrated, selectedModelId]);
  useEffect(() => {
    if (backend.runtime.data?.model_id !== selectedModel.id || backend.runtime.data.state !== 'loaded') return;
    setSelectedLayerId((current) => backend.runtime.data?.layers.some((layer) => layer.id === current) ? current : '');
  }, [backend.runtime.data, selectedModel.id]);
  useEffect(() => {
    setFeatureChannel(0);
    setProjection((current) => selectedFeatureProjectionOptions.includes(current) ? current : defaultFeatureProjection(selectedLayer));
  // Projection defaults follow tensor rank; vector layers never inherit spatial-only controls.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLayerId, selectedFeatureKind]);
  useEffect(() => {
    if (backend.mode !== 'online' || backend.runtime.phase === 'loading' || !selectedModel.id || backend.runtime.data?.model_id === selectedModel.id) return;
    void loadModelLayers(selectedModel.id).catch(() => undefined);
  }, [backend.mode, backend.runtime.data?.model_id, backend.runtime.phase, loadModelLayers, selectedModel.id]);
  const selectedOperator = nodes.find((node) => node.id === selectedNode) ?? visualizations.find((node) => node.id === selectedNode) ?? nodes[0];
  const selectedVisualization = visualizations.find((node) => node.id === selectedNode);
  const pipelineLinearNodes = useMemo(() => pipelineLinearItems(nodes, visualizations), [nodes, visualizations]);
  const pipelineGaps = useMemo(() => pipelineInsertionGaps(nodes, visualizations), [nodes, visualizations]);
  const serializedPipelineNodes = useMemo(() => serializePipelineNodes(nodes, visualizations), [nodes, visualizations]);
  const currentPipelineSignature = useMemo(() => pipelineSignature(nodes, visualizations), [nodes, visualizations]);
  const persistedPipelineSettings = useMemo(() => snapshotPipelineSettings({
    enabled: pipelineEnabled,
    scope: pipelineScope,
    nodes,
    visualizations,
    displayMode: visualizationDisplayMode,
    singleSource: singlePipelineSource,
    layerState: visualizationLayerState,
  }), [nodes, pipelineEnabled, pipelineScope, singlePipelineSource, visualizationDisplayMode, visualizationLayerState, visualizations]);
  const pipelineRequestNodes = useMemo(() => serializedPipelineNodes.map((node) => ({ id: node.id, kind: node.kind, enabled: node.enabled, parameters: node.parameters })), [serializedPipelineNodes]);
  const hasPipelineValidationDimensions = Boolean(currentFile?.width && currentFile.width > 0 && currentFile?.height && currentFile.height > 0);
  const pipelineValidationWidth = hasPipelineValidationDimensions ? currentFile!.width : undefined;
  const pipelineValidationHeight = hasPipelineValidationDimensions ? currentFile!.height : undefined;
  const pipelineCatalogRegistryHash = backend.pipelineRegistry.data?.registry_hash ?? 'registry-pending';
  const currentPipelineValidationKey = pipelineValidationKey(`${currentPipelineSignature}:registry-${pipelineCatalogRegistryHash}`, pipelineValidationWidth, pipelineValidationHeight);
  const searchedDatasetIndexRevision = backend.assetSearchDatasetId === dataset.id && backend.assetSearch.data.index_revision > 0 ? backend.assetSearch.data.index_revision : undefined;
  const currentDatasetIndexRevision = searchedDatasetIndexRevision
    ?? backend.datasets.data.datasets.find((item) => item.dataset_id === dataset.id)?.index_revision
    ?? 0;
  const currentPipelineRegistryHash = validatedPipelineKey === currentPipelineValidationKey ? pipelineValidationState.data?.registry_hash ?? pipelineCatalogRegistryHash : pipelineCatalogRegistryHash;
  const currentPipelineExecutionSignature = `${currentPipelineSignature}:index-${currentDatasetIndexRevision}:registry-${currentPipelineRegistryHash}`;
  const currentPipelinePrecomputeKey = dataset.id ? pipelinePrecomputeKey(dataset.id, `${currentPipelineExecutionSignature}:format-${PIPELINE_PREVIEW_FORMAT}`) : null;
  const pipelineValidationReady = validatedPipelineKey === currentPipelineValidationKey && pipelineValidationState.phase === 'ready' && Boolean(pipelineValidationState.data?.valid);
  const pipelineValidationLabel = backend.mode !== 'online' ? '服务离线'
    : !pipelineEnabled ? '流程关闭'
      : pipelineValidationState.phase === 'loading' ? '校验中'
        : pipelineValidationReady ? !dataset.id ? '定义已通过 · 未打开数据集'
          : !currentFile?.id ? '定义已通过 · 未选择图像'
            : pipelineScope === 'current' ? '已通过 · 实时当前图' : '已通过 · 当前图实时 + 全库后台预计算'
          : pipelineValidationState.phase === 'error' ? pipelineValidationState.error?.message ?? '校验失败'
            : pipelineValidationState.phase === 'ready' && pipelineValidationState.data && !pipelineValidationState.data.valid ? pipelineValidationState.data.message ?? pipelineValidationState.data.errors?.[0]?.message ?? '图定义无效'
              : '等待校验';
  const pipelineExecutionError = pipelineEnabled && backend.pipeline.phase === 'error' ? backend.pipeline.error?.message ?? '处理流执行失败' : '';
  const pipelineVisibleError = pipelineConstraintMessage || (pipelineValidationState.phase === 'error' || pipelineValidationState.data?.valid === false ? pipelineValidationLabel : '') || pipelineExecutionError;
  const pipelineValidationIndicatorState = pipelineVisibleError
    ? 'invalid'
    : pipelineValidationState.phase === 'loading' ? 'loading'
      : pipelineValidationReady ? 'valid' : 'idle';
  const pipelineValidationIndicatorIcon = pipelineValidationIndicatorState === 'valid' ? '✓'
    : pipelineValidationIndicatorState === 'invalid' ? '!'
      : pipelineValidationIndicatorState === 'loading' ? '◌' : '○';
  const pipelineValidationIndicatorText = pipelineVisibleError || pipelineValidationLabel;
  const requestPipelineValidation = useCallback(async () => {
    const result = await validatePipeline({
      nodes: pipelineRequestNodes,
      ...(pipelineValidationWidth !== undefined && pipelineValidationHeight !== undefined ? { width: pipelineValidationWidth, height: pipelineValidationHeight } : {}),
    });
    if (result.valid) setValidatedPipelineKey(currentPipelineValidationKey);
    return result;
  }, [currentPipelineValidationKey, pipelineRequestNodes, pipelineValidationHeight, pipelineValidationWidth, validatePipeline]);
  const pipelineHasTile = nodes.some((node) => node.enabled && node.kind === 'tile');
  const pipelineContracts = backend.pipelineRegistry.data?.operators ?? [];
  const selectedOperatorContract = pipelineContracts.find((contract) => contract.kind === selectedOperator?.kind);
  const pipelineHasModelFeature = nodes.some((node) => node.kind === 'model_feature');
  const pipelineFeatureNode = nodes.find((node) => node.kind === 'model_feature');
  const pipelineFailureNeedsModel = Boolean(pipelineVisibleError && pipelineHasModelFeature && /model|layer|runtime|onnx|weight|load|模型|中间层|加载/i.test(pipelineVisibleError));
  const pipelineFailureDetail = pipelineFailureNeedsModel
    ? '模型尚未加载，无法生成中间层结果。请前往推理页加载模型后重试。'
    : pipelineVisibleError;
  const selectedPipelineFeatureModelId = selectedOperator?.kind === 'model_feature' && typeof selectedOperator.parameters.model_id === 'string' ? selectedOperator.parameters.model_id : '';
  const pipelineFeatureRuntimeLayers = backend.runtime.data?.model_id === selectedPipelineFeatureModelId ? backend.runtime.data.layers : [];
  const pipelinePalette = backend.mode === 'online' && pipelineContracts.length
    ? pipelineContracts.filter((contract) => contract.node_role ? contract.node_role === 'transform' : !['source', 'output', 'visualize'].includes(contract.kind)).map((contract, index) => {
      const presentation = operatorPalette.find((item) => item.kind === contract.kind);
      return { kind: contract.kind, icon: presentation?.icon ?? '◇', name: contract.title, color: presentation?.color ?? ['mint', 'blue', 'violet', 'amber'][index % 4], ...(contract.kind === 'model_feature' && pipelineHasModelFeature ? { disabled: true, disabledReason: '当前处理流已有模型中间层；请先删除或配置现有节点' } : {}) };
    })
    : operatorPalette;
  const filteredPipelinePalette = useMemo(() => {
    const query = operatorSearch.trim().toLocaleLowerCase();
    if (!query) return pipelinePalette;
    return pipelinePalette.filter((item) => `${item.name} ${item.kind}`.toLocaleLowerCase().includes(query));
  }, [operatorSearch, pipelinePalette]);
  const visualizationContract = pipelineContracts.find((contract) => contract.kind === 'visualize');
  const visualizationName = visualizationContract?.title ?? '显示';
  const visualizationMatchesOperatorSearch = !operatorSearch.trim() || `${visualizationName} visualize`.toLocaleLowerCase().includes(operatorSearch.trim().toLocaleLowerCase());

  const realSearchInput = useMemo<DatasetAssetSearchInput>(() => ({
    q: search,
    mode: searchMode === 'query' ? 'condition' : searchMode,
    cursor: null,
    limit: 100,
    status: 'valid',
    has_annotation_file: filter === 'with_json' ? true : filter === 'without_json' ? false : undefined,
  }), [filter, search, searchMode]);
  const realSearchKey = dataset.id ? `${dataset.id}:${JSON.stringify(realSearchInput)}` : 'no-dataset';
  const listedFiles = realFiles;
  const listedFileIds = useMemo(() => listedFiles.slice(0, 200).map((file) => file.id), [listedFiles]);
  const fileProgressJob = selectFileProgressJob(
    backend.jobs.data.jobs,
    dataset.id,
    backend.jobEvents.job_id,
    { pipelineEnabled, pipelineNodes: pipelineRequestNodes },
  );
  const currentBatchInferenceJob = latestInferenceJobForModel(backendJobs, dataset.id, selectedModel.id);
  const searchError = isRealDataset ? backend.assetSearch.error?.message ?? '' : '';
  const matchedTotal = isRealDataset && backend.assetSearchDatasetId === dataset.id ? backend.assetSearch.data.total : 0;
  const latestTaskLiveProgress = latestBackgroundTask ? Object.values(backend.jobItemProgress).find((progress) => progress.job_id === latestBackgroundTask.job_id) : undefined;
  const latestTaskDownloadView = latestTaskLiveProgress ? formatDownloadProgress(latestTaskLiveProgress.received_bytes, latestTaskLiveProgress.total_bytes) : null;
  const latestTaskPercent = latestBackgroundTask ? latestTaskDownloadView?.percent ?? backgroundTaskPercent(latestBackgroundTask) : 0;
  const taskIconVisible = taskStreamJobs.length > 0 || taskStreamVisible;
  const taskIconComplete = taskStreamJobs.length > 0 && activeTaskCount === 0 && attentionTaskCount === 0;
  const taskButtonLabel = activeTaskCount > 0
    ? `后台任务，${activeTaskCount} 项运行中，最新进度 ${latestTaskDownloadView?.percent === null ? '未知' : `${latestTaskPercent}%`}`
    : attentionTaskCount > 0 ? `后台任务，${attentionTaskCount} 项需要处理`
      : taskIconComplete ? `后台任务，最近任务已完成` : '后台任务';
  const reusablePipelinePrecomputeRecord = currentPipelinePrecomputeKey ? backendJobs.find((job) => job.kind === 'pipeline'
    && job.dataset_id === dataset.id
    && job.request.priority === 'background'
    && job.request.output_policy?.mode === 'preview'
    && !job.request.asset_ids?.length
    && job.request.pipeline_context?.dataset_index_revision === currentDatasetIndexRevision
    && job.request.pipeline_context.registry_hash === currentPipelineRegistryHash
    && job.request.pipeline_context.output_format === PIPELINE_PREVIEW_FORMAT
    && pipelineRequestNodesEqual(job.request.pipeline_nodes, pipelineRequestNodes)) : undefined;
  const currentPipelinePrecomputeJob = useMemo(() => pipelinePrecomputeJob?.key === currentPipelinePrecomputeKey
    ? pipelinePrecomputeJob
    : reusablePipelinePrecomputeRecord && currentPipelinePrecomputeKey ? { key: currentPipelinePrecomputeKey, jobId: reusablePipelinePrecomputeRecord.job_id } : null, [currentPipelinePrecomputeKey, pipelinePrecomputeJob, reusablePipelinePrecomputeRecord]);
  const currentPipelinePrecomputeRecord = currentPipelinePrecomputeJob ? backendJobs.find((job) => job.job_id === currentPipelinePrecomputeJob.jobId) : undefined;
  const currentPipelinePrecomputeState = currentPipelinePrecomputeRecord?.state;
  const pipelinePrecomputeMayAutoResume = Boolean(currentPipelinePrecomputeRecord
    && ['paused', 'interrupted'].includes(currentPipelinePrecomputeRecord.state)
    && automaticallyPausedPipelineJobs.has(currentPipelinePrecomputeRecord.job_id));
  const neighborPipelineAssetIds = useMemo(() => neighboringAssetIds(listedFileIds, currentFile?.id ?? '', 4), [currentFile?.id, listedFileIds]);
  const preferredPipelineAssetIds = useMemo(() => currentFile ? [currentFile.id, ...neighborPipelineAssetIds] : neighborPipelineAssetIds, [currentFile, neighborPipelineAssetIds]);
  useEffect(() => {
    if (backend.mode !== 'online' || !dataset.id || neighborPipelineAssetIds.length === 0) return;
    let canceled = false;
    const timer = window.setTimeout(() => {
      for (const assetId of neighborPipelineAssetIds) {
        if (canceled) return;
        void prefetchAnnotation(dataset.id!, assetId).catch(() => undefined);
        const key = `${dataset.id}:${assetId}`;
        if (neighborImageCacheRef.current.has(key)) continue;
        const url = buildAssetUrl(dataset.id!, assetId, 'thumbnail', '?max_size=2048&format=webp');
        if (!url) continue;
        const image = new Image();
        image.decoding = 'async';
        image.src = url;
        neighborImageCacheRef.current.set(key, image);
        while (neighborImageCacheRef.current.size > 12) {
          neighborImageCacheRef.current.delete(neighborImageCacheRef.current.keys().next().value!);
        }
        void image.decode().catch(() => undefined);
      }
    }, 80);
    return () => {
      canceled = true;
      window.clearTimeout(timer);
    };
  }, [backend.mode, buildAssetUrl, dataset.id, neighborPipelineAssetIds, prefetchAnnotation]);
  const cachePipelinePreview = useCallback((result: PipelinePreviewResult, signature: string) => {
    const key = pipelinePreviewCacheKey(result.dataset_id, result.asset_id, signature);
    storeCachedPipelinePreview(pipelinePreviewCacheRef.current, key, result);
    const artifactIds = new Set([result.artifact_id, ...(result.visualizations ?? []).map((item) => item.artifact_id)]);
    for (const artifactId of artifactIds) {
      const url = pipelineArtifactUrl(artifactId);
      if (!url) continue;
      const image = new Image();
      image.decoding = 'async';
      image.crossOrigin = 'anonymous';
      image.src = url;
      void image.decode().catch(() => undefined);
    }
  }, [pipelineArtifactUrl]);
  const resetPipelineImageAttempts = useCallback(() => {
    for (const timer of pipelineImageRetryTimersRef.current.values()) window.clearTimeout(timer);
    pipelineImageRetryTimersRef.current.clear();
    setPipelineImageAttempts({});
    setPipelineImageEpoch((epoch) => epoch + 1);
  }, []);
  useEffect(() => {
    if (backend.pipeline.phase !== 'ready' || !backend.pipeline.data) return;
    resetPipelineImageAttempts();
  }, [backend.pipeline, resetPipelineImageAttempts]);
  useEffect(() => () => {
    for (const timer of pipelineImageRetryTimersRef.current.values()) window.clearTimeout(timer);
    pipelineImageRetryTimersRef.current.clear();
  }, []);
  const handlePipelineImageError = useCallback((url: string) => {
    if (pipelineImageRetryTimersRef.current.has(url)) return;
    const nextAttempt = (pipelineImageAttempts[url] ?? 0) + 1;
    if (pipelineImageRetryExhausted(nextAttempt)) {
      setPipelineImageAttempts((old) => ({ ...old, [url]: nextAttempt }));
      return;
    }
    const delay = nextAttempt === 1 ? 180 : 650;
    const timer = window.setTimeout(() => {
      pipelineImageRetryTimersRef.current.delete(url);
      setPipelineImageAttempts((old) => ({ ...old, [url]: Math.max(old[url] ?? 0, nextAttempt) }));
    }, delay);
    pipelineImageRetryTimersRef.current.set(url, timer);
  }, [pipelineImageAttempts]);
  const mergePipelineTimings = useCallback((result: PipelinePreviewResult) => {
    const averages = result.operator_average_timings_ms ?? result.operator_timings_ms;
    const counts = result.timing_sample_count;
    setPipelineTimingByNode((old) => ({ ...old, ...Object.fromEntries(Object.entries(averages).map(([key, milliseconds]) => [key, {
      milliseconds,
      samples: typeof counts === 'number' ? counts : counts?.[key],
    }])) }));
  }, []);

  useEffect(() => {
    if (!taskStreamVisible) return;
    const closeOutside = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && !taskActivityRef.current?.contains(target)) {
        setTaskStreamOpen(false);
        setTaskStreamHovered(false);
        setTaskStreamFocused(false);
      }
    };
    const closeWithEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setTaskStreamOpen(false);
      setTaskStreamHovered(false);
      setTaskStreamFocused(false);
      window.requestAnimationFrame(() => taskActivityButtonRef.current?.focus());
    };
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('keydown', closeWithEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('keydown', closeWithEscape);
    };
  }, [taskStreamVisible]);

  useEffect(() => {
    try {
      const dismissed = JSON.parse(window.localStorage.getItem(DISMISSED_BACKGROUND_TASKS_KEY) ?? '{}');
      if (dismissed && typeof dismissed === 'object' && !Array.isArray(dismissed)) {
        setDismissedTaskVersions(Object.fromEntries(Object.entries(dismissed).filter(([jobId, updatedAt]) => jobId && typeof updatedAt === 'string')));
      }
      const storedHours = Number(window.localStorage.getItem(BACKGROUND_TASK_HISTORY_HOURS_KEY));
      if (BACKGROUND_TASK_HISTORY_HOURS.includes(storedHours as BackgroundTaskHistoryHours)) setTaskHistoryHours(storedHours as BackgroundTaskHistoryHours);
    } catch { /* keep safe defaults */ }
  }, []);

  useEffect(() => {
    if (!dataset.id || backend.mode !== 'online') return;
    const timer = window.setTimeout(() => {
      void searchDatasetAssets(dataset.id!, realSearchInput).catch(() => undefined);
    }, 300);
    return () => {
      window.clearTimeout(timer);
      cancelAssetSearch();
    };
  }, [backend.mode, cancelAssetSearch, dataset.id, realSearchInput, searchDatasetAssets]);

  const loadMoreRealFiles = useCallback(() => {
    if (!dataset.id || backend.mode !== 'online' || backend.assetSearchLoadingMore || backend.assetSearch.data.next_cursor == null) return;
    void searchDatasetAssets(dataset.id, { ...realSearchInput, cursor: backend.assetSearch.data.next_cursor }, true).catch(() => undefined);
  }, [backend.assetSearch.data.next_cursor, backend.assetSearchLoadingMore, backend.mode, dataset.id, realSearchInput, searchDatasetAssets]);

  useEffect(() => {
    if (backend.mode !== 'online' || !fileProgressJob || listedFileIds.length === 0) return;
    void lookupJobItems(fileProgressJob.job_id, listedFileIds).catch(() => undefined);
  }, [backend.mode, fileProgressJob, listedFileIds, lookupJobItems]);

  useEffect(() => {
    if (backend.mode !== 'online' || !currentBatchInferenceJob || !currentFile?.id) return;
    void lookupJobItems(currentBatchInferenceJob.job_id, [currentFile.id]).catch(() => undefined);
  }, [backend.mode, currentBatchInferenceJob, currentFile?.id, lookupJobItems]);


  useEffect(() => {
    if (backend.mode !== 'online') return;
    void refreshPipelineRegistry().then((registry) => {
      if (!registry) return;
      const contracts = new Map(registry.operators.map((contract) => [contract.kind, contract]));
      setNodes((old) => old.map((node) => {
        const contract = contracts.get(node.kind);
        return contract ? {
          ...node,
          name: contract.title,
          operator_version: contract.version,
          parameters: { ...parameterDefaults(contract), ...node.parameters },
        } : node;
      }));
      setVisualizations((old) => old.map((node) => {
        const contract = contracts.get('visualize');
        return contract ? {
          ...node,
          name: contract.title,
          operator_version: contract.version,
          parameters: { ...parameterDefaults(contract), ...node.parameters },
        } : node;
      }));
    }).catch(() => undefined);
  }, [backend.mode, refreshPipelineRegistry]);

  useEffect(() => {
    invalidatePipelinePreview();
    cancelPipelinePrefetch();
    pipelinePrefetchGenerationRef.current += 1;
  }, [cancelPipelinePrefetch, currentFile?.id, currentPipelineExecutionSignature, dataset.id, invalidatePipelinePreview]);

  useEffect(() => {
    if (!datasetWorkspaceReady || backend.mode !== 'online' || !pipelineEnabled || pipelineHasTile || !dataset.id || !currentFile?.id || !currentFile.imagePath) return;
    if (completedPipelineSignature === currentPipelineExecutionSignature && backend.pipeline.data?.dataset_id === dataset.id && backend.pipeline.data.asset_id === currentFile.id) return;
    const key = pipelinePreviewCacheKey(dataset.id, currentFile.id, currentPipelineExecutionSignature);
    const cached = takeCachedPipelinePreview(pipelinePreviewCacheRef.current, key);
    if (!cached) return;
    restorePipelinePreview(cached);
    mergePipelineTimings(cached);
    setCompletedPipelineSignature(currentPipelineExecutionSignature);
  }, [backend.mode, backend.pipeline.data?.asset_id, backend.pipeline.data?.dataset_id, completedPipelineSignature, currentFile?.id, currentFile?.imagePath, currentPipelineExecutionSignature, dataset.id, datasetWorkspaceReady, mergePipelineTimings, pipelineEnabled, pipelineHasTile, restorePipelinePreview]);

  useEffect(() => {
    invalidatePipelineValidation();
    setValidatedPipelineKey(null);
  }, [currentPipelineValidationKey, invalidatePipelineValidation]);

  useEffect(() => {
    if (!datasetWorkspaceReady || backend.mode !== 'online' || !pipelineEnabled) return;
    void requestPipelineValidation().catch(() => undefined);
  }, [backend.mode, datasetWorkspaceReady, pipelineEnabled, requestPipelineValidation]);

  useEffect(() => {
    if (!datasetWorkspaceReady || backend.mode !== 'online' || !pipelineEnabled || pipelineHasTile || !dataset.id || !currentFile?.id || !currentFile.imagePath || validatedPipelineKey !== currentPipelineValidationKey || !pipelineValidationState.data?.valid) return;
    if (completedPipelineSignature === currentPipelineExecutionSignature && backend.pipeline.data?.dataset_id === dataset.id && backend.pipeline.data.asset_id === currentFile.id) return;
    const cacheKey = pipelinePreviewCacheKey(dataset.id, currentFile.id, currentPipelineExecutionSignature);
    if (takeCachedPipelinePreview(pipelinePreviewCacheRef.current, cacheKey)) return;
    void previewPipeline({ dataset_id: dataset.id, asset_id: currentFile.id, priority: 'interactive', nodes: pipelineRequestNodes }).then((result) => {
      cachePipelinePreview(result, currentPipelineExecutionSignature);
      mergePipelineTimings(result);
      setCompletedPipelineSignature(currentPipelineExecutionSignature);
    }).catch(() => undefined);
  }, [backend.mode, backend.pipeline.data?.asset_id, backend.pipeline.data?.dataset_id, cachePipelinePreview, completedPipelineSignature, currentFile?.id, currentFile?.imagePath, currentPipelineExecutionSignature, currentPipelineValidationKey, dataset.id, datasetWorkspaceReady, mergePipelineTimings, pipelineEnabled, pipelineHasTile, pipelineRequestNodes, pipelineScope, pipelineValidationState.data?.valid, previewPipeline, restorePipelinePreview, validatedPipelineKey]);

  useEffect(() => {
    if (!datasetWorkspaceReady || backend.mode !== 'online' || pipelineScope !== 'current' || !dataset.id || !currentFile?.id || completedPipelineSignature !== currentPipelineExecutionSignature || backend.pipeline.data?.asset_id !== currentFile.id || backend.pipeline.data.dataset_id !== dataset.id) return;
    const generation = ++pipelinePrefetchGenerationRef.current;
    let stopped = false;
    const run = async () => {
      for (const assetId of neighborPipelineAssetIds) {
        if (stopped || generation !== pipelinePrefetchGenerationRef.current) return;
        const key = pipelinePreviewCacheKey(dataset.id!, assetId, currentPipelineExecutionSignature);
        if (pipelinePreviewCacheRef.current.has(key)) continue;
        try {
          const result = await prefetchPipelinePreview({ dataset_id: dataset.id!, asset_id: assetId, priority: 'background', nodes: pipelineRequestNodes });
          if (stopped || generation !== pipelinePrefetchGenerationRef.current) return;
          cachePipelinePreview(result, currentPipelineExecutionSignature);
          mergePipelineTimings(result);
        } catch {
          if (stopped || generation !== pipelinePrefetchGenerationRef.current) return;
        }
      }
    };
    void run();
    return () => {
      stopped = true;
      pipelinePrefetchGenerationRef.current += 1;
      cancelPipelinePrefetch();
    };
  }, [backend.mode, backend.pipeline.data?.asset_id, backend.pipeline.data?.dataset_id, cachePipelinePreview, cancelPipelinePrefetch, completedPipelineSignature, currentFile?.id, currentPipelineExecutionSignature, dataset.id, datasetWorkspaceReady, mergePipelineTimings, neighborPipelineAssetIds, pipelineRequestNodes, pipelineScope, prefetchPipelinePreview]);

  useEffect(() => {
    if (!datasetWorkspaceReady
      || backend.mode !== 'online'
      || !pipelineEnabled
      || pipelineScope !== 'all'
      || !dataset.id
      || !currentPipelinePrecomputeKey
      || !pipelineValidationReady
      || pipelineHasTile
      || (currentPipelinePrecomputeJob && !pipelinePrecomputeMayAutoResume)) return;
    const generation = ++pipelinePrecomputeGenerationRef.current;
    const timer = window.setTimeout(() => {
      void ensurePipelinePrecompute({
        kind: 'pipeline',
        dataset_id: dataset.id!,
        concurrency: 3,
        priority: 'background',
        preferred_asset_ids: preferredPipelineAssetIds,
        pipeline_nodes: pipelineRequestNodes,
        output_policy: { mode: 'preview', image_format: PIPELINE_PREVIEW_FORMAT, conflict: 'reuse' },
      }).then(({ job, resumed }) => {
        if (generation !== pipelinePrecomputeGenerationRef.current) return;
        setPipelinePrecomputeJob({ key: currentPipelinePrecomputeKey, jobId: job.job_id });
        if (resumed) setAutomaticallyPausedPipelineJobs((old) => { const next = new Set(old); next.delete(job.job_id); return next; });
        watchPipelineJobEvents(job.job_id);
      }).catch(() => undefined);
    }, 520);
    return () => {
      window.clearTimeout(timer);
      pipelinePrecomputeGenerationRef.current += 1;
    };
  }, [backend.mode, currentPipelinePrecomputeJob, currentPipelinePrecomputeKey, dataset.id, datasetWorkspaceReady, ensurePipelinePrecompute, pipelineEnabled, pipelineHasTile, pipelinePrecomputeMayAutoResume, pipelineRequestNodes, pipelineScope, pipelineValidationReady, preferredPipelineAssetIds, watchPipelineJobEvents]);

  useEffect(() => {
    if (!datasetWorkspaceReady || backend.mode !== 'online') return;
    for (const job of backendJobs) {
      const context = job.request.pipeline_context;
      const automaticFullPreview = job.kind === 'pipeline'
        && job.request.priority === 'background'
        && job.request.output_policy?.mode === 'preview'
        && !job.request.asset_ids?.length
        && Boolean(context);
      if (!automaticFullPreview || !['queued', 'running'].includes(job.state) || pipelinePrecomputePausePendingRef.current.has(job.job_id)) continue;
      const stillCurrent = pipelineEnabled
        && pipelineScope === 'all'
        && !pipelineHasTile
        && context!.dataset_index_revision === currentDatasetIndexRevision
        && context!.registry_hash === currentPipelineRegistryHash
        && context!.output_format === PIPELINE_PREVIEW_FORMAT
        && pipelineRequestNodesEqual(job.request.pipeline_nodes, pipelineRequestNodes);
      if (stillCurrent) continue;
      pipelinePrecomputePausePendingRef.current.add(job.job_id);
      void controlPipelineJob(job.job_id, 'pause').then(() => setAutomaticallyPausedPipelineJobs((old) => new Set(old).add(job.job_id))).catch(() => undefined).finally(() => pipelinePrecomputePausePendingRef.current.delete(job.job_id));
    }
  }, [backend.mode, backendJobs, controlPipelineJob, currentDatasetIndexRevision, currentPipelineRegistryHash, datasetWorkspaceReady, pipelineEnabled, pipelineHasTile, pipelineRequestNodes, pipelineScope]);

  useEffect(() => {
    if (pipelineScope !== 'all' || automaticallyPausedPipelineJobs.size === 0) return;
    setAutomaticallyPausedPipelineJobs((old) => new Set([...old].filter((jobId) => {
      const state = backendJobs.find((job) => job.job_id === jobId)?.state;
      return state === 'paused' || state === 'interrupted';
    })));
  }, [automaticallyPausedPipelineJobs.size, backendJobs, pipelineScope]);

  useEffect(() => {
    if (!currentPipelinePrecomputeJob || !currentFile?.id) return;
    if (currentPipelinePrecomputeState && ['queued', 'running'].includes(currentPipelinePrecomputeState)) {
      void prioritizeJobItems(currentPipelinePrecomputeJob.jobId, preferredPipelineAssetIds).catch(() => undefined);
    }
    void lookupJobItems(currentPipelinePrecomputeJob.jobId, preferredPipelineAssetIds).catch(() => undefined);
  }, [currentFile?.id, currentPipelinePrecomputeJob, currentPipelinePrecomputeState, lookupJobItems, preferredPipelineAssetIds, prioritizeJobItems]);

  useEffect(() => {
    if (!currentPipelinePrecomputeJob || !dataset.id) return;
    for (const assetId of preferredPipelineAssetIds) {
      const item = backend.jobItemSnapshots[`${currentPipelinePrecomputeJob.jobId}:${assetId}`];
      if (item?.state !== 'succeeded') continue;
      const result = pipelinePreviewResultFromJobItem(item.result);
      if (!result || result.dataset_id !== dataset.id || result.asset_id !== assetId) continue;
      cachePipelinePreview(result, currentPipelineExecutionSignature);
      mergePipelineTimings(result);
      if (pipelineScope === 'all'
        && pipelineEnabled
        && assetId === currentFile?.id
        && !(completedPipelineSignature === currentPipelineExecutionSignature && backend.pipeline.data?.dataset_id === dataset.id && backend.pipeline.data.asset_id === assetId)) {
        restorePipelinePreview(result);
        setCompletedPipelineSignature(currentPipelineExecutionSignature);
      }
    }
  }, [backend.jobItemSnapshots, backend.pipeline.data?.asset_id, backend.pipeline.data?.dataset_id, cachePipelinePreview, completedPipelineSignature, currentFile?.id, currentPipelineExecutionSignature, currentPipelinePrecomputeJob, dataset.id, mergePipelineTimings, pipelineEnabled, pipelineScope, preferredPipelineAssetIds, restorePipelinePreview]);

  const featureOutputShape = selectedLayer ? '运行后由真实 Tensor 确认' : '请先选择层';

  const featureTransformParameters = {
    projection: ({ 'PCA-1': 'pca1', Mean: 'mean', Max: 'max', 'Single Channel': 'channel', 'Token Grid': 'token_grid' } as Record<string, string>)[projection] ?? 'none',
    channel: featureChannel,
    normalization: ({ 'Min-Max': 'minmax', 'Z-Score': 'zscore', L2: 'l2', None: 'none' } as Record<string, string>)[normalization] ?? 'none',
    interpolation: 'bilinear',
    spatial_scale: 1,
    gain: 1,
    gamma: 1,
    clip_percentiles: featureClip === 'p1p99' ? [1, 99] : featureClip === 'p5p95' ? [5, 95] : null,
  };
  useEffect(() => {
    if (!pipelineHasModelFeature || !selectedModel.id) return;
    setNodes((current) => {
      let changed = false;
      const next = current.map((node) => {
        if (node.kind !== 'model_feature') return node;
        const nextParameters = {
          ...node.parameters,
          model_id: selectedModel.id,
          layer_id: selectedLayer?.id,
          projection: featureProjectionValue[projection] ?? 'none',
          normalization: featureTransformParameters.normalization,
          channel: featureChannel,
          clip: featureClip,
        };
        for (const displayOnlyParameter of ['interpolation', 'spatial_scale', 'gain', 'gamma']) delete nextParameters[displayOnlyParameter];
        if (JSON.stringify(nextParameters) === JSON.stringify(node.parameters)) return node;
        changed = true;
        return { ...node, parameters: nextParameters };
      });
      return changed ? next : current;
    });
  }, [featureChannel, featureClip, featureTransformParameters.normalization, pipelineHasModelFeature, projection, selectedLayer?.id, selectedModel.id]);
  const singleInferenceParameters = {
    ...inferenceParameters,
    ...(captureFeatures && selectedLayer ? { feature_transform: featureTransformParameters } : {}),
    ...(isSamModel ? samPromptParameters : {}),
  };
  const currentInferenceRequestSignature = currentFile?.imagePath && selectedModel.id
    ? inferenceRequestSignature({
      model_id: selectedModel.id,
      image_path: currentFile.imagePath,
      provider: inferenceProvider,
      capture_layer: captureFeatures ? selectedLayer?.id ?? null : null,
      parameters: singleInferenceParameters,
    })
    : '';
  currentInferenceSignatureRef.current = currentInferenceRequestSignature;

  const updateInferenceParameter = (name: string, value: unknown) => {
    if (!Object.hasOwn(selectedModel.parametersSchema, name)) return;
    setInferenceParameters((current) => {
      const next = { ...current };
      if (value === undefined) delete next[name];
      else next[name] = value;
      return next;
    });
  };

  const resetInferenceParameters = () => {
    setInferenceParameters(inferenceParameterDefaults(selectedModel.parametersSchema));
  };

  const applyPipelineWorkspaceSettings = useCallback((candidate: WorkspacePipelineSettings | null | undefined) => {
    const settings = usablePipelineSettings(candidate);
    if (!settings) {
      setNodes(initialNodes);
      setVisualizations(initialVisualizations);
      setSelectedNode('source');
      setPipelineEnabled(true);
      setPipelineScope('current');
      setVisualizationDisplayMode('split');
      setSinglePipelineSource('source');
      setVisualizationLayerState({});
      nextNodeId.current = 2;
      nextVisualizationId.current = 2;
      return;
    }
    const restoredNodes: FlowNode[] = settings.nodes.map((node) => ({
      ...node,
      operator_version: node.operator_version ?? 'pending',
    }));
    const restoredVisualizations: VisualizationNode[] = settings.visualizations.map((node) => ({
      ...node,
      operator_version: node.operator_version ?? 'pending',
    }));
    setNodes(restoredNodes);
    setVisualizations(restoredVisualizations);
    setSelectedNode(restoredNodes[0]?.id ?? 'source');
    setPipelineEnabled(settings.enabled);
    setPipelineScope(settings.scope);
    setVisualizationDisplayMode(settings.display_mode);
    setSinglePipelineSource(settings.single_source);
    setVisualizationLayerState(settings.layer_state);
    nextNodeId.current = nextGeneratedSequence(restoredNodes.map((node) => node.id));
    nextVisualizationId.current = nextGeneratedSequence(restoredVisualizations.map((node) => node.id));
  }, []);

  const notify = (message: string) => {
    setToast(message);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(''), 2200);
  };

  const markProjectRecent = (datasetId: string) => {
    setRecentProjectIds((current) => {
      const next = [datasetId, ...current.filter((id) => id !== datasetId)].slice(0, 8);
      window.localStorage.setItem('labelone-recent-projects-v1', JSON.stringify(next));
      return next;
    });
  };

  const toggleFullscreen = async () => {
    try {
      if (isFullscreen) {
        await document.exitFullscreen();
        return;
      }
      if (!appShellRef.current?.requestFullscreen) {
        notify('当前浏览器不支持全屏');
        return;
      }
      await appShellRef.current.requestFullscreen();
    } catch (error) {
      notify(error instanceof Error ? `无法切换全屏：${error.message}` : '无法切换全屏');
    }
  };

  const openGlobalSettings = () => {
    setTaskStreamOpen(false);
    setTaskStreamHovered(false);
    setTaskStreamFocused(false);
    setSettingsSection('models');
    setShortcutFeedback('');
    setModelSettingsStatus('');
    setRecordingShortcut(null);
    setSettingsOpen(true);
    if (backend.mode === 'online') void backend.refreshApplicationSettings().catch(() => undefined);
  };

  const closeGlobalSettings = () => {
    setRecordingShortcut(null);
    setSettingsOpen(false);
    window.requestAnimationFrame(() => settingsButtonRef.current?.focus());
  };

  const copyTextToClipboard = async (value: string, successMessage: string) => {
    if (!value) return;
    try {
      let copied = false;
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(value);
          copied = true;
        } catch { /* fall back to a temporary selection below */ }
      }
      if (!copied) {
        const fallback = document.createElement('textarea');
        fallback.value = value;
        fallback.setAttribute('readonly', '');
        fallback.style.position = 'fixed';
        fallback.style.opacity = '0';
        document.body.append(fallback);
        fallback.select();
        try { copied = document.execCommand('copy'); }
        finally { fallback.remove(); }
      }
      if (!copied) throw new Error('Clipboard copy was rejected');
      notify(successMessage);
    } catch {
      notify('复制失败，请检查浏览器剪贴板权限');
    }
  };

  function scheduleAnnotationPersistence(message = '等待 700ms 保护草稿') {
    if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    setAnnotationPersistence({ phase: 'pending', message });
    annotationAutoSaveTimerRef.current = window.setTimeout(() => {
      annotationAutoSaveTimerRef.current = null;
      if (annotationEditActiveRef.current) {
        scheduleAnnotationPersistence('编辑事务进行中，完成后保护草稿');
        return;
      }
      void persistCurrentAnnotation(false);
    }, 700);
  }

  const resetAnnotationHistory = (document: AnnotationEnvelope['document']) => {
    annotationBaseDocumentRef.current = structuredClone(document);
    annotationHistoryRef.current = createAnnotationHistory(document);
    setAnnotationHistoryVersion((value) => value + 1);
  };

  const documentIsDirty = (document: AnnotationEnvelope['document']) => {
    const base = annotationBaseDocumentRef.current;
    return !base || annotationFingerprint(base) !== annotationFingerprint(document);
  };

  const replaceDraftDocument = (document: AnnotationEnvelope['document']) => {
    const envelope = annotationDraftRef.current;
    if (!envelope) return;
    const next = { ...envelope, document: structuredClone(document) };
    annotationDraftRef.current = next;
    setAnnotationDraft(next);
    setAnnotationDirty(documentIsDirty(document));
  };

  const commitAnnotationDocument = (document: AnnotationEnvelope['document']) => {
    const envelope = annotationDraftRef.current;
    if (!envelope) return false;
    const currentHistory = annotationHistoryRef.current ?? createAnnotationHistory(envelope.document);
    const nextHistory = commitAnnotationHistory(currentHistory, document);
    if (nextHistory === currentHistory) return false;
    annotationHistoryRef.current = nextHistory;
    setAnnotationHistoryVersion((value) => value + 1);
    replaceDraftDocument(nextHistory.present);
    scheduleAnnotationPersistence();
    return true;
  };

  const commitTransientAnnotationEdit = () => {
    const document = annotationDraftRef.current?.document;
    if (document) commitAnnotationDocument(document);
    annotationEditActiveRef.current = false;
    setAnnotationEditActive(false);
  };

  const undoAnnotation = () => {
    cancelAnnotationDrafting();
    const history = annotationHistoryRef.current;
    if (!history?.past.length) return;
    const next = undoAnnotationHistory(history);
    annotationHistoryRef.current = next;
    setAnnotationHistoryVersion((value) => value + 1);
    setSelectedShapeIndex(null);
    setHiddenShapeIndexes(new Set());
    replaceDraftDocument(next.present);
    scheduleAnnotationPersistence('撤销结果等待保护草稿');
  };

  const redoAnnotation = () => {
    cancelAnnotationDrafting();
    const history = annotationHistoryRef.current;
    if (!history?.future.length) return;
    const next = redoAnnotationHistory(history);
    annotationHistoryRef.current = next;
    setAnnotationHistoryVersion((value) => value + 1);
    setSelectedShapeIndex(null);
    setHiddenShapeIndexes(new Set());
    replaceDraftDocument(next.present);
    scheduleAnnotationPersistence('重做结果等待保护草稿');
  };

  const resetSamPrompts = () => {
    samBoxDragRef.current = null;
    setSamPoints([]);
    setSamBoxes([]);
    setSamBoxPreview(null);
    setSamPromptMode(false);
  };

  const cancelAnnotationDrafting = () => {
    selectOnNextCanvasBlankRef.current = false;
    const activeTransaction = Boolean(shapeDragRef.current || controlPointDragRef.current || realRotationDragRef.current);
    drawRef.current = null;
    brushRef.current = null;
    shapeDragRef.current = null;
    controlPointDragRef.current = null;
    realRotationDragRef.current = null;
    setDrawPreview(null);
    polygonDraftRef.current = [];
    setPolygonDraft([]);
    setPolygonPointer(null);
    setPolygonCloseReady(false);
    setBrushPreview([]);
    pendingManualShapeRef.current = null;
    setPendingManualShape(null);
    setManualShapeLabel('');
    annotationEditActiveRef.current = false;
    setAnnotationEditActive(false);
    if (activeTransaction && annotationHistoryRef.current) replaceDraftDocument(annotationHistoryRef.current.present);
  };

  const activateTool = (nextTool: AnnotationTool) => {
    cancelAnnotationDrafting();
    setSamPromptMode(false);
    setTool(nextTool);
  };

  const changePipelineEnabled = (enabled: boolean) => {
    setPipelineEnabled(enabled);
  };

  const changePipelineScope = (scope: PipelineScope) => {
    setPipelineScope(scope);
  };

  const loadAnnotationFor = async (file: FakeFile, datasetId = dataset.id) => {
    const requestId = ++annotationLoadRequestId.current;
    if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    annotationAutoSaveTimerRef.current = null;
    annotationAutoSaveRef.current = DEFAULT_ANNOTATION_AUTO_SAVE;
    setAnnotationAutoSave(DEFAULT_ANNOTATION_AUTO_SAVE);
    cancelAnnotationDrafting();
    setPendingShapeLabelEdit(null);
    setPendingCategoryLabelEdit(null);
    setSelectedShapeIndex(null);
    setHiddenShapeIndexes(new Set());
    setAnnotationDirty(false);
    setAnnotationRecovery(null);
    setAnnotationPersistence({ phase: 'idle' });
    annotationBaseDocumentRef.current = null;
    annotationHistoryRef.current = null;
    setAnnotationHistoryVersion((value) => value + 1);
    if (backend.mode !== 'online' || !datasetId || file.status !== 'valid' || file.selectable === false) {
      annotationDraftRef.current = null;
      setAnnotationDraft(null);
      return;
    }
    annotationDraftRef.current = null;
    setAnnotationDraft(null);
    try {
      const loaded = await backend.loadAnnotation(datasetId, file.id);
      if (requestId !== annotationLoadRequestId.current) return;
      const serverEnvelope = structuredClone(loaded);
      annotationDraftRef.current = serverEnvelope;
      setAnnotationDraft(serverEnvelope);
      resetAnnotationHistory(serverEnvelope.document);
      setSelectedShapeIndex(null);
      setAnnotationDirty(false);
      try {
        const local = await getAnnotationDraft(datasetId, file.id);
        if (requestId !== annotationLoadRequestId.current || !local) return;
        setAnnotationRecovery({ kind: local.base_revision === loaded.revision ? 'recoverable' : 'conflict', local, server: serverEnvelope });
        setAnnotationPersistence(local.base_revision === loaded.revision
          ? { phase: 'local', message: '发现意外退出前的本地草稿' }
          : { phase: 'error', message: '本地草稿与服务端 revision 冲突' });
      } catch (error) {
        if (requestId === annotationLoadRequestId.current) setAnnotationPersistence({ phase: 'error', message: error instanceof Error ? error.message : '本地草稿读取失败' });
      }
    } catch (error) {
      if (requestId !== annotationLoadRequestId.current) return;
      annotationDraftRef.current = null;
      setAnnotationDraft(null);
      notify(error instanceof Error ? error.message : '标注加载失败');
    }
  };
  annotationLoaderRef.current = loadAnnotationFor;

  const performFileSelection = (file: FakeFile, datasetId = dataset.id) => {
    if (file.status !== 'valid' || (datasetId && file.selectable !== true)) return;
    setRemoteInferenceConfirmation(null);
    lastAutoInferenceSignatureRef.current = '';
    setCompletedInferenceSignature('');
    resetSamPrompts();
    if (datasetId) setOpenedDataset((old) => old && old.id === datasetId && !old.files.some((item) => item.id === file.id) ? { ...old, files: [...old.files, file] } : old);
    setCurrentFileId(file.id);
    void loadAnnotationFor(file, datasetId);
  };

  const selectFile = (file: FakeFile, datasetId = dataset.id) => {
    if (file.status !== 'valid' || (datasetId && file.selectable !== true)) return;
    if (file.id === currentFile?.id && datasetId === dataset.id) return;
    cancelAnnotationDrafting();
    const source = annotationDraftRef.current;
    const hasUnsavedChanges = Boolean(source && documentIsDirty(source.document));
    annotationDirtyRef.current = hasUnsavedChanges;
    setAnnotationDirty(hasUnsavedChanges);
    if (!hasUnsavedChanges || !source) {
      performFileSelection(file, datasetId);
      return;
    }
    if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    annotationAutoSaveTimerRef.current = null;
    setAnnotationNavigationError('');
    setAnnotationNavigationDecision(null);
    setPendingAnnotationNavigation({
      file,
      datasetId,
      targetLabel: file.name,
      sourceDatasetId: source.dataset_id,
      sourceAssetId: source.asset_id,
      sourceFingerprint: annotationFingerprint(source.document),
    });
  };

  const stepFile = (direction: number) => {
    const index = Math.max(0, validFiles.findIndex((file) => file.id === currentFile?.id));
    const next = validFiles[index + direction];
    if (next) {
      selectFile(next);
      return;
    }
    if (direction > 0 && backend.mode === 'online' && dataset.id && backend.assetSearch.data.next_cursor != null && !backend.assetSearchLoadingMore) {
      void searchDatasetAssets(
        dataset.id,
        { ...realSearchInput, cursor: backend.assetSearch.data.next_cursor },
        true,
      ).then((response) => {
        const first = response?.items.map(fileFromAsset).find((file) => file.selectable === true && file.rawStatus === 'valid');
        if (first) selectFile(first);
      }).catch(() => undefined);
    }
  };

  useEffect(() => {
    const saved = window.localStorage.getItem('labelone-prototype-session');
    const restoreTimer = window.setTimeout(() => {
      try {
        const state = saved ? JSON.parse(saved) : {};
        if (['layers', 'pipeline', 'inference', 'agent'].includes(state.rightTab)) setRightTab(state.rightTab as RightTab);
        else if (state.rightTab === 'tasks') setRightTab('layers');
        if (state.view) setView(state.view);
        setAnnotationCategoryColorOverrides(sanitizeAnnotationCategoryColorOverrides(state.annotationCategoryColorOverrides));
      } catch { /* ignore invalid prototype session */ }
      finally { sessionHydratedRef.current = true; }
    }, 0);
    return () => window.clearTimeout(restoreTimer);
  }, []);

  useEffect(() => {
    try {
      const value = JSON.parse(window.localStorage.getItem('labelone-recent-projects-v1') ?? '[]');
      if (Array.isArray(value)) setRecentProjectIds(value.filter((item): item is string => typeof item === 'string').slice(0, 8));
    } catch { /* ignore invalid recent-project state */ }
  }, []);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem('labelone-global-settings-v1');
      const state = saved ? JSON.parse(saved) : {};
      setShortcutOverrides(sanitizeShortcutOverrides(state.shortcutOverrides));
      if (state.defaultInferenceProvider === 'CPUExecutionProvider') {
        setInferenceProvider(state.defaultInferenceProvider);
      }
      setUseMacShortcutSymbols(/Mac|iPhone|iPad/.test(navigator.platform));
    } catch { /* ignore invalid global settings */ }
    finally { globalSettingsHydratedRef.current = true; }
  }, []);

  useEffect(() => {
    if (!globalSettingsHydratedRef.current) return;
    window.localStorage.setItem('labelone-global-settings-v1', JSON.stringify({
      schemaVersion: 1,
      shortcutOverrides,
      defaultInferenceProvider: inferenceProvider,
    }));
  }, [inferenceProvider, shortcutOverrides]);

  useEffect(() => {
    const remote = backend.applicationSettings.data;
    if (!remote) return;
    setModelWeightsPathInput(remote.model_weights_dir);
    setModelDownloadSource(remote.model_download_source || 'auto');
    setNetworkProxyDraft(remote.network_proxy ?? { mode: 'system', url: '', bypass: 'localhost,127.0.0.1,::1' });
    if (remote.cloud_ai) setCloudAiDraft({
      enabled: remote.cloud_ai.enabled,
      provider: remote.cloud_ai.provider,
      endpoint: remote.cloud_ai.endpoint,
      model: remote.cloud_ai.model,
      api_key_env: remote.cloud_ai.api_key_env,
      timeout_seconds: remote.cloud_ai.timeout_seconds,
      max_output_tokens: remote.cloud_ai.max_output_tokens,
    });
    if (!remoteWorkspaceHydratedRef.current) {
      remoteWorkspaceHydratedRef.current = true;
      const workspace = remote.workspace;
      if (!openedDataset) applyPipelineWorkspaceSettings(workspace?.pipeline);
      setInferenceProvider(workspace?.inference.provider || 'CPUExecutionProvider');
      let localModelId = '';
      try { localModelId = window.localStorage.getItem(LAST_SELECTED_MODEL_KEY) ?? ''; } catch { /* keep remote/default selection */ }
      setSelectedModelId(workspace?.inference.model_id ?? localModelId);
      setInferenceParameters(workspace?.inference.parameters ?? {});
      globalInferenceFingerprintRef.current = JSON.stringify({
        model_id: workspace?.inference.model_id ?? null,
        provider: workspace?.inference.provider || 'CPUExecutionProvider',
        parameters: workspace?.inference.parameters ?? {},
      });
      setGlobalWorkspaceHydrated(true);
    }
  }, [applyPipelineWorkspaceSettings, backend.applicationSettings.data, openedDataset]);

  useEffect(() => {
    const remote = backend.applicationSettings.data;
    if (!globalWorkspaceHydrated || backend.mode !== 'online' || !remote) return;
    const inference = {
      model_id: selectedModelId || null,
      provider: inferenceProvider,
      parameters: inferenceParameters,
    };
    const fingerprint = JSON.stringify(inference);
    if (fingerprint === globalInferenceFingerprintRef.current) return;
    if (globalInferenceSaveTimerRef.current) window.clearTimeout(globalInferenceSaveTimerRef.current);
    globalInferenceSaveTimerRef.current = window.setTimeout(() => {
      void updateApplicationSettings({
        workspace: globalWorkspaceSettings(remote.workspace?.pipeline ?? persistedPipelineSettings, inference),
      }).then(() => {
        globalInferenceFingerprintRef.current = fingerprint;
      }).catch(() => undefined);
    }, 500);
    return () => {
      if (globalInferenceSaveTimerRef.current) window.clearTimeout(globalInferenceSaveTimerRef.current);
    };
  }, [backend.applicationSettings.data, backend.mode, globalWorkspaceHydrated, inferenceParameters, inferenceProvider, persistedPipelineSettings, selectedModelId, updateApplicationSettings]);

  useEffect(() => {
    if (!dataset.id || datasetWorkspaceHydratedId !== dataset.id || !currentFileId || backend.mode !== 'online') return;
    const settings = {
      schema_version: 1 as const,
      last_asset_id: currentFileId,
      pipeline: persistedPipelineSettings,
    };
    const fingerprint = JSON.stringify(settings);
    if (datasetWorkspaceFingerprintRef.current === null) {
      datasetWorkspaceFingerprintRef.current = fingerprint;
      return;
    }
    if (fingerprint === datasetWorkspaceFingerprintRef.current) return;
    if (datasetWorkspaceSaveTimerRef.current) window.clearTimeout(datasetWorkspaceSaveTimerRef.current);
    const datasetId = dataset.id;
    const operationId = datasetOperationRef.current;
    datasetWorkspaceSaveTimerRef.current = window.setTimeout(() => {
      setWorkspaceSettingsSaving(true);
      datasetWorkspaceSaveChainRef.current = datasetWorkspaceSaveChainRef.current.catch(() => undefined).then(() => saveDatasetSettings(datasetId, {
        ...settings,
        expected_revision: datasetWorkspaceRevisionRef.current,
      })).then((saved) => {
        if (datasetOperationRef.current !== operationId) return;
        datasetWorkspaceRevisionRef.current = saved.revision;
        datasetWorkspaceFingerprintRef.current = fingerprint;
        setWorkspaceSettingsStatus('数据集配置已保存');
      }).catch((error) => {
        if (datasetOperationRef.current === operationId) setWorkspaceSettingsStatus(error instanceof Error ? error.message : '数据集配置保存失败');
      }).finally(() => {
        if (datasetOperationRef.current === operationId) setWorkspaceSettingsSaving(false);
      });
    }, 500);
    return () => {
      if (datasetWorkspaceSaveTimerRef.current) window.clearTimeout(datasetWorkspaceSaveTimerRef.current);
    };
  }, [backend.mode, currentFileId, dataset.id, datasetWorkspaceHydratedId, persistedPipelineSettings, saveDatasetSettings]);

  useEffect(() => {
    const syncFullscreenState = () => setIsFullscreen(Boolean(document.fullscreenElement));
    syncFullscreenState();
    document.addEventListener('fullscreenchange', syncFullscreenState);
    return () => document.removeEventListener('fullscreenchange', syncFullscreenState);
  }, []);

  useEffect(() => {
    if (!settingsOpen) return;
    const focusTimer = window.setTimeout(() => settingsCloseRef.current?.focus(), 0);
    return () => window.clearTimeout(focusTimer);
  }, [settingsOpen]);

  useEffect(() => {
    if (!settingsOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (recordingShortcut) {
        setRecordingShortcut(null);
        setShortcutFeedback('已取消快捷键录制');
        return;
      }
      setSettingsOpen(false);
      settingsButtonRef.current?.focus();
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [recordingShortcut, settingsOpen]);

  useEffect(() => {
    if (!sessionHydratedRef.current) return;
    window.localStorage.setItem('labelone-prototype-session', JSON.stringify({ rightTab, view, annotationCategoryColorOverrides }));
  }, [rightTab, view, annotationCategoryColorOverrides]);

  useEffect(() => {
    const persistBeforeExit = () => {
      const current = annotationDraftRef.current;
      const base = annotationBaseDocumentRef.current;
      if (!current || !base || annotationFingerprint(current.document) === annotationFingerprint(base)) return;
      void putAnnotationDraft({
        dataset_id: current.dataset_id,
        asset_id: current.asset_id,
        base_revision: current.revision,
        document: current.document,
      }).catch(() => undefined);
    };
    const warnBeforeExit = (event: BeforeUnloadEvent) => {
      const current = annotationDraftRef.current;
      const base = annotationBaseDocumentRef.current;
      if (!current || !base || annotationFingerprint(current.document) === annotationFingerprint(base)) return;
      event.preventDefault();
    };
    window.addEventListener('pagehide', persistBeforeExit);
    window.addEventListener('beforeunload', warnBeforeExit);
    return () => {
      window.removeEventListener('pagehide', persistBeforeExit);
      window.removeEventListener('beforeunload', warnBeforeExit);
      if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    };
  }, []);

  const maximumCanvasScale = useCallback(() => {
    const pipelineIsDisplayed = Boolean(
      pipelineEnabled
      && backend.pipeline.data?.dataset_id === dataset.id
      && backend.pipeline.data?.asset_id === currentFile?.id,
    );
    const pipelineItems = pipelineIsDisplayed && backend.pipeline.data
      ? backend.pipeline.data.visualizations?.length ? backend.pipeline.data.visualizations : [backend.pipeline.data]
      : [];
    const dimensions = pipelineItems.length
      ? pipelineItems.map((item) => ({ width: item.width, height: item.height }))
      : [{ width: currentFile?.width ?? 0, height: currentFile?.height ?? 0 }];
    const stageWidth = stageRef.current?.clientWidth ?? 840;
    const stageHeight = stageRef.current?.clientHeight ?? 592;
    const split = pipelineItems.length > 1;
    const fallbackPaneWidth = split ? stageWidth / 2 : stageWidth;
    const fallbackPaneHeight = pipelineItems.length > 2 ? stageHeight / 2 : stageHeight;
    const paneWidth = imageRef.current?.clientWidth ?? fallbackPaneWidth;
    const paneHeight = imageRef.current?.clientHeight ?? fallbackPaneHeight;
    return Math.max(1, ...dimensions.flatMap((item) => {
      const contained = containedPipelineImageRect({ left: 0, top: 0, width: paneWidth, height: paneHeight }, item.width, item.height);
      return [
        maximumCanvasScaleForPixelInspection(item.width, contained.width, MAX_SOURCE_PIXEL_INSPECTION_SIZE),
        maximumCanvasScaleForPixelInspection(item.height, contained.height, MAX_SOURCE_PIXEL_INSPECTION_SIZE),
      ];
    }));
  }, [backend.pipeline.data, currentFile?.height, currentFile?.id, currentFile?.width, dataset.id, pipelineEnabled]);

  const beginZoomEditing = () => {
    setZoomDraft(String(Math.round(view.scale * 100)));
    setZoomEditing(true);
    window.requestAnimationFrame(() => {
      zoomValueInputRef.current?.focus();
      zoomValueInputRef.current?.select();
    });
  };

  const restoreZoomValueFocus = () => {
    window.requestAnimationFrame(() => zoomValueInputRef.current?.focus());
  };

  const cancelZoomEditing = (restoreFocus = false) => {
    setZoomDraft('');
    setZoomEditing(false);
    if (restoreFocus) restoreZoomValueFocus();
  };

  const resetZoomPercent = () => {
    setZoomDraft('');
    setZoomEditing(false);
    setView((old) => zoomCanvasView(old, 1 / old.scale, maximumCanvasScale()));
  };

  const commitZoomEditing = (restoreFocus = false) => {
    const normalizedDraft = zoomDraft.trim().replace(/%$/, '').trim();
    const requestedPercent = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(normalizedDraft) ? Number(normalizedDraft) : Number.NaN;
    if (!Number.isFinite(requestedPercent)) {
      cancelZoomEditing(restoreFocus);
      notify('请输入有效的缩放百分比');
      return;
    }
    const maximumScale = maximumCanvasScale();
    const maximumPercent = Math.round(maximumScale * 100);
    const targetScale = canvasScaleFromPercent(requestedPercent, maximumScale, view.scale);
    setView((old) => zoomCanvasView(old, targetScale / old.scale, maximumScale));
    setZoomEditing(false);
    setZoomDraft(String(Math.round(targetScale * 100)));
    if (requestedPercent < 25 || requestedPercent > maximumPercent) notify(`缩放已限制在 25%–${maximumPercent}%`);
    if (restoreFocus) restoreZoomValueFocus();
  };

  const zoomAtCenter = (factor: number) => {
    setView((old) => zoomCanvasView(old, factor, maximumCanvasScale()));
  };

  const zoomOneToOne = () => {
    const sourceWidth = currentFile?.width;
    const surfaceWidth = imageRef.current?.clientWidth;
    if (!sourceWidth || !surfaceWidth) {
      setView((old) => zoomCanvasView(old, 1 / old.scale, maximumCanvasScale()));
      return;
    }
    const targetScale = sourceWidth / surfaceWidth;
    setView((old) => zoomCanvasView(old, targetScale / old.scale, maximumCanvasScale()));
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (settingsOpen) return;
      const canvasHasKeyboardFocus = event.target === stageRef.current;
      if (isBrowserZoomKeyboardShortcut(event)) {
        event.preventDefault();
        if (!canvasHasKeyboardFocus) return;
      }
      if (pendingManualShape) {
        if (event.key === 'Escape') {
          event.preventDefault();
          pendingManualShapeRef.current = null;
          setPendingManualShape(null);
          setManualShapeLabel('');
          window.requestAnimationFrame(() => stageRef.current?.focus({ preventScroll: true }));
        }
        return;
      }
      if (pendingShapeLabelEdit || pendingCategoryLabelEdit) {
        if (event.key === 'Escape' && !event.isComposing) {
          event.preventDefault();
          closePendingLabelEditor();
        }
        return;
      }
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement || (event.target instanceof HTMLElement && event.target.isContentEditable)) return;
      if (pendingAnnotationNavigation) {
        event.preventDefault();
        if (event.key === 'Escape' && !annotationNavigationDecision) {
          setPendingAnnotationNavigation(null);
          setAnnotationNavigationError('');
          const current = annotationDraftRef.current;
          if (current && documentIsDirty(current.document)) scheduleAnnotationPersistence();
        }
        return;
      }
      const shortcutAction = resolveShortcutAction(event, shortcuts, canvasHasKeyboardFocus ? 'canvas' : 'app');
      if (shortcutAction === 'edit.undo') { event.preventDefault(); undoAnnotation(); return; }
      if (shortcutAction === 'edit.redo') { event.preventDefault(); redoAnnotation(); return; }
      if (shortcutAction === 'edit.save') { event.preventDefault(); void saveCurrentAnnotation(); return; }
      if (shortcutAction === 'edit.changeCategory') {
        if (selectedShapeIndex !== null && annotationDraft) {
          event.preventDefault();
          openShapeLabelEditor(selectedShapeIndex);
        }
        return;
      }
      if (!canvasHasKeyboardFocus) return;
      if (event.code === 'Space') { event.preventDefault(); setSpaceDown(true); }
      if (event.key === 'Escape') { cancelAnnotationDrafting(); setSamPromptMode(false); return; }
      if (event.key === 'Enter' && tool === 'polygon') { event.preventDefault(); finishPolygonDraft(); return; }
      if (event.key === 'Backspace' && tool === 'polygon' && polygonDraft.length) {
        event.preventDefault();
        const next = polygonDraftRef.current.slice(0, -1);
        polygonDraftRef.current = next;
        setPolygonDraft(next);
        setPolygonPointer(next.at(-1) ?? null);
        setPolygonCloseReady(false);
        return;
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedShapeIndex !== null) { event.preventDefault(); deleteSelectedShape(); return; }
      if (shortcutAction === 'navigation.previous') { event.preventDefault(); stepFile(-1); return; }
      if (shortcutAction === 'navigation.next') { event.preventDefault(); stepFile(1); return; }
      if (shortcutAction === 'canvas.zoomIn') { event.preventDefault(); zoomAtCenter(CANVAS_ZOOM_STEP); return; }
      if (shortcutAction === 'canvas.zoomOut') { event.preventDefault(); zoomAtCenter(1 / CANVAS_ZOOM_STEP); return; }
      if (shortcutAction === 'canvas.fit') { event.preventDefault(); setView({ scale: .92, x: 0, y: 0 }); return; }
      if (shortcutAction === 'canvas.actualSize') { event.preventDefault(); zoomOneToOne(); return; }
      const shortcutTool = shortcutAction ? shortcutToolActions[shortcutAction] : null;
      if (shortcutTool) { event.preventDefault(); activateTool(shortcutTool); }
    };
    const onKeyUp = (event: KeyboardEvent) => { if (event.code === 'Space') setSpaceDown(false); };
    const onWindowBlur = () => {
      dragRef.current = null;
      setSpaceDown(false);
      setCanvasPanning(false);
      canvasCrosshairRef.current?.removeAttribute('data-visible');
      pixelSampleGenerationRef.current += 1;
      pendingPixelSampleRef.current = null;
      if (pixelSampleFrameRef.current !== null) window.cancelAnimationFrame(pixelSampleFrameRef.current);
      pixelSampleFrameRef.current = null;
      setCursor((old) => old.insideImage || old.pixel ? { ...old, insideImage: false, pixel: null } : old);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWindowBlur);
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp); window.removeEventListener('blur', onWindowBlur); };
  });

  useEffect(() => {
    const preventBrowserZoomKey = (event: KeyboardEvent) => {
      if (isBrowserZoomKeyboardShortcut(event)) event.preventDefault();
    };
    window.addEventListener('keydown', preventBrowserZoomKey, { capture: true });
    return () => window.removeEventListener('keydown', preventBrowserZoomKey, { capture: true });
  }, []);

  useEffect(() => {
    if (!pendingManualShape) return;
    const dismissOnOutsidePointer = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && manualLabelMenuRef.current?.contains(target)) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      pendingManualShapeRef.current = null;
      setPendingManualShape(null);
      setManualShapeLabel('');
      window.requestAnimationFrame(() => stageRef.current?.focus({ preventScroll: true }));
    };
    window.addEventListener('pointerdown', dismissOnOutsidePointer, true);
    return () => window.removeEventListener('pointerdown', dismissOnOutsidePointer, true);
  }, [pendingManualShape]);

  useEffect(() => {
    if (!pendingManualShape) return;
    const measureMenu = () => {
      const bounds = manualLabelMenuRef.current?.getBoundingClientRect();
      const next = {
        width: bounds?.width ?? 300,
        height: bounds?.height ?? 320,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      };
      setManualLabelMenuGeometry((current) => current.width === next.width && current.height === next.height && current.viewportWidth === next.viewportWidth && current.viewportHeight === next.viewportHeight ? current : next);
    };
    measureMenu();
    window.addEventListener('resize', measureMenu);
    window.visualViewport?.addEventListener('resize', measureMenu);
    return () => {
      window.removeEventListener('resize', measureMenu);
      window.visualViewport?.removeEventListener('resize', measureMenu);
    };
  }, [pendingManualShape]);

  const restoreRegisteredWorkspace = async (
    registered: RegisteredDataset,
    assets: AssetCursorPage,
    workspace: DatasetWorkspaceSettingsResponse,
    operationId: number,
    defaultPipeline?: WorkspacePipelineSettings | null,
  ) => {
    if (operationId !== datasetOperationRef.current) return null;
    let opened = datasetFromRegistered(registered, assets);
    let chosen = workspace.last_asset_id
      ? opened.files.find((file) => file.id === workspace.last_asset_id && file.status === 'valid' && file.selectable === true)
      : undefined;
    if (!chosen && workspace.last_asset_id) {
      try {
        const restoredFile = fileFromAsset(await getDatasetAsset(registered.dataset_id, workspace.last_asset_id));
        if (operationId !== datasetOperationRef.current) return null;
        if (restoredFile.status === 'valid' && restoredFile.selectable === true) {
          chosen = restoredFile;
          opened = { ...opened, files: [...opened.files, restoredFile] };
        }
      } catch { /* a rescan may have removed the saved asset */ }
    }
    chosen ??= opened.files.find((file) => file.status === 'valid' && file.selectable === true);
    if (!chosen) throw new Error('这个项目没有可打开的有效图像，请重新选择项目文件夹');
    const effectivePipeline = usablePipelineSettings(workspace.pipeline)
      ?? usablePipelineSettings(defaultPipeline)
      ?? persistedPipelineSettings;
    datasetWorkspaceFingerprintRef.current = null;
    datasetWorkspaceRevisionRef.current = workspace.revision;
    setDatasetWorkspaceHydratedId(null);
    setWorkspaceSettingsStatus('');
    applyPipelineWorkspaceSettings(effectivePipeline);
    setOpenedDataset(opened);
    setCurrentFileId(chosen.id);
    await loadAnnotationFor(chosen, registered.dataset_id);
    if (operationId !== datasetOperationRef.current) return null;
    const normalizedSettings = { schema_version: 1 as const, last_asset_id: chosen.id, pipeline: effectivePipeline };
    const fingerprint = JSON.stringify(normalizedSettings);
    if (workspace.revision === 0 || workspace.last_asset_id !== chosen.id || !usablePipelineSettings(workspace.pipeline)) {
      const saved = await saveDatasetSettings(registered.dataset_id, {
        ...normalizedSettings,
        expected_revision: workspace.revision,
      });
      datasetWorkspaceRevisionRef.current = saved.revision;
    }
    datasetWorkspaceFingerprintRef.current = fingerprint;
    setDatasetWorkspaceHydratedId(registered.dataset_id);
    return chosen;
  };

  const startDatasetScan = async (selection: { imageDir: string }, operationId: number) => {
    if (annotationDirty) { notify('当前标注有未保存更改；请先保存当前标注'); return; }
    if (backend.mode === 'online') {
      const selectedImage = selection.imageDir.trim();
      if (!selectedImage || operationId !== datasetOperationRef.current) return;
      setRootPath(selectedImage);
      setDatasetOpen(true);
      setScanRegistering(false);
      setAutoScanIntent(null);
      setAutoOpenError('');
      try {
        if (backend.scan.data?.state === 'queued' || backend.scan.data?.state === 'running') {
          await backend.interruptScan().catch(() => undefined);
          if (operationId !== datasetOperationRef.current) return;
        }
        const session = await backend.startScan({
          root_dir: selectedImage,
          layout: 'auto',
          match_strategy: 'relative_stem',
          recursive: true,
          validate_images: true,
          validate_annotations: true,
        });
        if (operationId !== datasetOperationRef.current) return;
        setAutoScanIntent({ sessionId: session.session_id, name: directoryBasename(selectedImage), operationId });
        notify(`扫描会话已创建：${session.session_id.slice(0, 12)}；首批数据就绪后即可开始标注`);
      } catch (error) {
        if (operationId !== datasetOperationRef.current) return;
        const message = error instanceof Error ? error.message : '扫描会话创建失败';
        setAutoOpenError(message);
        notify(message);
      }
      return;
    }
    notify('本地服务未连接，请启动 LabelOne server 后重试');
  };

  const openScannedDataset = async (sessionId: string, registrationName: string, operationId: number, progressive = false) => {
    if (backend.mode === 'online') {
      if (operationId !== datasetOperationRef.current) return;
      if (annotationDirty) { setAutoOpenError('当前标注有未保存更改；请先保存当前标注'); notify('当前标注有未保存更改；请先保存当前标注'); return; }
      setRemoteInferenceConfirmation(null);
      resetSamPrompts();
      setScanRegistering(true);
      setAutoOpenError('');
      try {
        const registered = await backend.registerScan(sessionId, registrationName || directoryBasename(rootPath));
        if (operationId !== datasetOperationRef.current) return;
        const [assets, workspace, applicationSettings] = await Promise.all([
          backend.openRegisteredDataset(registered.dataset_id),
          getDatasetSettings(registered.dataset_id),
          backend.applicationSettings.data ? Promise.resolve(backend.applicationSettings.data) : refreshApplicationSettings(),
        ]);
        if (operationId !== datasetOperationRef.current || annotationDirtyRef.current) return;
        const chosen = await restoreRegisteredWorkspace(registered, assets, workspace, operationId, applicationSettings?.workspace.pipeline);
        if (!chosen) return;
        markProjectRecent(registered.dataset_id);
        setDatasetOpen(false);
        if (!progressive) setAutoScanIntent(null);
        setWelcomeError('');
        void backend.refreshDatasets();
        notify(progressive
          ? `已先打开 ${registered.name} 的首批 ${assets.total} 项；其余数据继续后台加载`
          : `已注册并打开 ${registered.name}：索引 revision ${registered.index_revision}`);
      } catch (error) {
        if (operationId !== datasetOperationRef.current) return;
        if (progressive) progressivelyOpenedScanSessionsRef.current.delete(sessionId);
        const message = error instanceof Error ? error.message : '数据集注册失败';
        setAutoOpenError(message);
        notify(message);
      } finally {
        if (operationId === datasetOperationRef.current) setScanRegistering(false);
      }
      return;
    }
    setDatasetOpen(false);
    notify('扫描尚未成功，未注册数据集');
  };
  finishAutoScanRef.current = openScannedDataset;

  const openRecentProject = async (registered: RegisteredDataset) => {
    if (backend.mode !== 'online' || openingRecentDatasetId || directoryPickerPending) return;
    const operationId = ++datasetOperationRef.current;
    setOpeningRecentDatasetId(registered.dataset_id);
    setWelcomeError('');
    try {
      const [assets, workspace, applicationSettings] = await Promise.all([
        backend.openRegisteredDataset(registered.dataset_id),
        getDatasetSettings(registered.dataset_id),
        backend.applicationSettings.data ? Promise.resolve(backend.applicationSettings.data) : refreshApplicationSettings(),
      ]);
      if (operationId !== datasetOperationRef.current) return;
      setSearch('');
      setFilter('all');
      const chosen = await restoreRegisteredWorkspace(registered, assets, workspace, operationId, applicationSettings?.workspace.pipeline);
      if (!chosen) return;
      markProjectRecent(registered.dataset_id);
      notify(`已打开 ${registered.name}`);
    } catch (error) {
      if (operationId !== datasetOperationRef.current) return;
      setWelcomeError(error instanceof Error ? error.message : '最近项目打开失败');
    } finally {
      if (operationId === datasetOperationRef.current) setOpeningRecentDatasetId(null);
    }
  };

  const pickImageDataset = async () => {
    if (annotationDirty) { notify('当前标注有未保存更改；请先保存当前标注'); return; }
    if (directoryPickerInFlightRef.current || scanRegistering) return;
    if (backend.mode !== 'online') {
      notify('本地服务未连接，无法打开系统文件夹选择器');
      return;
    }
    const observedOperation = datasetOperationRef.current;
    directoryPickerInFlightRef.current = true;
    setDirectoryPickerPending('image');
    setWelcomeError('');
    try {
      const result = await backend.pickDirectory({ title: '选择图像数据集文件夹', initial_dir: rootPath || undefined });
      if (result.canceled) return;
      if (!result.path) throw new Error('系统目录选择器未返回路径');
      if (observedOperation !== datasetOperationRef.current || annotationDirtyRef.current) return;
      const operationId = ++datasetOperationRef.current;
      await startDatasetScan({ imageDir: result.path }, operationId);
    } catch (error) {
      if (observedOperation !== datasetOperationRef.current) return;
      const message = error instanceof Error ? error.message : '系统文件夹选择器不可用';
      setWelcomeError(message);
      notify(message);
    } finally {
      directoryPickerInFlightRef.current = false;
      setDirectoryPickerPending(null);
    }
  };

  const finalizeProgressiveScan = async (sessionId: string, registrationName: string, operationId: number) => {
    try {
      const registered = await backend.registerScan(sessionId, registrationName || directoryBasename(rootPath));
      if (operationId !== datasetOperationRef.current) return;
      const assets = await backend.openRegisteredDataset(registered.dataset_id);
      if (operationId !== datasetOperationRef.current) return;
      const refreshed = datasetFromRegistered(registered, assets);
      setOpenedDataset((current) => current?.id === registered.dataset_id ? { ...refreshed, files: refreshed.files } : current);
      await searchDatasetAssets(registered.dataset_id, realSearchInput).catch(() => undefined);
      setAutoScanIntent(null);
      autoOpenedScanSessionsRef.current.add(sessionId);
      scanFinalizeAttemptsRef.current.delete(sessionId);
      void backend.refreshDatasets();
      notify(`${registered.name} 已在后台加载完成，共 ${assets.total.toLocaleString()} 项`);
    } catch (error) {
      autoOpenedScanSessionsRef.current.delete(sessionId);
      if (operationId !== datasetOperationRef.current) return;
      const attempts = (scanFinalizeAttemptsRef.current.get(sessionId) ?? 0) + 1;
      scanFinalizeAttemptsRef.current.set(sessionId, attempts);
      const message = error instanceof Error ? error.message : '后台数据索引合并失败';
      if (attempts < 3) {
        window.setTimeout(() => {
          if (operationId !== datasetOperationRef.current) return;
          setAutoScanIntent((current) => current?.sessionId === sessionId ? { ...current } : current);
        }, attempts * 600);
      } else {
        setAutoOpenError(`${message}；可重新打开该项目恢复最终索引`);
        notify(message);
      }
    }
  };

  useEffect(() => {
    const session = backend.scan.data;
    if (!autoScanIntent || !session || session.session_id !== autoScanIntent.sessionId || autoScanIntent.operationId !== datasetOperationRef.current) return;
    const hasSelectableBatch = session.items.some((item) => item.selectable && item.status === 'valid');
    if (session.state === 'running'
      && session.persisted_items >= 32
      && hasSelectableBatch
      && !progressivelyOpenedScanSessionsRef.current.has(session.session_id)) {
      progressivelyOpenedScanSessionsRef.current.add(session.session_id);
      const timer = window.setTimeout(() => {
        void finishAutoScanRef.current(session.session_id, autoScanIntent.name, autoScanIntent.operationId, true);
      }, 0);
      return () => window.clearTimeout(timer);
    }
    if (session.state !== 'succeeded' || autoOpenedScanSessionsRef.current.has(session.session_id)) return;
    const timer = window.setTimeout(() => {
      if (autoOpenedScanSessionsRef.current.has(session.session_id)) return;
      autoOpenedScanSessionsRef.current.add(session.session_id);
      if (progressivelyOpenedScanSessionsRef.current.has(session.session_id)) {
        void finalizeProgressiveScan(session.session_id, autoScanIntent.name, autoScanIntent.operationId);
      } else {
        void finishAutoScanRef.current(session.session_id, autoScanIntent.name, autoScanIntent.operationId);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  // finalizeProgressiveScan intentionally consumes the latest operation-scoped state.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoScanIntent, backend.scan.data]);

  const closeModelPicker = useCallback(() => {
    setModelPickerOpen(false);
    window.requestAnimationFrame(() => modelPickerTriggerRef.current?.focus());
  }, []);

  const chooseModel = (id: string, closePicker = true, autoLoad = true) => {
    const model = displayedModelCatalog.find((item) => item.id === id);
    if (!model) return;
    const operationId = ++modelSelectionOperationRef.current;
    setRemoteInferenceConfirmation(null);
    setSelectedModelId(id);
    try { window.localStorage.setItem(LAST_SELECTED_MODEL_KEY, id); } catch { /* persistence is best effort */ }
    setInferenceParameters(inferenceParameterDefaults(model.parametersSchema));
    setSelectedLayerId('');
    setModelLoaded(false);
    setModelLoadError('');
    samBoxDragRef.current = null;
    setSamPoints([]);
    setSamBoxes([]);
    setSamBoxPreview(null);
    setSamPromptMode(false);
    if (autoLoad && model.availability === 'available') {
      if (closePicker) closeModelPicker();
      if (model.runtimeState === 'loaded') {
        void loadModelLayers(model.id).then((runtime) => {
          if (runtime?.state === 'loaded') setModelLoaded(true);
        }).catch(() => undefined);
        void recordModelUsage(model.id).catch(() => undefined);
        notify(`${model.name} 已经加载`);
        return;
      }
      notify(`正在加载 ${model.name}…`);
      void loadModelById(model, operationId);
    } else if (model.availability === 'missing_weights') {
      notify(`已选择 ${model.name}，请先下载权重`);
    }
  };

  const chooseModelTask = (task: string) => {
    setModelTask(task);
  };

  const refreshModelWeights = useCallback(async (modelId: string) => {
    try {
      return await listModelWeights(modelId);
    } catch {
      return null;
    }
  }, [listModelWeights]);

  const toggleModelPicker = () => {
    if (modelPickerOpen) {
      closeModelPicker();
      return;
    }
    setModelPickerOpen(true);
    if (backend.mode !== 'online' || modelStatusRefreshing) return;
    setModelStatusRefreshing(true);
    void Promise.all([
      refreshBackendModels(),
      selectedModel.id ? refreshModelWeights(selectedModel.id) : Promise.resolve(null),
      refreshBackendHealth(),
    ]).catch((error) => {
      notify(error instanceof Error ? error.message : '模型状态刷新失败');
    }).finally(() => setModelStatusRefreshing(false));
  };

  useEffect(() => {
    if (rightTab !== 'inference' || backend.mode !== 'online' || !selectedModel.real) return;
    void refreshModelWeights(selectedModel.id);
  }, [backend.mode, refreshModelWeights, rightTab, selectedModel.id, selectedModel.real]);

  useEffect(() => {
    for (const job of backendJobs) {
      const completionKey = `${job.job_id}:${job.generation}:${job.state}`;
      if (job.kind !== 'model_download' || !['succeeded', 'succeeded_with_errors', 'failed', 'canceled'].includes(job.state) || completedWeightJobsRef.current.has(completionKey)) continue;
      completedWeightJobsRef.current.add(completionKey);
      const modelId = job.request.model_id;
      if (job.state === 'succeeded' && modelId) {
        void Promise.all([refreshModelWeights(modelId), refreshBackendModels(), refreshBackendHealth()]);
      }
    }
  }, [backendJobs, refreshBackendHealth, refreshBackendModels, refreshModelWeights]);

  useEffect(() => {
    for (const job of backendJobs) {
      const completionKey = `${job.job_id}:${job.generation}:${job.state}`;
      if (job.kind !== 'category_rename' || !['succeeded', 'succeeded_with_errors', 'failed', 'canceled'].includes(job.state) || completedCategoryRenameJobsRef.current.has(completionKey)) continue;
      completedCategoryRenameJobsRef.current.add(completionKey);
      if (job.dataset_id !== dataset.id) continue;
      void (async () => {
        await Promise.all([
          refreshBackendDatasets(),
          searchDatasetAssets(job.dataset_id, realSearchInput).catch(() => null),
          currentFile ? revalidateDatasetAsset(job.dataset_id, currentFile.id).catch(() => null) : Promise.resolve(null),
        ]);
        if (currentFile && !annotationDirtyRef.current) {
          await annotationLoaderRef.current(currentFile, job.dataset_id);
        }
        notify(job.state === 'succeeded'
          ? '全数据集类别重命名已完成'
          : job.state === 'succeeded_with_errors'
            ? `类别重命名部分完成：${job.failed} 张失败，可在右上角后台任务中重试`
            : `类别重命名任务${backgroundTaskStateLabel(job.state)}`);
      })();
    }
  }, [backendJobs, currentFile, dataset.id, realSearchInput, refreshBackendDatasets, revalidateDatasetAsset, searchDatasetAssets]);

  const handleWheel = useCallback((event: WheelEvent) => {
    const stage = stageRef.current;
    if (!stage) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest(CANVAS_CONTROL_SELECTOR)) return;
    event.preventDefault();
    if (gestureZoomRef.current && (event.ctrlKey || event.metaKey)) return;
    stage.focus({ preventScroll: true });
    const activePreviewPane = target?.closest<HTMLElement>('[data-pipeline-preview-pane]');
    const rect = activePreviewPane ? activePreviewPane.getBoundingClientRect() : stage.getBoundingClientRect();
    const paneAnchor = {
      x: event.clientX - (rect.left + rect.width / 2),
      y: event.clientY - (rect.top + rect.height / 2),
    };
    const referenceWidth = stage.clientWidth;
    const referenceHeight = stage.clientHeight;
    const anchor = activePreviewPane
      ? pipelinePaneVectorToReference(paneAnchor, rect.width, rect.height, referenceWidth, referenceHeight)
      : paneAnchor;
    const wheelInput = activePreviewPane ? pipelineWheelInputToReference({
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      deltaMode: event.deltaMode,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
    }, rect.width, rect.height, referenceWidth, referenceHeight) : event;
    setView((old) => applyCanvasWheel(old, wheelInput, maximumCanvasScale(), anchor, rect.height).view);
  }, [maximumCanvasScale]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    stage.addEventListener('wheel', handleWheel, { passive: false });
    return () => stage.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  useEffect(() => {
    const preventBrowserWheelZoom = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) event.preventDefault();
    };
    window.addEventListener('wheel', preventBrowserWheelZoom, { capture: true, passive: false });
    return () => window.removeEventListener('wheel', preventBrowserWheelZoom, { capture: true });
  }, []);

  useEffect(() => {
    type WebKitGestureEvent = Event & { scale?: number; clientX?: number; clientY?: number };
    const gestureTarget = (event: Event) => event.composedPath().find((item): item is Element => item instanceof Element) ?? null;
    const gestureStartsOnCanvas = (event: Event) => {
      const stage = stageRef.current;
      const target = gestureTarget(event);
      return Boolean(stage && event.composedPath().includes(stage) && !target?.closest(CANVAS_CONTROL_SELECTOR));
    };
    const onGestureStart = (event: Event) => {
      event.preventDefault();
      if (!gestureStartsOnCanvas(event)) {
        gestureZoomRef.current = null;
        return;
      }
      const gesture = event as WebKitGestureEvent;
      const scale = Number.isFinite(gesture.scale) && Number(gesture.scale) > 0 ? Number(gesture.scale) : 1;
      gestureZoomRef.current = { lastScale: scale };
      stageRef.current?.focus({ preventScroll: true });
    };
    const onGestureChange = (event: Event) => {
      event.preventDefault();
      const activeGesture = gestureZoomRef.current;
      const stage = stageRef.current;
      if (!activeGesture || !stage) return;
      const gesture = event as WebKitGestureEvent;
      const scale = Number(gesture.scale);
      if (!Number.isFinite(scale) || scale <= 0) return;
      const factor = scale / activeGesture.lastScale;
      activeGesture.lastScale = scale;
      const target = gestureTarget(event);
      const pane = target?.closest<HTMLElement>('[data-pipeline-preview-pane]');
      const rect = pane?.getBoundingClientRect() ?? stage.getBoundingClientRect();
      const clientX = Number.isFinite(gesture.clientX) ? Number(gesture.clientX) : rect.left + rect.width / 2;
      const clientY = Number.isFinite(gesture.clientY) ? Number(gesture.clientY) : rect.top + rect.height / 2;
      const paneAnchor = {
        x: clientX - (rect.left + rect.width / 2),
        y: clientY - (rect.top + rect.height / 2),
      };
      const anchor = pane
        ? pipelinePaneVectorToReference(paneAnchor, rect.width, rect.height, stage.clientWidth, stage.clientHeight)
        : paneAnchor;
      setView((old) => zoomCanvasView(old, factor, maximumCanvasScale(), anchor));
    };
    const onGestureEnd = (event: Event) => {
      event.preventDefault();
      gestureZoomRef.current = null;
    };
    window.addEventListener('gesturestart', onGestureStart, { capture: true, passive: false });
    window.addEventListener('gesturechange', onGestureChange, { capture: true, passive: false });
    window.addEventListener('gestureend', onGestureEnd, { capture: true, passive: false });
    return () => {
      gestureZoomRef.current = null;
      window.removeEventListener('gesturestart', onGestureStart, { capture: true });
      window.removeEventListener('gesturechange', onGestureChange, { capture: true });
      window.removeEventListener('gestureend', onGestureEnd, { capture: true });
    };
  }, [maximumCanvasScale]);

  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setNavigatorLayoutVersion((version) => version + 1));
    const targets = [stageRef.current, imageRef.current, navigatorImageRef.current].filter((target): target is HTMLElement => Boolean(target));
    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, [backend.pipeline.data?.asset_id, backend.pipeline.data?.dataset_id, backend.pipeline.data?.height, backend.pipeline.data?.width, currentFile?.height, currentFile?.id, currentFile?.width, dataset.id, pipelineEnabled, pipelineScope]);

  const focusCanvasFromPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest(CANVAS_CONTROL_SELECTOR)) return;
    event.currentTarget.focus({ preventScroll: true });
    const blankTarget = target === event.currentTarget || target?.classList.contains('screen-annotation-layer') === true;
    if (shouldSwitchToSelectAfterBlankClick({
      armed: selectOnNextCanvasBlankRef.current,
      tool,
      button: event.button,
      spaceDown,
      blankTarget,
    })) {
      selectOnNextCanvasBlankRef.current = false;
      event.preventDefault();
      event.stopPropagation();
      setTool('select');
    }
  };

  const hideCanvasCrosshair = () => {
    canvasCrosshairRef.current?.removeAttribute('data-visible');
  };

  const cancelPendingPixelSample = () => {
    pixelSampleGenerationRef.current += 1;
    pendingPixelSampleRef.current = null;
    if (pixelSampleFrameRef.current !== null) window.cancelAnimationFrame(pixelSampleFrameRef.current);
    pixelSampleFrameRef.current = null;
  };

  const readDisplayedPixel = (clientX: number, clientY: number, activePane: HTMLElement | null): PixelSample | null => {
    const surface = activePane?.isConnected ? activePane : stageRef.current ?? imageRef.current;
    if (!surface) return null;
    const tiles = Array.from(surface.querySelectorAll<HTMLImageElement>('.tiled-image-tile')).reverse();
    const fullImage = surface.querySelector<HTMLImageElement>('.real-image');
    const previewImages = Array.from(surface.querySelectorAll<HTMLImageElement>('.pipeline-preview-image')).filter((image) => !image.closest('figure.hidden')).reverse();
    const placeholder = surface.querySelector<HTMLImageElement>('.tiled-image-placeholder');
    const candidates = [...previewImages, ...tiles, ...(fullImage ? [fullImage] : []), ...(placeholder ? [placeholder] : [])];
    for (const image of candidates) {
      if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) continue;
      const imageBounds = image.getBoundingClientRect();
      const rect = image.classList.contains('pipeline-preview-image')
        ? containedPipelineImageRect(imageBounds, image.naturalWidth, image.naturalHeight)
        : imageBounds;
      const source = sourcePixelAtDisplayPoint(clientX, clientY, rect, image.naturalWidth, image.naturalHeight);
      if (!source) continue;
      const canvas = pixelSampleCanvasRef.current ?? document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      pixelSampleCanvasRef.current = canvas;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) return null;
      try {
        context.clearRect(0, 0, 1, 1);
        context.imageSmoothingEnabled = false;
        context.drawImage(image, source.x, source.y, 1, 1, 0, 0, 1, 1);
        return pixelSampleFromRgba(context.getImageData(0, 0, 1, 1).data);
      } catch {
        pixelSampleCanvasRef.current = null;
        return null;
      }
    }
    return null;
  };

  const schedulePixelSample = (sample: { clientX: number; clientY: number; x: number; y: number; pane: HTMLElement | null }) => {
    pendingPixelSampleRef.current = sample;
    if (pixelSampleFrameRef.current !== null) return;
    const generation = pixelSampleGenerationRef.current;
    pixelSampleFrameRef.current = window.requestAnimationFrame(() => {
      pixelSampleFrameRef.current = null;
      const latest = pendingPixelSampleRef.current;
      pendingPixelSampleRef.current = null;
      if (!latest || generation !== pixelSampleGenerationRef.current) return;
      setCursor({
        x: latest.x,
        y: latest.y,
        insideImage: true,
        pixel: readDisplayedPixel(latest.clientX, latest.clientY, latest.pane),
      });
    });
  };

  const hideCanvasGuides = () => {
    hideCanvasCrosshair();
    cancelPendingPixelSample();
    setPipelineSharedCursor(null);
    setCursor((old) => old.insideImage || old.pixel ? { ...old, insideImage: false, pixel: null } : old);
  };

  const updateCanvasCrosshair = (event: React.MouseEvent<HTMLDivElement> | React.PointerEvent<HTMLDivElement>) => {
    const crosshair = canvasCrosshairRef.current;
    if (!crosshair) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const target = event.target instanceof Element ? event.target : null;
    const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    const pointerType = 'pointerType' in event ? event.pointerType : 'mouse';
    if (!inside || pointerType === 'touch' || dragRef.current || target?.closest(CANVAS_CONTROL_SELECTOR)) {
      hideCanvasCrosshair();
      return;
    }
    crosshair.style.setProperty('--canvas-crosshair-x', `${Math.round(event.clientX - rect.left)}px`);
    crosshair.style.setProperty('--canvas-crosshair-y', `${Math.round(event.clientY - rect.top)}px`);
    crosshair.dataset.visible = 'true';
  };

  const preventCanvasNativeDrag = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = event.target instanceof Element ? event.target : null;
    if (event.button === 0 && tool === 'select' && !spaceDown && !target?.closest(CANVAS_CONTROL_SELECTOR)) {
      event.preventDefault();
      return;
    }
    const primaryPan = event.button === 0 && (spaceDown || tool === 'pan');
    if (!primaryPan && event.button !== 1) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { startX: event.clientX, startY: event.clientY, viewX: view.x, viewY: view.y };
    setCanvasPanning(true);
    hideCanvasGuides();
  };

  const handleCanvasMove = (event: React.MouseEvent<HTMLDivElement> | React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current) {
      setView((old) => ({ ...old, x: dragRef.current!.viewX + event.clientX - dragRef.current!.startX, y: dragRef.current!.viewY + event.clientY - dragRef.current!.startY }));
      return;
    }
    const target = event.target instanceof Element ? event.target : null;
    const pointerType = 'pointerType' in event ? event.pointerType : 'mouse';
    if (pointerType === 'touch' || target?.closest(CANVAS_CONTROL_SELECTOR)) {
      cancelPendingPixelSample();
      setPipelineSharedCursor(null);
      setCursor((old) => old.insideImage || old.pixel ? { ...old, insideImage: false, pixel: null } : old);
      return;
    }
    const activePreviewPane = target?.closest<HTMLElement>('[data-pipeline-preview-pane]');
    const rect = activePreviewPane ? pipelinePreviewContainedRect(activePreviewPane) : imageRef.current?.getBoundingClientRect();
    if (!rect) return;
    const insideImage = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!insideImage) {
      cancelPendingPixelSample();
      setPipelineSharedCursor(null);
      setCursor((old) => old.insideImage || old.pixel ? { ...old, insideImage: false, pixel: null } : old);
      return;
    }
    const rx = (event.clientX - rect.left) / rect.width;
    const ry = (event.clientY - rect.top) / rect.height;
    const paneWidth = Number(activePreviewPane?.dataset.pipelineWidth);
    const paneHeight = Number(activePreviewPane?.dataset.pipelineHeight);
    const pixelWidth = Math.max(1, Number.isFinite(paneWidth) && paneWidth > 0 ? paneWidth : displayedWidth ?? currentFile?.width ?? 8192);
    const pixelHeight = Math.max(1, Number.isFinite(paneHeight) && paneHeight > 0 ? paneHeight : displayedHeight ?? currentFile?.height ?? 6144);
    const localX = Math.min(pixelWidth - Number.EPSILON, Math.max(0, rx * pixelWidth));
    const localY = Math.min(pixelHeight - Number.EPSILON, Math.max(0, ry * pixelHeight));
    const activeVisualizationId = activePreviewPane?.dataset.pipelineVisualizationId;
    const activePipelineItem = activeVisualizationId ? pipelineDisplayItems.find((item) => item.visualization_id === activeVisualizationId) : undefined;
    setPipelineSharedCursor(activePreviewPane && activeVisualizationId
      ? createPipelineSharedCursor(
        activeVisualizationId,
        activePreviewPane.dataset.pipelineLabel ?? '显示',
        localX,
        localY,
        activePipelineItem ? pipelineCoordinateMappingForItem(activePipelineItem) : null,
        pixelWidth,
        pixelHeight,
      )
      : null);
    schedulePixelSample({
      clientX: event.clientX,
      clientY: event.clientY,
      x: Math.min(pixelWidth - 1, Math.max(0, Math.floor(localX))),
      y: Math.min(pixelHeight - 1, Math.max(0, Math.floor(localY))),
      pane: activePreviewPane ?? null,
    });
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    lastPointerMoveAtRef.current = performance.now();
    handleCanvasMove(event);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (performance.now() - lastPointerMoveAtRef.current < 24) return;
    handleCanvasMove(event);
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current && event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    dragRef.current = null;
    setCanvasPanning(false);
    if (event.type === 'pointercancel') hideCanvasGuides();
    else updateCanvasCrosshair(event);
  };

  useEffect(() => {
    pixelSampleGenerationRef.current += 1;
    pendingPixelSampleRef.current = null;
    if (pixelSampleFrameRef.current !== null) window.cancelAnimationFrame(pixelSampleFrameRef.current);
    pixelSampleFrameRef.current = null;
    pixelSampleCanvasRef.current = null;
    setPipelineSharedCursor(null);
    setCursor((old) => old.insideImage || old.pixel ? { ...old, insideImage: false, pixel: null } : old);
    return () => {
      pixelSampleGenerationRef.current += 1;
      if (pixelSampleFrameRef.current !== null) window.cancelAnimationFrame(pixelSampleFrameRef.current);
      pixelSampleFrameRef.current = null;
    };
  }, [backend.pipeline.data?.artifact_id, currentFile?.id, pipelineEnabled, pipelineScope, visualizationDisplayMode]);

  const readNavigatorMetrics = (): CanvasNavigatorMetrics | null => {
    const navigator = navigatorImageRef.current;
    const image = imageRef.current;
    const stage = stageRef.current;
    if (!navigator || !image || !stage || !navigator.clientWidth || !navigator.clientHeight || !image.offsetWidth || !image.offsetHeight || !stage.clientWidth || !stage.clientHeight) return null;
    return {
      navigatorWidth: navigator.clientWidth,
      navigatorHeight: navigator.clientHeight,
      imageWidth: image.offsetWidth,
      imageHeight: image.offsetHeight,
      viewportWidth: stage.clientWidth,
      viewportHeight: stage.clientHeight,
    };
  };

  const updateViewFromNavigator = (event: React.PointerEvent<HTMLDivElement>) => {
    const navigator = navigatorImageRef.current;
    const metrics = readNavigatorMetrics();
    if (!navigator || !metrics) return;
    const rect = navigator.getBoundingClientRect();
    setView((current) => navigatorPointToView({ x: event.clientX - rect.left, y: event.clientY - rect.top }, current, metrics));
  };

  const startNavigatorDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    navigatorDragRef.current = { pointerId: event.pointerId };
    updateViewFromNavigator(event);
  };

  const moveNavigatorDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (navigatorDragRef.current?.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    updateViewFromNavigator(event);
  };

  const endNavigatorDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (navigatorDragRef.current?.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    navigatorDragRef.current = null;
  };

  const imagePoint = (event: Pick<React.PointerEvent<SVGElement>, 'clientX' | 'clientY'>) => {
    const rect = imageRef.current?.getBoundingClientRect();
    const width = displayedWidth ?? 1;
    const height = displayedHeight ?? 1;
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.max(0, Math.min(width, (event.clientX - rect.left) * width / rect.width)),
      y: Math.max(0, Math.min(height, (event.clientY - rect.top) * height / rect.height)),
    };
  };

  const sourceAnnotationPoint = (point: { x: number; y: number }) => {
    if (!canvasEditCoordinateTransform) return point;
    const [x, y] = inverseTransformCanvasPoint([point.x, point.y], canvasEditCoordinateTransform);
    return { x, y };
  };

  const canvasAnnotationHitCandidates = (event: Pick<React.PointerEvent<SVGElement>, 'clientX' | 'clientY'>) => {
    if (!showGT) return [];
    const point = imagePoint(event);
    return annotationHitCandidates(displayedShapes, [point.x, point.y], 7 * imageUnitsPerScreenPixel(), hiddenShapeIndexes);
  };
  const canvasPredictionHitCandidates = (event: Pick<React.PointerEvent<SVGElement>, 'clientX' | 'clientY'>) => {
    if (showingPipelineImage || currentAnnotationsAreSegmentation) return [];
    const point = imagePoint(event);
    const hidden = new Set<number>();
    currentDetectionPredictions.forEach((prediction, index) => {
      const key = inferencePredictionKey(selectedModel.id, currentInferenceResult?.image_path ?? '', prediction);
      if (promotedPredictionKeys.has(key) || hiddenPredictionCategories.has(prediction.label.trim() || '未命名预测') || hiddenPredictionKeys.has(key)) hidden.add(index);
    });
    return annotationHitCandidates(currentDetectionPredictions as AnnotationShape[], [point.x, point.y], 7 * imageUnitsPerScreenPixel(), hidden);
  };

  const startCanvasPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!isSamModel || !samPromptMode) {
      if (tool === 'select' && event.button === 0 && !spaceDown) {
        const candidates = canvasAnnotationHitCandidates(event);
        const currentIndex = selectedShapeIndex;
        const hitIndex = selectAnnotationHitIndex(candidates, currentIndex, event.detail > 1);
        if (hitIndex !== null) {
          setSelectedPredictionIndex(null);
          setObjectSourceTab('manual');
          startShapeMove(hitIndex, event);
          return;
        }
        const predictionHitIndex = selectAnnotationHitIndex(canvasPredictionHitCandidates(event), selectedPredictionIndex, event.detail > 1);
        if (predictionHitIndex !== null) {
          event.preventDefault();
          event.stopPropagation();
          if (event.detail > 1) {
            promotePredictionToManual(predictionHitIndex);
            return;
          }
          setSelectedShapeIndex(null);
          setSelectedPredictionIndex(predictionHitIndex);
          setObjectSourceTab('ai');
          setRightTab('layers');
          return;
        }
        event.preventDefault();
      }
      startRealDraw(event);
      return;
    }
    if (spaceDown || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const point = imagePoint(event);
    if (samPromptTool === 'positive' || samPromptTool === 'negative') {
      setSamPoints((old) => [...old, { x: point.x, y: point.y, label: samPromptTool === 'positive' ? 1 : 0 }]);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    samBoxDragRef.current = { startX: point.x, startY: point.y };
    setSamBoxPreview([point.x, point.y, point.x, point.y]);
  };

  const moveCanvasPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!isSamModel || !samPromptMode) {
      if (shapeDragRef.current) {
        moveShape(event);
        return;
      }
      moveRealDraw(event);
      return;
    }
    const drag = samBoxDragRef.current;
    if (!drag) return;
    event.preventDefault();
    const point = imagePoint(event);
    setSamBoxPreview([
      Math.min(drag.startX, point.x),
      Math.min(drag.startY, point.y),
      Math.max(drag.startX, point.x),
      Math.max(drag.startY, point.y),
    ]);
  };

  const endCanvasPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!isSamModel || !samPromptMode) {
      if (shapeDragRef.current) {
        endShapeMove(event);
        return;
      }
      endRealDraw(event);
      return;
    }
    const drag = samBoxDragRef.current;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    const point = imagePoint(event);
    const box: SamPromptBox = [
      Math.min(drag.startX, point.x),
      Math.min(drag.startY, point.y),
      Math.max(drag.startX, point.x),
      Math.max(drag.startY, point.y),
    ];
    samBoxDragRef.current = null;
    setSamBoxPreview(null);
    if (box[2] - box[0] >= 2 && box[3] - box[1] >= 2) setSamBoxes((old) => [...old, box]);
  };

  const cancelCanvasPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!isSamModel || !samPromptMode) {
      if (shapeDragRef.current) {
        endShapeMove(event);
        return;
      }
      cancelRealDraw(event);
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    samBoxDragRef.current = null;
    setSamBoxPreview(null);
  };

  const handleCanvasDoubleClick = (event: React.MouseEvent<SVGSVGElement>) => {
    if (tool === 'select' && (canvasAnnotationHitCandidates(event).length > 0 || canvasPredictionHitCandidates(event).length > 0)) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (tool === 'line') {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (tool === 'polygon') {
      event.preventDefault();
      event.stopPropagation();
      if (polygonDraftRef.current.length >= 3) finishPolygonDraft({ x: event.clientX, y: event.clientY });
    }
  };

  const appendDraftShape = (shape: AnnotationShape, anchor = lastCanvasPointerRef.current) => {
    const envelope = annotationDraftRef.current;
    if (!envelope || pendingManualShapeRef.current) return;
    const imageBounds = imageRef.current?.getBoundingClientRect();
    const resolvedAnchor = Number.isFinite(anchor.x) && Number.isFinite(anchor.y)
      ? anchor
      : { x: imageBounds ? imageBounds.left + imageBounds.width / 2 : 24, y: imageBounds ? imageBounds.top + imageBounds.height / 2 : 24 };
    setManualShapeLabel('');
    const nextPending = {
      shape: { ...shape, points: shape.points.map((point) => [...point]) },
      datasetId: envelope.dataset_id,
      assetId: envelope.asset_id,
      anchor: resolvedAnchor,
    };
    pendingManualShapeRef.current = nextPending;
    setPendingManualShape(nextPending);
    setPolygonPointer(null);
    setPolygonCloseReady(false);
  };

  const cancelPendingManualShape = () => {
    pendingManualShapeRef.current = null;
    setPendingManualShape(null);
    setManualShapeLabel('');
    window.requestAnimationFrame(() => stageRef.current?.focus({ preventScroll: true }));
  };

  const commitPendingManualShape = (rawLabel = manualShapeLabel) => {
    const label = normalizeAnnotationLabel(rawLabel);
    if (!label) {
      notify('标签类别必须为 1–128 个字符');
      return;
    }
    const pending = pendingManualShapeRef.current;
    const envelope = annotationDraftRef.current;
    if (!pending || !envelope || envelope.dataset_id !== pending.datasetId || envelope.asset_id !== pending.assetId) {
      cancelPendingManualShape();
      notify('当前图片已经切换，这次未确认的标注已取消');
      return;
    }
    const document = envelope.document;
    const nextIndex = document.shapes?.length ?? 0;
    const displayShape = { ...pending.shape, label };
    const shape = canvasEditCoordinateTransform
      ? inverseTransformCanvasShape(displayShape, canvasEditCoordinateTransform)
      : displayShape;
    const committed = commitAnnotationDocument({ ...document, shapes: [...(document.shapes ?? []), shape] });
    if (!committed) {
      notify('标注框暂时无法写入当前图片，请保留类别选择后重试');
      return;
    }
    pendingManualShapeRef.current = null;
    setPendingManualShape(null);
    setManualShapeLabel('');
    setLastManualLabel(label);
    setSelectedShapeIndex(nextIndex);
    setSelectedShape(shape.shape_type);
    selectOnNextCanvasBlankRef.current = true;
    window.requestAnimationFrame(() => stageRef.current?.focus({ preventScroll: true }));
  };

  const finishPolygonDraft = (anchor = lastCanvasPointerRef.current) => {
    const points = compactFreehandPoints(polygonDraftRef.current, 0.5);
    if (points.length < 3) {
      notify('多边形至少需要 3 个点；继续点击或按 Esc 取消');
      return;
    }
    polygonDraftRef.current = [];
    setPolygonDraft([]);
    setPolygonPointer(null);
    setPolygonCloseReady(false);
    appendDraftShape({ label: 'object', shape_type: 'polygon', points }, anchor);
  };

  const startRealDraw = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!annotationDraft || pendingManualShape || spaceDown || event.button !== 0 || ['select', 'pan'].includes(tool)) return;
    const pointerAnchor = { x: event.clientX, y: event.clientY };
    lastCanvasPointerRef.current = pointerAnchor;
    event.preventDefault();
    const point = imagePoint(event);
    const coordinate: AnnotationPoint = [point.x, point.y];
    if (tool === 'point') {
      appendDraftShape({ label: 'object', shape_type: 'point', points: [coordinate] }, pointerAnchor);
      return;
    }
    if (tool === 'polygon') {
      if (event.detail > 1) {
        if (polygonDraftRef.current.length >= 3) finishPolygonDraft(pointerAnchor);
        return;
      }
      const closeTolerance = 14 * imageUnitsPerScreenPixel();
      if (canClosePolygonAtPoint(polygonDraftRef.current, coordinate, closeTolerance)) {
        finishPolygonDraft(pointerAnchor);
        return;
      }
      if (polygonDraftRef.current.length >= 10_000) { notify('多边形已达到 10,000 点安全上限'); return; }
      const next = [...polygonDraftRef.current, coordinate];
      polygonDraftRef.current = next;
      setPolygonDraft(next);
      setPolygonPointer(coordinate);
      setPolygonCloseReady(false);
      return;
    }
    if (tool === 'line') {
      const start = drawRef.current;
      if (!start) {
        drawRef.current = { startX: point.x, startY: point.y };
        setDrawPreview({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
        return;
      }
      const shape = createDragShape('line', [start.startX, start.startY], coordinate);
      if (!shape) {
        notify('直线两点距离过近，请重新选择终点');
        return;
      }
      drawRef.current = null;
      setDrawPreview(null);
      appendDraftShape(shape as AnnotationShape, pointerAnchor);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    if (tool === 'brush') {
      brushRef.current = { pointerId: event.pointerId, points: [coordinate] };
      setBrushPreview([coordinate]);
      return;
    }
    drawRef.current = { startX: point.x, startY: point.y };
    setDrawPreview({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  };

  const moveRealDraw = (event: React.PointerEvent<SVGSVGElement>) => {
    if (pendingManualShape) return;
    lastCanvasPointerRef.current = { x: event.clientX, y: event.clientY };
    if (tool === 'polygon' && polygonDraftRef.current.length > 0) {
      const point = imagePoint(event);
      const coordinate: AnnotationPoint = [point.x, point.y];
      const closeReady = canClosePolygonAtPoint(polygonDraftRef.current, coordinate, 14 * imageUnitsPerScreenPixel());
      setPolygonPointer(closeReady ? polygonDraftRef.current[0] : coordinate);
      setPolygonCloseReady(closeReady);
      return;
    }
    const brush = brushRef.current;
    if (brush) {
      if (brush.pointerId !== event.pointerId || brush.points.length >= 10_000) return;
      const point = imagePoint(event);
      const next = compactFreehandPoints([...brush.points, [point.x, point.y]], Math.max(0.5, (displayedWidth ?? 1000) / 2000));
      brushRef.current = { ...brush, points: next };
      setBrushPreview(next);
      return;
    }
    if (!drawRef.current) return;
    const point = imagePoint(event);
    setDrawPreview({ x1: drawRef.current.startX, y1: drawRef.current.startY, x2: point.x, y2: point.y });
  };

  const endRealDraw = (event: React.PointerEvent<SVGSVGElement>) => {
    const pointerAnchor = { x: event.clientX, y: event.clientY };
    lastCanvasPointerRef.current = pointerAnchor;
    const brush = brushRef.current;
    if (brush) {
      if (brush.pointerId !== event.pointerId) return;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      const endPoint = imagePoint(event);
      const points = compactFreehandPoints([...brush.points, [endPoint.x, endPoint.y]], Math.max(0.5, (displayedWidth ?? 1000) / 2000));
      brushRef.current = null;
      setBrushPreview([]);
      const shape = createFreehandLine(points, currentFile?.width ?? displayedWidth ?? 1, currentFile?.height ?? displayedHeight ?? 1);
      if (shape) appendDraftShape(shape as AnnotationShape, pointerAnchor);
      else notify('拖动距离太短，未创建连续线；请按住并拖动一段距离');
      return;
    }
    if (!drawRef.current || !annotationDraft || !['rect', 'rotation', 'circle'].includes(tool)) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    const point = imagePoint(event);
    const start: AnnotationPoint = [drawRef.current.startX, drawRef.current.startY];
    const end: AnnotationPoint = [point.x, point.y];
    drawRef.current = null;
    setDrawPreview(null);
    const shape = createDragShape(tool, start, end);
    if (shape) appendDraftShape(shape as AnnotationShape, pointerAnchor);
  };

  const cancelRealDraw = (event?: React.PointerEvent<SVGSVGElement>) => {
    if (event && event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    cancelAnnotationDrafting();
  };

  const startShapeMove = (index: number, event: React.PointerEvent<SVGElement>) => {
    if (tool !== 'select' || spaceDown) return;
    event.stopPropagation();
    event.preventDefault();
    const shape = draftShapes[index];
    if (!shape) return;
    const point = imagePoint(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    shapeDragRef.current = { index, startX: point.x, startY: point.y, points: shape.points.map((value) => [...value]) };
    annotationEditActiveRef.current = true;
    setAnnotationEditActive(true);
    setSelectedShapeIndex(index);
    setSelectedShape(shape.shape_type === 'rotation' ? 'rotation' : 'rectangle');
    setRightTab('layers');
  };

  const moveShape = (event: React.PointerEvent<SVGElement>) => {
    const drag = shapeDragRef.current;
    if (!drag) return;
    const point = imagePoint(event);
    const dx = point.x - drag.startX;
    const dy = point.y - drag.startY;
    const [sourceDx, sourceDy] = canvasEditCoordinateTransform ? inverseTransformCanvasDelta([dx, dy], canvasEditCoordinateTransform) : [dx, dy];
    const envelope = annotationDraftRef.current;
    if (!envelope) return;
    const shapes = [...(envelope.document.shapes ?? [])];
    shapes[drag.index] = translateShapeWithinImage(
      { ...shapes[drag.index], points: drag.points },
      sourceDx,
      sourceDy,
      currentFile?.width ?? displayedWidth ?? 1,
      currentFile?.height ?? displayedHeight ?? 1,
    ) as AnnotationShape;
    replaceDraftDocument({ ...envelope.document, shapes });
  };

  const endShapeMove = (event: React.PointerEvent<SVGElement>) => {
    if (shapeDragRef.current && event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    shapeDragRef.current = null;
    commitTransientAnnotationEdit();
  };

  const startControlPointMove = (shapeIndex: number, pointIndex: number, event: React.PointerEvent<SVGElement>) => {
    if (spaceDown || (tool !== 'select' && !(tool === 'rotation' && draftShapes[shapeIndex]?.shape_type === 'rotation'))) return;
    event.stopPropagation();
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    controlPointDragRef.current = { shapeIndex, pointIndex };
    annotationEditActiveRef.current = true;
    setAnnotationEditActive(true);
    setSelectedShapeIndex(shapeIndex);
  };

  const startNearestMultiPointMove = (shapeIndex: number, event: React.PointerEvent<SVGElement>) => {
    const shape = draftShapes[shapeIndex];
    if (!shape?.points.length) return;
    const pointer = sourceAnnotationPoint(imagePoint(event));
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    shape.points.forEach(([x, y], index) => {
      const distance = (x - pointer.x) ** 2 + (y - pointer.y) ** 2;
      if (distance < nearestDistance) { nearestDistance = distance; nearestIndex = index; }
    });
    startControlPointMove(shapeIndex, nearestIndex, event);
  };

  const moveControlPoint = (event: React.PointerEvent<SVGElement>) => {
    const drag = controlPointDragRef.current;
    if (!drag) return;
    const point = sourceAnnotationPoint(imagePoint(event));
    const envelope = annotationDraftRef.current;
    if (!envelope) return;
    const shapes = [...(envelope.document.shapes ?? [])];
    const shape = shapes[drag.shapeIndex];
    if (!shape) return;
    shapes[drag.shapeIndex] = moveShapeControlPoint(
      shape,
      drag.pointIndex,
      [point.x, point.y],
      currentFile?.width ?? displayedWidth ?? 1,
      currentFile?.height ?? displayedHeight ?? 1,
    ) as AnnotationShape;
    replaceDraftDocument({ ...envelope.document, shapes });
  };

  const endControlPointMove = (event: React.PointerEvent<SVGElement>) => {
    if (controlPointDragRef.current && event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    controlPointDragRef.current = null;
    commitTransientAnnotationEdit();
  };

  const startRealRotation = (shapeIndex: number, event: React.PointerEvent<SVGElement>) => {
    if (spaceDown || !['select', 'rotation'].includes(tool) || event.button !== 0) return;
    const shape = draftShapes[shapeIndex];
    const center = shape?.shape_type === 'rotation' ? rotationCenter(shape.points) : null;
    if (!shape || !center) return;
    event.stopPropagation();
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const pointer = sourceAnnotationPoint(imagePoint(event));
    realRotationDragRef.current = {
      shapeIndex,
      center,
      startAngle: Math.atan2(pointer.y - center[1], pointer.x - center[0]),
      startDirection: rotationDirection(shape.points),
      shape: { ...shape, points: shape.points.map((value) => [...value]) },
    };
    annotationEditActiveRef.current = true;
    setAnnotationEditActive(true);
    setSelectedShapeIndex(shapeIndex);
    setSelectedShape('rotation');
  };

  const moveRealRotation = (event: React.PointerEvent<SVGElement>) => {
    const drag = realRotationDragRef.current;
    if (!drag) return;
    event.stopPropagation();
    const point = sourceAnnotationPoint(imagePoint(event));
    const pointerAngle = Math.atan2(point.y - drag.center[1], point.x - drag.center[0]);
    let delta = Math.atan2(Math.sin(pointerAngle - drag.startAngle), Math.cos(pointerAngle - drag.startAngle));
    if (event.shiftKey) {
      const step = Math.PI / 12;
      const snappedDirection = Math.round((drag.startDirection + delta) / step) * step;
      delta = Math.atan2(Math.sin(snappedDirection - drag.startDirection), Math.cos(snappedDirection - drag.startDirection));
    }
    const envelope = annotationDraftRef.current;
    if (!envelope) return;
    const shapes = [...(envelope.document.shapes ?? [])];
    shapes[drag.shapeIndex] = rotateRotationShape(drag.shape, delta) as AnnotationShape;
    replaceDraftDocument({ ...envelope.document, shapes });
  };

  const endRealRotation = (event: React.PointerEvent<SVGElement>) => {
    event.stopPropagation();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!realRotationDragRef.current) return;
    realRotationDragRef.current = null;
    commitTransientAnnotationEdit();
  };

  async function persistCurrentAnnotation(manual: boolean): Promise<AnnotationPersistOutcome> {
    const captured = annotationDraftRef.current;
    if (!captured) {
      if (manual) notify('当前不是可保存的真实标注');
      return 'failed';
    }
    const dirty = documentIsDirty(captured.document);
    if (!dirty) {
      await deleteAnnotationDraft(captured.dataset_id, captured.asset_id).catch(() => undefined);
      annotationDirtyRef.current = false;
      setAnnotationDirty(false);
      setAnnotationPersistence({ phase: 'saved', message: '已与服务端同步' });
      if (manual) notify('当前标注已经保存');
      return 'unchanged';
    }
    const capturedDocument = structuredClone(captured.document);
    let localDraftSaved = false;
    try {
      await putAnnotationDraft({
        dataset_id: captured.dataset_id,
        asset_id: captured.asset_id,
        base_revision: captured.revision,
        document: capturedDocument,
      });
      localDraftSaved = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'IndexedDB 草稿写入失败';
      setAnnotationPersistence({ phase: 'error', message });
      if (manual) notify(message);
      if (backendModeRef.current !== 'online') return 'failed';
    }
    if (!shouldWriteAnnotationFile(manual, annotationAutoSaveRef.current)) {
      setAnnotationPersistence({ phase: 'local', message: '自动保存已关闭；本机草稿已保护' });
      return localDraftSaved ? 'local' : 'failed';
    }
    if (backendModeRef.current !== 'online') {
      setAnnotationPersistence({ phase: 'offline', message: '服务离线，仅保存到本机 IndexedDB' });
      if (manual) notify('服务离线：草稿仅保存在本机，未写入标注文件');
      return localDraftSaved ? 'local' : 'failed';
    }
    if (annotationSavePendingRef.current) {
      if (manual) notify('已有标注保存请求正在进行');
      return 'busy';
    }
    annotationSavePendingRef.current = true;
    setAnnotationSaving(true);
    setAnnotationPersistence({ phase: 'saving', message: manual ? '正在显式保存' : '正在自动保存' });
    try {
      const saved = await saveAnnotationRef.current(captured.dataset_id, captured.asset_id, captured.revision, capturedDocument);
      const current = annotationDraftRef.current;
      if (!current || current.dataset_id !== captured.dataset_id || current.asset_id !== captured.asset_id) return 'failed';
      annotationBaseDocumentRef.current = structuredClone(capturedDocument);
      const nextEnvelope = { ...current, revision: saved.revision };
      annotationDraftRef.current = nextEnvelope;
      setAnnotationDraft(nextEnvelope);
      const hasNewerChanges = annotationFingerprint(current.document) !== annotationFingerprint(capturedDocument);
      annotationDirtyRef.current = hasNewerChanges;
      setAnnotationDirty(hasNewerChanges);
      setOpenedDataset((old) => old ? {
        ...old,
        files: old.files.map((file) => file.id === captured.asset_id ? {
          ...file,
          annotations: capturedDocument.shapes?.length ?? 0,
          annotationFileExists: true,
          labels: [...new Set((capturedDocument.shapes ?? []).map((shape) => shape.label.trim()).filter(Boolean))],
          meta: `${capturedDocument.shapes?.length ?? 0} 标注 · ${file.width ?? '?'} × ${file.height ?? '?'}`,
        } : file),
      } : old);
      void searchDatasetAssets(captured.dataset_id, realSearchInput).catch(() => undefined);
      if (hasNewerChanges) {
        try {
          await putAnnotationDraft({
            dataset_id: current.dataset_id,
            asset_id: current.asset_id,
            base_revision: saved.revision,
            document: current.document,
          });
        } catch (error) {
          setAnnotationPersistence({ phase: 'error', message: error instanceof Error ? `服务端已保存，但新草稿持久化失败：${error.message}` : '服务端已保存，但新草稿持久化失败' });
        }
        scheduleAnnotationPersistence('保存期间出现新编辑，等待保护最新草稿');
        return 'failed';
      } else {
        try {
          await deleteAnnotationDraft(current.dataset_id, current.asset_id);
          setAnnotationPersistence({ phase: 'saved', message: manual ? '显式保存完成' : '自动保存完成' });
        } catch (error) {
          setAnnotationPersistence({ phase: 'error', message: error instanceof Error ? `服务端已保存，但本地草稿清理失败：${error.message}` : '服务端已保存，但本地草稿清理失败' });
        }
      }
      if (manual) notify('标注已原子保存并创建历史备份');
      return 'remote';
    } catch (error) {
      const message = error instanceof Error ? error.message : '标注保存失败';
      setAnnotationPersistence({ phase: 'error', message: `保存失败，草稿仍在本机：${message}` });
      if (manual) notify(message);
      return 'failed';
    } finally {
      annotationSavePendingRef.current = false;
      setAnnotationSaving(false);
    }
  }

  const saveCurrentAnnotation = async () => {
    if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    annotationAutoSaveTimerRef.current = null;
    return persistCurrentAnnotation(true);
  };

  const changeAnnotationAutoSave = (enabled: boolean) => {
    annotationAutoSaveRef.current = enabled;
    setAnnotationAutoSave(enabled);
    const current = annotationDraftRef.current;
    if (current && documentIsDirty(current.document)) {
      scheduleAnnotationPersistence(enabled ? '自动保存已开启，等待 700ms' : '自动保存已关闭，正在保护本机草稿');
    }
    notify(enabled ? '自动保存标注已开启' : '自动保存标注已关闭；切图时将询问是否保存');
  };

  const pendingNavigationSourceIsCurrent = (pending: PendingAnnotationNavigation) => {
    const current = annotationDraftRef.current;
    return Boolean(
      current
      && current.dataset_id === pending.sourceDatasetId
      && current.asset_id === pending.sourceAssetId
      && annotationFingerprint(current.document) === pending.sourceFingerprint,
    );
  };

  const closeAnnotationNavigationPrompt = (resumeAutoSave: boolean) => {
    setPendingAnnotationNavigation(null);
    setAnnotationNavigationDecision(null);
    setAnnotationNavigationError('');
    const current = annotationDraftRef.current;
    if (resumeAutoSave && current && documentIsDirty(current.document)) scheduleAnnotationPersistence();
  };

  const keepChangesAndNavigate = async () => {
    const pending = pendingAnnotationNavigation;
    if (!pending) return;
    if (!pendingNavigationSourceIsCurrent(pending)) {
      setAnnotationNavigationError('当前标注在确认期间发生了变化。请取消后重新切换，避免保存错误版本。');
      return;
    }
    setAnnotationNavigationDecision('keep');
    setAnnotationNavigationError('');
    const outcome = await saveCurrentAnnotation();
    if (outcome === 'remote' || outcome === 'local' || outcome === 'unchanged') {
      annotationDirtyRef.current = false;
      setPendingAnnotationNavigation(null);
      setAnnotationNavigationDecision(null);
      performFileSelection(pending.file, pending.datasetId);
      return;
    }
    setAnnotationNavigationDecision(null);
    setAnnotationNavigationError(outcome === 'busy' ? '当前保存尚未完成，请稍后再继续切换。' : '更改未能安全保留，已停留在当前图片。你可以重试或取消切换。');
  };

  const discardChangesAndNavigate = async () => {
    const pending = pendingAnnotationNavigation;
    if (!pending) return;
    if (annotationSavePendingRef.current || annotationSaving) {
      setAnnotationNavigationError('当前保存请求已经开始，完成前不能选择“不保留”。');
      return;
    }
    if (!pendingNavigationSourceIsCurrent(pending)) {
      setAnnotationNavigationError('当前标注在确认期间发生了变化。请取消后重新切换。');
      return;
    }
    setAnnotationNavigationDecision('discard');
    setAnnotationNavigationError('');
    if (annotationAutoSaveTimerRef.current) window.clearTimeout(annotationAutoSaveTimerRef.current);
    annotationAutoSaveTimerRef.current = null;
    try {
      await deleteAnnotationDraft(pending.sourceDatasetId, pending.sourceAssetId);
      annotationDirtyRef.current = false;
      setAnnotationDirty(false);
      setPendingAnnotationNavigation(null);
      setAnnotationNavigationDecision(null);
      performFileSelection(pending.file, pending.datasetId);
    } catch (error) {
      setAnnotationNavigationDecision(null);
      setAnnotationNavigationError(error instanceof Error ? `无法清理本机草稿：${error.message}` : '无法清理本机草稿，已停留在当前图片。');
    }
  };

  const deleteShapeAtIndex = (index: number) => {
    const envelope = annotationDraftRef.current;
    const shape = envelope?.document.shapes?.[index];
    if (!envelope || !shape) return;
    const document = envelope.document;
    const shapes = [...(document.shapes ?? [])];
    shapes.splice(index, 1);
    commitAnnotationDocument({ ...document, shapes });
    setHiddenShapeIndexes((hidden) => remapHiddenShapesAfterDeletion(hidden, index));
    setSelectedShapeIndex((selected) => remapSelectedShapeAfterDeletion(selected, index));
    notify(`已删除标注框 #${index + 1} · ${shape.label}，可撤销`);
  };

  const deleteAnnotationCategory = (category: string) => {
    const envelope = annotationDraftRef.current;
    if (!envelope) return;
    const document = envelope.document;
    const shapes = document.shapes ?? [];
    const deletedIndexes = annotationIndexesForCategory(shapes.map((shape) => shape.label), category);
    if (!deletedIndexes.length) return;
    const deletedIndexSet = new Set(deletedIndexes);
    const nextShapes = shapes.filter((_shape, index) => !deletedIndexSet.has(index));
    commitAnnotationDocument({ ...document, shapes: nextShapes });
    setHiddenShapeIndexes((hidden) => remapHiddenShapesAfterDeletions(hidden, deletedIndexes));
    setSelectedShapeIndex((selected) => remapSelectedShapeAfterDeletions(selected, deletedIndexes));
    notify(`已删除当前图类别「${category}」及 ${deletedIndexes.length} 个标注框，可撤销`);
  };

  const setAnnotationCategoryColor = (category: string, rawColor: string) => {
    const color = normalizeAnnotationCategoryColor(rawColor);
    if (!color) return;
    setAnnotationCategoryColorOverrides((current) => ({ ...current, [normalizeAnnotationCategory(category)]: color }));
  };

  const deleteSelectedShape = () => {
    if (selectedShapeIndex !== null) deleteShapeAtIndex(selectedShapeIndex);
  };

  const updateShapeLabel = (shapeIndex: number, rawLabel: string) => {
    const label = normalizeAnnotationLabel(rawLabel);
    if (!label) { notify('标签必须为 1–128 个字符'); return false; }
    const document = annotationDraftRef.current?.document;
    if (!document) return false;
    const shapes = [...(document.shapes ?? [])];
    const shape = shapes[shapeIndex];
    if (!shape) return false;
    if (shape.label === label) return true;
    shapes[shapeIndex] = { ...shape, label };
    const committed = commitAnnotationDocument({ ...document, shapes });
    if (!committed) return false;
    setLastManualLabel(label);
    return true;
  };

  const closePendingLabelEditor = () => {
    setPendingShapeLabelEdit(null);
    setPendingCategoryLabelEdit(null);
    setManualShapeLabel('');
  };

  const openShapeLabelEditor = (shapeIndex: number) => {
    const envelope = annotationDraftRef.current;
    const shape = envelope?.document.shapes?.[shapeIndex];
    if (!envelope || !shape || annotationSaving) return;
    setPendingCategoryLabelEdit(null);
    setManualShapeLabel(shape.label);
    setPendingShapeLabelEdit({ index: shapeIndex, label: shape.label, shapeType: shape.shape_type, datasetId: envelope.dataset_id, assetId: envelope.asset_id });
  };

  const openSelectedShapeLabelEditor = () => {
    if (selectedShapeIndex !== null) openShapeLabelEditor(selectedShapeIndex);
  };

  const openCategoryLabelEditor = (category: string, count: number) => {
    const envelope = annotationDraftRef.current;
    if (!envelope || annotationSaving) return;
    setPendingShapeLabelEdit(null);
    setManualShapeLabel(category);
    setPendingCategoryLabelEdit({ category, count, datasetId: envelope.dataset_id, assetId: envelope.asset_id, idempotencyKey: crypto.randomUUID() });
  };

  const commitPendingShapeLabelEdit = (rawLabel = manualShapeLabel) => {
    const pending = pendingShapeLabelEdit;
    const envelope = annotationDraftRef.current;
    const label = normalizeAnnotationLabel(rawLabel);
    if (!label) {
      notify('标签类别必须为 1–128 个字符');
      return;
    }
    const currentShape = envelope?.document.shapes?.[pending?.index ?? -1];
    if (!pending || !envelope || envelope.dataset_id !== pending.datasetId || envelope.asset_id !== pending.assetId || !currentShape || currentShape.label !== pending.label) {
      closePendingLabelEditor();
      notify('编辑上下文已变化，未修改类别');
      return;
    }
    setSelectedShapeIndex(pending.index);
    updateShapeLabel(pending.index, label);
    closePendingLabelEditor();
  };

  const commitPendingCategoryLabelEdit = async (rawLabel = manualShapeLabel) => {
    if (categoryRenameCreating) return;
    const pending = pendingCategoryLabelEdit;
    const envelope = annotationDraftRef.current;
    const label = normalizeAnnotationLabel(rawLabel);
    if (!label) {
      notify('标签类别必须为 1–128 个字符');
      return;
    }
    if (!pending || !envelope || envelope.dataset_id !== pending.datasetId || envelope.asset_id !== pending.assetId) {
      closePendingLabelEditor();
      notify('当前图片已经切换，未修改类别');
      return;
    }
    if (backend.mode !== 'online' || !dataset.id || dataset.id !== pending.datasetId) {
      notify('全数据集改名需要连接真实本地服务');
      return;
    }
    if (backendJobs.some((job) => job.kind === 'category_rename' && job.dataset_id === pending.datasetId && ACTIVE_TASK_STATES.has(job.state))) {
      notify('当前数据集已有类别重命名任务在运行');
      return;
    }
    const sourceCategory = normalizeAnnotationCategory(pending.category);
    const targetLabel = label.normalize('NFC');
    const targetCategory = normalizeAnnotationCategory(targetLabel);
    if (targetCategory === sourceCategory) {
      closePendingLabelEditor();
      return;
    }
    setCategoryRenameCreating(true);
    try {
      if (documentIsDirty(envelope.document)) {
        const outcome = await persistCurrentAnnotation(true);
        if (!['remote', 'unchanged'].includes(outcome)) {
          notify('当前图标注未能先安全写入服务端，未创建全局改名任务');
          return;
        }
      }
      const job = await backend.createJob({
        kind: 'category_rename',
        dataset_id: pending.datasetId,
        concurrency: 2,
        source_category: sourceCategory,
        target_category: targetLabel,
      }, pending.idempotencyKey);
      setSelectedJobId(job.job_id);
      backend.watchJobEvents(job.job_id);
      setLastManualLabel(targetLabel);
      setAnnotationCategoryColorOverrides((current) => {
        if (!Object.hasOwn(current, sourceCategory) || Object.hasOwn(current, targetCategory)) return current;
        return { ...current, [targetCategory]: current[sourceCategory] };
      });
      closePendingLabelEditor();
      notify(`已创建全数据集类别重命名任务：${job.total} 张图片`);
    } catch (error) {
      notify(error instanceof Error ? error.message : '全数据集类别重命名任创建失败');
    } finally {
      setCategoryRenameCreating(false);
    }
  };

  const rotateSelectedShape = (deltaDegrees: number) => {
    if (selectedShapeIndex === null) return;
    const document = annotationDraftRef.current?.document;
    if (!document) return;
    const shapes = [...(document.shapes ?? [])];
    const shape = shapes[selectedShapeIndex];
    if (!shape || shape.shape_type !== 'rotation' || shape.points.length !== 4) return;
    shapes[selectedShapeIndex] = rotateRotationShape(shape, deltaDegrees * Math.PI / 180) as AnnotationShape;
    commitAnnotationDocument({ ...document, shapes });
  };

  const downloadAnnotationDocument = (documentValue: AnnotationEnvelope['document'], fileName: string) => {
    const blob = new Blob([JSON.stringify(documentValue, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const restoreRecoveredAnnotation = () => {
    if (!annotationRecovery || annotationRecovery.kind !== 'recoverable') return;
    const server = structuredClone(annotationRecovery.server);
    annotationDraftRef.current = server;
    setAnnotationDraft(server);
    resetAnnotationHistory(server.document);
    commitAnnotationDocument(annotationRecovery.local.document);
    setSelectedShapeIndex(null);
    setHiddenShapeIndexes(new Set());
    setAnnotationRecovery(null);
    notify(annotationAutoSave ? '本机草稿已恢复；将在 700ms 后自动保存' : '本机草稿已恢复；切图时将询问是否保存');
  };

  const discardRecoveredAnnotation = async () => {
    if (!annotationRecovery) return;
    try {
      await deleteAnnotationDraft(annotationRecovery.local.dataset_id, annotationRecovery.local.asset_id);
      setAnnotationRecovery(null);
      setAnnotationPersistence({ phase: 'saved', message: '本机草稿已丢弃，保留服务端版本' });
      notify('本机草稿已丢弃');
    } catch (error) {
      setAnnotationPersistence({ phase: 'error', message: error instanceof Error ? error.message : '草稿删除失败' });
    }
  };

  const exportConflictingAnnotation = () => {
    if (!annotationRecovery) return;
    const stem = currentFile?.name.replace(/\.[^.]+$/, '') ?? annotationRecovery.local.asset_id;
    downloadAnnotationDocument(annotationRecovery.local.document, `${stem}.conflict-draft.json`);
    notify('冲突草稿已导出；未覆盖服务端标注');
  };

  const importOperatorPackageFile = async (file: File | undefined) => {
    if (!file) return;
    if (backend.mode !== 'online') {
      notify('请先启动本地服务');
      return;
    }
    setOperatorImporting(true);
    setOperatorImportStatus(`正在检查算子包 ${file.name}…`);
    setOperatorCandidateFile(null);
    setOperatorInspection(null);
    try {
      const inspected = await inspectPipelineOperator(file);
      setOperatorCandidateFile(file);
      setOperatorInspection(inspected);
      setOperatorImportStatus(`检查通过：${inspected.operator.title} · 等待安装并合并`);
    } catch (error) {
      const message = error instanceof Error ? error.message : '算子 ZIP 检查失败';
      setOperatorImportStatus(message);
    } finally {
      setOperatorImporting(false);
    }
  };

  const confirmOperatorPackageImport = async () => {
    if (!operatorCandidateFile || !operatorInspection || operatorImporting) return;
    setOperatorImporting(true);
    setOperatorImportStatus(`正在安装并合并 ${operatorInspection.operator.title}…`);
    try {
      const installed = await importPipelineOperator(operatorCandidateFile);
      setOperatorImportStatus(`已安装并合并 ${installed.operator.title} · ${installed.digest.slice(0, 12)}`);
      setOperatorCandidateFile(null);
      setOperatorInspection(null);
    } catch (error) {
      setOperatorImportStatus(error instanceof Error ? error.message : '算子 ZIP 安装失败');
    } finally {
      setOperatorImporting(false);
      if (operatorPackageInputRef.current) operatorPackageInputRef.current.value = '';
    }
  };

  const cancelOperatorPackageImport = () => {
    setOperatorCandidateFile(null);
    setOperatorInspection(null);
    setOperatorImportStatus('');
    if (operatorPackageInputRef.current) operatorPackageInputRef.current.value = '';
  };

  const addNode = (kind: string, gap?: PipelineInsertionGap<FlowNode, VisualizationNode>) => {
    if (kind === 'model_feature' && pipelineHasModelFeature) {
      const message = '当前处理流已有模型中间层；每个处理流最多只能添加一个';
      setPipelineConstraintMessage(message);
      notify(message);
      return;
    }
    const operator = pipelinePalette.find((item) => item.kind === kind);
    if (!operator) return;
    const contract = pipelineContracts.find((item) => item.kind === kind);
    const id = `${kind}-${nextNodeId.current++}`;
    const parameters = parameterDefaults(contract);
    if (kind === 'model_feature' && selectedModel.id) {
      parameters.model_id = selectedModel.id;
      if (selectedLayer?.id) parameters.layer_id = selectedLayer.id;
      parameters.projection = featureTransformParameters.projection;
      parameters.normalization = featureTransformParameters.normalization;
      parameters.channel = featureChannel;
      parameters.clip = featureClip;
    }
    const node: FlowNode = {
      id,
      kind,
      name: contract?.title ?? operator.name,
      enabled: true,
      parameters,
      operator_version: contract?.version ?? 'pending',
    };
    const next = gap
      ? insertPipelineNodeAtGap(nodes, visualizations, node, gap)
      : (() => {
        const nextNodes = insertPipelineNode(nodes, node, nodes.length);
        return { nodes: nextNodes, visualizations: normalizeVisualizationTaps(nextNodes, visualizations) };
      })();
    const nextVisualizations = kind === 'model_feature'
      ? next.visualizations.map((item) => item.tap_after_node_id === id
        ? { ...item, parameters: { ...item.parameters, label: '中间层特征' } }
        : item)
      : next.visualizations;
    setNodes(next.nodes);
    setVisualizations(nextVisualizations);
    setPipelineConstraintMessage('');
    setSelectedNode(id);
    setPipelineInsertGapKey(null);
    if (kind === 'model_feature' && selectedModel.id) void loadModelLayers(selectedModel.id).catch(() => undefined);
    notify(`已添加 ${operator.name}`);
  };

  const deleteNode = (nodeId: string) => {
    const nextNodes = removePipelineNode(nodes, nodeId) as FlowNode[];
    if (nextNodes === nodes) return;
    setNodes(nextNodes);
    setVisualizations((old) => normalizeVisualizationTaps(nextNodes, old));
    setPipelineConstraintMessage('');
    if (selectedNode === nodeId) setSelectedNode('source');
  };

  const addVisualization = (tapAfterNodeId = nodes[nodes.length - 1]?.id ?? 'source') => {
    if (visualizations.length >= MAX_PIPELINE_VISUALIZATIONS) {
      const message = `最多添加 ${MAX_PIPELINE_VISUALIZATIONS} 个显示节点`;
      setPipelineConstraintMessage(message);
      notify(message);
      return;
    }
    if (visualizations.some((item) => item.tap_after_node_id === tapAfterNodeId)) {
      const message = '这个位置已经有显示节点；显示节点不能连续连接';
      setPipelineConstraintMessage(message);
      notify(message);
      return;
    }
    const contract = pipelineContracts.find((item) => item.kind === 'visualize');
    const sequence = nextVisualizationId.current++;
    const id = `visualize-${sequence}`;
    const next = normalizeVisualizationTaps(nodes, [...visualizations, {
      id,
      kind: 'visualize',
      name: contract?.title ?? '显示',
      enabled: true,
      parameters: contract ? parameterDefaults(contract) : { label: '显示' },
      operator_version: contract?.version ?? 'pending',
      tap_after_node_id: tapAfterNodeId,
    }]);
    setVisualizations(next);
    setPipelineConstraintMessage('');
    setSelectedNode(id);
    setPipelineInsertGapKey(null);
  };

  const deleteVisualization = (visualizationId: string) => {
    if (visualizations.length <= 1) {
      notify('至少保留一个显示节点');
      return;
    }
    const next = normalizeVisualizationTaps(nodes, visualizations.filter((item) => item.id !== visualizationId));
    setVisualizations(next);
    setPipelineConstraintMessage('');
    if (selectedNode === visualizationId) setSelectedNode(next[0]?.id ?? 'source');
  };

  const updateNodeParameter = (nodeId: string, name: string, value: unknown) => {
    setNodes((old) => old.map((node) => {
      if (node.id !== nodeId) return node;
      const parameters = { ...node.parameters };
      if (value === undefined) delete parameters[name];
      else parameters[name] = value;
      return { ...node, parameters };
    }));
    setVisualizations((old) => old.map((node) => {
      if (node.id !== nodeId) return node;
      const parameters = { ...node.parameters };
      if (value === undefined) delete parameters[name];
      else parameters[name] = value;
      return { ...node, parameters };
    }));
  };

  const runInference = async (remoteConfirmed = false) => {
    if (backend.mode === 'online') {
      const requestSignature = currentInferenceRequestSignature;
      if (!currentFile?.imagePath) {
        notify('请先打开真实数据集并选择可用图像');
        return;
      }
      if (selectedModel.availability !== 'available') {
        notify(`模型当前不可运行：${selectedModel.availability ?? '适配器未完成或权重缺失'}`);
        return;
      }
      if (!modelIsLoaded || backend.runtime.data?.model_id !== selectedModel.id || backend.runtime.data.state !== 'loaded') {
        notify('请先加载模型');
        return;
      }
      if (isSamModel && samPromptCount === 0) {
        notify('SAM 推理至少需要一个正点、负点或框选提示');
        return;
      }
      if (isTrustedRemoteModel && !remoteConfirmed) {
        setRemoteInferenceConfirmation({ action: 'current', model_id: selectedModel.id, dataset_id: dataset.id ?? null, asset_id: currentFile.id, fileName: currentFile.name });
        return;
      }
      try {
        const runtime = backend.runtime.data!;
        const captureLayer = captureFeatures ? runtime.layers.find((layer) => layer.id === selectedLayerId) : undefined;
        if (captureFeatures && !captureLayer) {
          notify('请先从已解析的 ONNX 图中明确选择一个中间层');
          return;
        }
        const result = await backend.runInference({
          model_id: selectedModel.id,
          image_path: currentFile.imagePath,
          capture_layers: captureLayer ? [captureLayer.id] : [],
          parameters: singleInferenceParameters,
        });
        if (!result || requestSignature !== currentInferenceSignatureRef.current) return;
        setCompletedInferenceSignature(requestSignature);
        notify(`真实推理完成：${result.annotations.length} 个对象，${result.classifications.length} 个分类结果，${result.artifacts.length} 个 tensor artifact`);
      } catch (error) {
        const message = error instanceof Error ? error.message : '真实推理失败';
        if (!/cancelled|aborted|superseded/i.test(message)) notify(message);
      }
      return;
    }
    notify('本地服务未连接，无法运行模型');
  };

  autoInferenceRunnerRef.current = () => { void runInference(); };
  useEffect(() => {
    const signature = currentInferenceRequestSignature;
    const runtimeReady = backend.runtime.data?.model_id === selectedModel.id && backend.runtime.data.state === 'loaded';
    if (!datasetWorkspaceReady || !signature || backend.mode !== 'online' || selectedModel.availability !== 'available' || !runtimeReady) return;
    if (isSamModel && samPromptCount === 0) return;
    if (captureFeatures && !selectedLayer) return;
    if (lastAutoInferenceSignatureRef.current === signature) return;
    const timer = window.setTimeout(() => {
      if (currentInferenceSignatureRef.current !== signature) return;
      lastAutoInferenceSignatureRef.current = signature;
      autoInferenceRunnerRef.current();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [backend.mode, backend.runtime.data?.model_id, backend.runtime.data?.state, captureFeatures, currentInferenceRequestSignature, datasetWorkspaceReady, isSamModel, samPromptCount, selectedLayer, selectedModel.availability, selectedModel.id]);

  const controlTask = async (jobId: string, action: 'pause' | 'resume' | 'cancel') => {
    setTaskActionPending(`${jobId}:${action}`);
    try {
      await backend.controlJob(jobId, action);
      notify(action === 'pause' ? '任务正在安全暂停' : action === 'resume' ? '任务已重新排队' : '任务正在取消，当前项结束后收敛');
    } catch (error) {
      notify(error instanceof Error ? error.message : '任务控制失败');
    } finally {
      setTaskActionPending(null);
    }
  };

  const promotePredictionToManual = (predictionIndex: number) => {
    const prediction = currentDetectionPredictions[predictionIndex];
    const envelope = annotationDraftRef.current;
    if (!prediction || !envelope) {
      notify('当前 AI 预测或人工标注草稿不可用');
      return;
    }
    const document = envelope.document;
    const nextIndex = document.shapes?.length ?? 0;
    const promoted: AnnotationShape = {
      ...structuredClone(prediction),
      label: prediction.label.trim() || '未命名',
      source: 'ai_promoted',
      model_id: selectedModel.id,
      prediction_index: predictionIndex,
      prediction_key: inferencePredictionKey(selectedModel.id, currentInferenceResult?.image_path ?? '', prediction),
      ai_original_score: prediction.score,
    };
    delete promoted.score;
    const committed = commitAnnotationDocument({ ...document, shapes: [...(document.shapes ?? []), promoted] });
    if (!committed) return;
    setSelectedPredictionIndex(null);
    setSelectedShapeIndex(nextIndex);
    setObjectSourceTab('manual');
    notify(`已将 AI 预测「${promoted.label}」转为人工框，可继续编辑`);
  };

  const persistDismissedTasks = (next: Record<string, string>) => {
    setDismissedTaskVersions(next);
    try { window.localStorage.setItem(DISMISSED_BACKGROUND_TASKS_KEY, JSON.stringify(next)); } catch { /* keep the in-memory view */ }
  };

  const dismissBackgroundTasks = (jobs: JobRecord[]) => {
    if (!jobs.length) return;
    const next = { ...dismissedTaskVersions, ...Object.fromEntries(jobs.map((job) => [job.job_id, job.updated_at])) };
    persistDismissedTasks(next);
    if (selectedJobId && jobs.some((job) => job.job_id === selectedJobId)) setSelectedJobId(null);
  };

  const clearCompletedBackgroundTasks = () => {
    const completed = taskStreamJobs.filter((job) => clearableTaskIds.includes(job.job_id));
    dismissBackgroundTasks(completed);
    notify(`已从最近任务隐藏 ${completed.length} 个完成项；服务端任务记录仍保留`);
  };

  const ignoreBackgroundTask = (job: JobRecord) => {
    dismissBackgroundTasks([job]);
    notify('已忽略该任务提醒；任务更新后会再次出现');
  };

  const retryBackgroundTask = (job: JobRecord) => {
    if (dismissedTaskVersions[job.job_id]) {
      const next = { ...dismissedTaskVersions };
      delete next[job.job_id];
      persistDismissedTasks(next);
    }
    void controlTask(job.job_id, 'resume');
  };

  const restoreHiddenBackgroundTasks = () => {
    persistDismissedTasks({});
    notify('已恢复本机隐藏的任务');
  };

  const changeTaskHistoryHours = (hours: BackgroundTaskHistoryHours) => {
    setTaskHistoryHours(hours);
    try { window.localStorage.setItem(BACKGROUND_TASK_HISTORY_HOURS_KEY, String(hours)); } catch { /* keep the in-memory choice */ }
  };

  const confirmRemoteInference = () => {
    const pending = remoteInferenceConfirmation;
    if (!pending) return;
    const current: RemoteInferenceConsentContext = { action: 'current', model_id: selectedModel.id, dataset_id: dataset.id ?? null, asset_id: currentFile?.id ?? null };
    setRemoteInferenceConfirmation(null);
    if (!isTrustedRemoteModel || !remoteInferenceConsentMatches(pending, current)) {
      notify('模型或数据集上下文已变化，远程推理未执行');
      return;
    }
    if (pending.action === 'current') void runInference(true);
  };

  const pickGlobalSettingsDirectory = async () => {
    if (backend.mode !== 'online' || modelDirectoryPicking) return;
    setModelDirectoryPicking('weights');
    setModelSettingsStatus('');
    try {
      const selected = await backend.pickDirectory({
        title: '选择模型权重下载目录',
        initial_dir: modelWeightsPathInput || undefined,
      });
      if (selected.canceled || !selected.path) return;
      setModelWeightsPathInput(selected.path);
    } catch (error) {
      setModelSettingsStatus(error instanceof Error ? error.message : '系统文件夹选择器不可用');
    } finally {
      setModelDirectoryPicking(null);
    }
  };

  const saveModelWeightsDirectory = async () => {
    const value = modelWeightsPathInput.trim();
    if (backend.mode !== 'online') {
      setModelSettingsStatus('需要连接真实本地服务后才能修改模型下载目录。');
      return;
    }
    if (!/^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(value)) {
      setModelSettingsStatus('请选择或输入一个绝对目录。');
      return;
    }
    setModelSettingsSaving(true);
    setModelSettingsStatus('');
    try {
      const result = await backend.updateApplicationSettings({ model_weights_dir: value });
      setModelWeightsPathInput(result.model_weights_dir);
      setModelSettingsStatus(result.restart_required ? '已保存；重启本地服务后生效，不会自动迁移旧权重。' : '模型下载目录已经生效。');
    } catch (error) {
      setModelSettingsStatus(error instanceof Error ? error.message : '模型下载目录保存失败');
    } finally {
      setModelSettingsSaving(false);
    }
  };

  const saveCloudAiSettings = async () => {
    if (backend.mode !== 'online') {
      setCloudAiStatus('需要连接真实本地服务后才能保存云端 AI 配置。');
      return;
    }
    setCloudAiSaving(true);
    setCloudAiStatus('');
    try {
      const result = await backend.updateApplicationSettings({ cloud_ai: cloudAiDraft });
      setCloudAiStatus(result.cloud_ai.enabled
        ? result.cloud_ai.credential_configured
          ? 'Agent 后端配置已保存；新任务会使用该模型规划受控工具。'
          : `配置已保存，但本地服务尚未检测到 ${result.cloud_ai.api_key_env}。`
        : 'AI 服务已关闭；Agent 将保持不可用。');
    } catch (error) {
      setCloudAiStatus(error instanceof Error ? error.message : '云端 AI 配置保存失败');
    } finally {
      setCloudAiSaving(false);
    }
  };

  const saveModelDownloadSource = async () => {
    if (backend.mode !== 'online') {
      setModelSettingsStatus('需要连接真实本地服务后才能保存模型下载源。');
      return;
    }
    setModelSettingsSaving(true);
    setModelSettingsStatus('');
    try {
      const result = await backend.updateApplicationSettings({ model_download_source: modelDownloadSource });
      setModelDownloadSource(result.model_download_source);
      setModelSettingsStatus('首选下载源已保存；新的权重列表会优先显示该来源。');
      if (selectedModel.id) await refreshModelWeights(selectedModel.id);
    } catch (error) {
      setModelSettingsStatus(error instanceof Error ? error.message : '模型下载源保存失败');
    } finally {
      setModelSettingsSaving(false);
    }
  };

  const saveWorkspaceDefaults = async () => {
    if (backend.mode !== 'online') {
      setWorkspaceSettingsStatus('需要连接本地服务后才能保存全局工作区默认值。');
      return;
    }
    setWorkspaceSettingsSaving(true);
    setWorkspaceSettingsStatus('');
    try {
      const inference = {
        model_id: selectedModel.id || null,
        provider: inferenceProvider,
        parameters: inferenceParameters,
      };
      await updateApplicationSettings({
        workspace: globalWorkspaceSettings(persistedPipelineSettings, inference),
      });
      globalInferenceFingerprintRef.current = JSON.stringify(inference);
      setWorkspaceSettingsStatus('当前处理流和推理模型已保存为全局默认值。');
    } catch (error) {
      setWorkspaceSettingsStatus(error instanceof Error ? error.message : '全局工作区默认值保存失败');
    } finally {
      setWorkspaceSettingsSaving(false);
    }
  };

  const saveNetworkProxy = async () => {
    if (backend.mode !== 'online') {
      setNetworkProxyStatus('需要连接真实本地服务后才能保存代理设置。');
      return;
    }
    setNetworkProxySaving(true);
    setNetworkProxyStatus('');
    try {
      const result = await backend.updateApplicationSettings({ network_proxy: networkProxyDraft });
      setNetworkProxyDraft(result.network_proxy);
      setNetworkProxyStatus(result.network_proxy_restart_required
        ? '代理设置已保存；重启本地服务后统一用于新的出站连接。'
        : '代理设置已保存并处于当前生效状态。');
    } catch (error) {
      setNetworkProxyStatus(error instanceof Error ? error.message : '代理设置保存失败');
    } finally {
      setNetworkProxySaving(false);
    }
  };

  const recordShortcut = (action: ShortcutActionId, event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (recordingShortcut !== action) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === 'Escape') {
      setRecordingShortcut(null);
      setShortcutFeedback('已取消快捷键录制');
      return;
    }
    const binding = shortcutFromKeyboardEvent(event.nativeEvent);
    if (!binding) return;
    if (isForbiddenShortcut(binding)) {
      setShortcutFeedback(`不能使用 ${displayShortcut(binding, useMacShortcutSymbols)}，它属于系统或固定交互键。`);
      return;
    }
    const conflict = findShortcutConflict(shortcuts, action, binding);
    if (conflict) {
      setShortcutFeedback(`${displayShortcut(binding, useMacShortcutSymbols)} 已用于“${conflict.label}”。`);
      return;
    }
    const defaults = defaultShortcutMap();
    setShortcutOverrides((current) => {
      const next = { ...current };
      if (binding === defaults[action]) delete next[action];
      else next[action] = binding;
      return next;
    });
    setRecordingShortcut(null);
    setShortcutFeedback(`“${shortcutDefinitions.find((definition) => definition.id === action)?.label ?? action}”已改为 ${displayShortcut(binding, useMacShortcutSymbols)}。`);
  };

  const resetShortcut = (action: ShortcutActionId) => {
    setShortcutOverrides((current) => {
      const next = { ...current };
      delete next[action];
      return next;
    });
    setRecordingShortcut(null);
    setShortcutFeedback('已恢复该项默认快捷键。');
  };

  const resetAllShortcuts = () => {
    setShortcutOverrides({});
    setRecordingShortcut(null);
    setShortcutFeedback('全部快捷键已恢复默认。');
  };

  const loadModelById = async (model: ModelView, operationId = modelSelectionOperationRef.current) => {
    if (backend.mode !== 'online' || !model.id) return;
    if (model.availability !== 'available') {
      setModelLoadError(`模型当前不可加载：${model.availability ?? '配置不可用'}`);
      return;
    }
    setModelLoadError('');
    try {
      const runtime = await backend.loadModel(model.id, [inferenceProvider]);
      if (!runtime || operationId !== modelSelectionOperationRef.current) return;
      setModelLoaded(runtime.state === 'loaded');
      setSelectedLayerId('');
      if (runtime.state === 'loaded') void recordModelUsage(model.id).catch(() => undefined);
      notify(runtime.layers.length
        ? `${model.name} 已加载 · 解析到 ${runtime.layers.length} 个可捕获层`
        : `${model.name} 已加载 · 没有可安全捕获的浮点特征层`);
    } catch (error) {
      if (operationId !== modelSelectionOperationRef.current) return;
      const message = error instanceof Error ? error.message : '模型加载失败';
      setModelLoadError(message);
      notify(message);
    }
  };

  const loadSelectedModel = async () => {
    const operationId = ++modelSelectionOperationRef.current;
    await loadModelById(selectedModel, operationId);
  };

  const downloadSelectedModel = async () => {
    if (backend.mode !== 'online' || !selectedModel.id || modelDownloadActionPending) return;
    setModelDownloadActionPending(true);
    autoLoadAfterDownloadRef.current = selectedModel.id;
    setModelLoadError('');
    try {
      const weights = await refreshModelWeights(selectedModel.id);
      const missing = weights?.filter((weight) => !weight.downloaded) ?? [];
      if (!missing.length) throw new Error('当前模型没有可下载的缺失权重');
      const jobs = [];
      for (const weight of missing) {
        const key = `${selectedModel.id}:${weight.url_index}`;
        weightDownloadPendingRef.current.add(key);
        const job = await backend.downloadModelWeight(selectedModel.id, weight.url_index);
        jobs.push(job);
        weightDownloadPendingRef.current.delete(key);
      }
      const first = jobs[0];
      if (first) {
        setSelectedJobId(first.job_id);
        backend.watchJobEvents(first.job_id);
      }
      void backend.refreshJobs();
      notify(`已创建 ${jobs.length} 个权重下载任务，可在右上角查看进度`);
    } catch (error) {
      autoLoadAfterDownloadRef.current = null;
      const message = error instanceof Error ? error.message : '模型下载失败';
      setModelLoadError(message);
      notify(message);
    } finally {
      for (const key of [...weightDownloadPendingRef.current]) {
        if (key.startsWith(`${selectedModel.id}:`)) weightDownloadPendingRef.current.delete(key);
      }
      setModelDownloadActionPending(false);
    }
  };

  const selectedModelDownloadJobs = backendJobs.filter((job) => job.kind === 'model_download' && job.request.model_id === selectedModel.id && ACTIVE_TASK_STATES.has(job.state));
  const selectedModelDownloadActive = selectedModelDownloadJobs.length > 0;
  const selectedModelDownloadJobIds = new Set(selectedModelDownloadJobs.map((job) => job.job_id));
  const selectedModelDownloadProgressItems = Object.values(backend.jobItemProgress).filter((item) => selectedModelDownloadJobIds.has(item.job_id));
  const selectedModelDownloadProgress = selectedModelDownloadActive && selectedModelDownloadProgressItems.length >= selectedModelDownloadJobs.length && selectedModelDownloadProgressItems.every((item) => item.total_bytes !== null && item.total_bytes > 0)
    ? Math.max(0, Math.min(100, Math.round(selectedModelDownloadProgressItems.reduce((sum, item) => sum + item.received_bytes, 0) / selectedModelDownloadProgressItems.reduce((sum, item) => sum + (item.total_bytes ?? 0), 0) * 100)))
    : null;

  useEffect(() => {
    if (autoLoadAfterDownloadRef.current !== selectedModel.id || selectedModel.availability !== 'available' || selectedModelDownloadActive) return;
    autoLoadAfterDownloadRef.current = null;
    closeModelPicker();
    const operationId = ++modelSelectionOperationRef.current;
    void loadModelById(selectedModel, operationId);
    // The selected model fields are the stable completion signal after catalog refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel.id, selectedModel.availability, selectedModelDownloadActive]);

  const inspectJob = async (jobId: string) => {
    setSelectedJobId(jobId);
    backend.watchJobEvents(jobId);
    try {
      await backend.loadJobItems(jobId);
    } catch (error) {
      notify(error instanceof Error ? error.message : '任务明细加载失败');
    }
  };

  const agentBackendReady = backend.mode === 'online' && backend.agentStatus.data?.state === 'ready';
  useEffect(() => {
    setAgentMessages([]);
    setAgentConfirmingProposal(null);
    setAgentExecutionErrors({});
  }, [dataset.id]);
  const sendAgent = async (
    text = agentInput,
    toolCall?: { tool: AgentToolName; arguments: Record<string, unknown> },
  ) => {
    const clean = text.trim();
    if (!clean || agentSending) return;
    const context = {
      label: `${dataset.name}${currentFile ? ` / ${currentFile.name}` : ''}`,
      datasetId: dataset.id ?? null,
      assetId: currentFile?.id ?? null,
    };
    if (!agentBackendReady) {
      notify(backend.mode !== 'online' ? '本地服务未连接，Agent 不可用' : backend.agentStatus.data?.message ?? 'Agent 后端尚未配置');
      return;
    }
    setAgentMessages((old) => [...old, { id: crypto.randomUUID(), role: 'user', text: clean, context }]);
    setAgentInput('');
    if (!dataset.id) {
      setAgentMessages((old) => [...old, { id: crypto.randomUUID(), role: 'agent', source: 'error', context, text: '请先打开真实数据集，再运行 Agent 任务。' }]);
      return;
    }
    setAgentSending(true);
    try {
      const result = await backend.runAgent({
        dataset_id: dataset.id,
        asset_id: currentFile?.id ?? null,
        message: clean,
        tool_call: toolCall,
      });
      setAgentMessages((old) => [...old, { id: result.run_id, role: 'agent', source: result.state === 'failed' ? 'error' : 'live', context, text: result.reply, run: result }]);
    } catch (error) {
      setAgentMessages((old) => [...old, { id: crypto.randomUUID(), role: 'agent', source: 'error', context, text: error instanceof Error ? error.message : '本地 Agent 请求失败' }]);
    } finally {
      setAgentSending(false);
    }
  };

  const executeAgentProposal = async (runId: string, proposalId: string) => {
    const key = `${runId}:${proposalId}`;
    if (!agentBackendReady) {
      setAgentExecutionErrors((old) => ({ ...old, [key]: backend.agentStatus.data?.message ?? 'Agent 后端不可用，不能执行提案。' }));
      return;
    }
    setAgentProposalPending(key);
    setAgentConfirmingProposal(null);
    setAgentExecutionErrors((old) => ({ ...old, [key]: '' }));
    try {
      const result = await backend.executeAgentProposal(runId, proposalId);
      setAgentMessages((old) => old.map((message) => message.run?.run_id === runId ? { ...message, text: result.reply, run: result } : message));
      const executedProposal = result.proposals.find((proposal) => proposal.id === proposalId && proposal.executed && proposal.result);
      const action = executedProposal?.result?.action;
      if (action === 'ui.open_dataset') {
        void pickImageDataset();
      } else if (action === 'ui.import_operator') {
        setSettingsSection('operators');
        setSettingsOpen(true);
        window.setTimeout(() => operatorPackageInputRef.current?.click(), 0);
      } else if (action === 'ui.open_models') {
        setRightTab('inference');
        setModelPickerOpen(true);
        if (backend.mode === 'online') void Promise.all([refreshBackendModels(), refreshBackendHealth()]);
      } else if (action === 'pipeline.draft') {
        const rawNodes = executedProposal?.result?.pipeline_nodes;
        if (!Array.isArray(rawNodes)) throw new Error('Agent 返回的处理流草案无效');
        const transforms: FlowNode[] = rawNodes.flatMap((value, index) => {
          if (!value || typeof value !== 'object') return [];
          const node = value as Record<string, unknown>;
          const kind = typeof node.kind === 'string' ? node.kind : '';
          if (!kind || ['source', 'output', 'visualize'].includes(kind)) return [];
          const contract = pipelineContracts.find((candidate) => candidate.kind === kind);
          if (!contract) return [];
          return [{
            id: `agent-${index + 1}-${kind}-${Date.now().toString(36)}`,
            kind,
            name: contract.title,
            enabled: node.enabled !== false,
            parameters: node.parameters && typeof node.parameters === 'object' && !Array.isArray(node.parameters) ? node.parameters as Record<string, unknown> : {},
            operator_version: contract.version,
          }];
        });
        if (!transforms.length) throw new Error('Agent 草案没有可用的已注册图像算子');
        const nextNodes = [{ ...initialNodes[0] }, ...transforms];
        setNodes(nextNodes);
        setVisualizations((old) => normalizeVisualizationTaps(nextNodes, old));
        setPipelineEnabled(true);
        setSelectedNode(transforms[0].id);
        setRightTab('pipeline');
        notify(`已载入 Agent 处理流草案：${transforms.length} 个算子`);
      }
      void backend.refreshJobs().catch(() => undefined);
      notify(result.state === 'completed' ? '真实 Agent 操作已完成' : 'Agent 已返回更新后的受控提案');
    } catch (error) {
      setAgentExecutionErrors((old) => ({ ...old, [key]: error instanceof Error ? error.message : 'Agent 提案执行失败' }));
    } finally {
      setAgentProposalPending(null);
    }
  };

  const imageUnitsPerScreenPixel = () => {
    const sourceWidth = displayedWidth ?? currentFile?.width ?? 1;
    const imageSurfaceWidth = imageRef.current?.clientWidth;
    if (imageSurfaceWidth && imageSurfaceWidth > 0) return sourceWidth / (imageSurfaceWidth * Math.max(view.scale, 0.01));
    return sourceWidth / (960 * Math.max(view.scale, 0.01));
  };

  const renderPendingManualShape = (shape: AnnotationShape, categoryStyle: React.CSSProperties) => {
    const className = `draw-preview pending-label ${annotationShapeClass(shape.shape_type)}`;
    const screenUnit = imageUnitsPerScreenPixel();
    const points = shape.points.map((point) => point.join(',')).join(' ');
    if (shape.shape_type === 'point') return <circle className={className} cx={shape.points[0]?.[0]} cy={shape.points[0]?.[1]} r={6 * screenUnit} style={categoryStyle} />;
    if (shape.shape_type === 'line' || shape.shape_type === 'linestrip') return <polyline className={className} points={points} fill="none" style={categoryStyle} />;
    if (shape.shape_type === 'circle' && shape.points.length >= 2) {
      const [center, edge] = shape.points;
      return <circle className={className} cx={center[0]} cy={center[1]} r={Math.hypot(edge[0] - center[0], edge[1] - center[1])} style={categoryStyle} />;
    }
    return <polygon className={className} points={points} style={categoryStyle} />;
  };

  const renderRealShape = (shape: AnnotationShape, index: number, prediction = false, layer: 'geometry' | 'label' | 'controls' = 'geometry') => {
    const typeClass = annotationShapeClass(shape.shape_type);
    const categoryStyle = annotationCategoryStyle(shape.label, annotationCategoryColorOverrides);
    const selected = prediction ? selectedPredictionIndex === index : selectedShapeIndex === index;
    const labelOpacity = canvasLabelOpacity(view.scale, selected);
    const className = `real-shape ${prediction ? 'prediction' : ''} ${selected ? 'selected' : ''}`;
    const shapeTitle = `${shape.label}${typeof shape.score === 'number' ? ` · ${(shape.score * 100).toFixed(1)}%` : ''} · ${shape.shape_type}`;
    const drawPoints = shape.shape_type === 'rectangle' && shape.points.length === 2
      ? [[shape.points[0][0], shape.points[0][1]], [shape.points[1][0], shape.points[0][1]], [shape.points[1][0], shape.points[1][1]], [shape.points[0][0], shape.points[1][1]]]
      : shape.points;
    const points = drawPoints.map((point) => point.join(',')).join(' ');
    const fullLabelText = `${shape.label || '未命名'}${typeof shape.score === 'number' ? ` · ${(shape.score * 100).toFixed(0)}%` : ''}`;
    const canvasLabelText = fitCanvasLabelText(fullLabelText);
    const layoutKey = `${prediction ? 'pred' : 'gt'}-${index}`;
    const labelLayout = realCanvasLabelLayouts[layoutKey] ?? null;
    let body: React.ReactNode;
    if (shape.shape_type === 'point') {
      body = <circle className={className} cx={shape.points[0]?.[0]} cy={shape.points[0]?.[1]} r={Math.max(3, (currentFile?.width ?? 1000) / 400)}><title>{shapeTitle}</title></circle>;
    } else if (shape.shape_type === 'line' || shape.shape_type === 'linestrip') {
      body = <polyline className={className} points={points} fill="none"><title>{shapeTitle}</title></polyline>;
    } else if (shape.shape_type === 'circle' && shape.points.length >= 2) {
      const [center, edge] = shape.points;
      const radius = Math.hypot(edge[0] - center[0], edge[1] - center[1]);
      body = <circle className={className} cx={center[0]} cy={center[1]} r={radius}><title>{shapeTitle}</title></circle>;
    } else {
      body = <polygon className={className} points={points}><title>{shapeTitle}</title></polygon>;
    }
    const showControlPoints = !prediction && selectedShapeIndex === index && (tool === 'select' || (shape.shape_type === 'rotation' && tool === 'rotation')) && !samPromptMode;
    const showRotationCorner = !prediction && selectedShapeIndex === index && shape.shape_type === 'rotation' && ['select', 'rotation'].includes(tool) && !samPromptMode;
    const screenUnit = showControlPoints || showRotationCorner ? imageUnitsPerScreenPixel() : 1;
    const rotationCornerPoint = showRotationCorner ? rotationCornerHandle(shape.points) : null;
    const controlPointIndexes = showControlPoints ? editableControlPointIndexes(shape) : [];
    const multiVertexPath = showControlPoints && ['polygon', 'linestrip'].includes(shape.shape_type)
      ? polygonVertexControlPath(shape.points, Math.max(0.001, screenUnit * 0.001))
      : '';
    const rotationHandlers = {
      onPointerDown: (event: React.PointerEvent<SVGElement>) => startRealRotation(index, event),
      onPointerMove: moveRealRotation,
      onPointerUp: endRealRotation,
      onPointerCancel: endRealRotation,
    };
    if (layer === 'geometry') return <g key={`geometry-${layoutKey}`} className={typeClass} style={categoryStyle} data-shape-index={index}>{body}</g>;
    if (layer === 'label') return labelLayout && labelOpacity > 0 ? <g key={`label-${layoutKey}`} className={`shape-canvas-label ${prediction ? 'prediction' : ''} ${selected ? 'selected' : ''}`} style={{ ...categoryStyle, opacity: labelOpacity }} data-shape-index={index} data-placement={labelLayout.placement} transform={`translate(${labelLayout.x} ${labelLayout.y}) scale(${labelLayout.unit})`} aria-hidden="true"><title>{fullLabelText}</title><rect x={0} y={0} width={labelLayout.width / labelLayout.unit} height={labelLayout.height / labelLayout.unit} rx={labelLayout.radius / labelLayout.unit} /><text x={labelLayout.paddingX / labelLayout.unit} y={labelLayout.height / labelLayout.unit / 2} dy="0.34em" style={{ fontSize: labelLayout.fontSize / labelLayout.unit }}>{canvasLabelText}</text></g> : null;
    return <g key={`controls-${layoutKey}`} className={typeClass} style={categoryStyle}>
      {showControlPoints && ['polygon', 'linestrip'].includes(shape.shape_type) && <path className="shape-control-vertices" d={multiVertexPath} onPointerDown={(event) => startNearestMultiPointMove(index, event)} onPointerMove={moveControlPoint} onPointerUp={endControlPointMove} onPointerCancel={endControlPointMove}><title>拖动任一顶点</title></path>}
      {showControlPoints && !['polygon', 'linestrip'].includes(shape.shape_type) && controlPointIndexes.map((pointIndex) => { const point = shape.points[pointIndex]; const resizeCursor = resolveResizeCursor(shape.points, pointIndex); return point ? <circle key={`handle-${pointIndex}`} className={`shape-control-point resize-${resizeCursor}`} cx={point[0]} cy={point[1]} r={CANVAS_CONTROL_POINT_RADIUS_PX * screenUnit} onPointerDown={(event) => startControlPointMove(index, pointIndex, event)} onPointerMove={moveControlPoint} onPointerUp={endControlPointMove} onPointerCancel={endControlPointMove} /> : null; })}
      {rotationCornerPoint && <circle className="rotation-corner-handle" cx={rotationCornerPoint[0]} cy={rotationCornerPoint[1]} r={CANVAS_CONTROL_POINT_RADIUS_PX * screenUnit} {...rotationHandlers}><title>拖动右上角旋转 · 按住 Shift 以 15° 吸附</title></circle>}
    </g>;
  };

  const readPipelinePaneMetrics = (item: Pick<PipelineVisualizationResult, 'width' | 'height'>, pixelGridEnabled = showPixel) => {
    const stageWidth = stageRef.current?.clientWidth ?? 840;
    const stageHeight = stageRef.current?.clientHeight ?? 592;
    const paneCount = Math.max(1, pipelineDisplaySlots.length);
    const fallbackWidth = paneCount > 1 ? stageWidth / 2 : stageWidth;
    const fallbackHeight = paneCount > 2 ? stageHeight / 2 : stageHeight;
    const paneWidth = imageRef.current?.clientWidth ?? fallbackWidth;
    const paneHeight = imageRef.current?.clientHeight ?? fallbackHeight;
    return pipelinePaneMetrics(paneWidth, paneHeight, item.width, item.height, view, stageWidth, stageHeight, pixelGridEnabled);
  };

  const renderPipelinePreviewAnnotationLayer = (item: PipelineVisualizationResult, screenAligned = false) => {
    const shapes = item.annotation_document.shapes ?? [];
    const metrics = readPipelinePaneMetrics(item, false);
    const labelEntries = showGT ? shapes.flatMap((shape, index) => {
      const selected = selectedShapeIndex === index;
      const labelOpacity = canvasLabelOpacity(view.scale, selected);
      return labelOpacity <= 0 ? [] : [{ shape, index, selected, opacity: labelOpacity }];
    }) : [];
    const labelLayouts = canvasLabelLayouts(
      labelEntries.map(({ shape, selected }) => ({
        points: canvasLabelPointsForShape(shape),
        text: fitCanvasLabelText(shape.label || '未命名'),
        priority: selected ? 2 : 1,
        anchor: canvasLabelAnchorForShape(shape),
      })),
      item.width,
      item.height,
      metrics.imageUnitsPerScreenPixel,
      metrics.visibleBounds ?? undefined,
    );
    const layerStyle = screenAligned ? {
      opacity: opacity / 100,
      left: metrics.display.left,
      top: metrics.display.top,
      right: 'auto',
      bottom: 'auto',
      width: metrics.display.width,
      height: metrics.display.height,
    } : { opacity: opacity / 100 };
    return <svg className={`pipeline-preview-annotation-layer ${screenAligned ? 'screen-aligned' : ''}`} viewBox={`0 0 ${item.width} ${item.height}`} preserveAspectRatio="xMidYMid meet" aria-label={`${item.label} 的处理后标注`} style={layerStyle}>
      {showGT && shapes.map((shape, index) => {
        const drawPoints = shape.shape_type === 'rectangle' && shape.points.length === 2
          ? [[shape.points[0][0], shape.points[0][1]], [shape.points[1][0], shape.points[0][1]], [shape.points[1][0], shape.points[1][1]], [shape.points[0][0], shape.points[1][1]]]
          : shape.points;
        const points = drawPoints.map((point) => point.join(',')).join(' ');
        const className = `real-shape ${selectedShapeIndex === index ? 'selected' : ''}`;
        const categoryStyle = annotationCategoryStyle(shape.label, annotationCategoryColorOverrides);
        let body: React.ReactNode;
        if (shape.shape_type === 'point') {
          body = <circle className={className} cx={shape.points[0]?.[0]} cy={shape.points[0]?.[1]} r={Math.max(3, Math.min(item.width, item.height) / 300)} />;
        } else if (shape.shape_type === 'line' || shape.shape_type === 'linestrip') {
          body = <polyline className={className} points={points} fill="none" />;
        } else if (shape.shape_type === 'circle' && shape.points.length >= 2) {
          const [center, edge] = shape.points;
          body = <circle className={className} cx={center[0]} cy={center[1]} r={Math.hypot(edge[0] - center[0], edge[1] - center[1])} />;
        } else {
          body = <polygon className={className} points={points} />;
        }
        return <g key={`${item.visualization_id}-annotation-${index}`} className={annotationShapeClass(shape.shape_type)} style={categoryStyle} data-shape-index={index}>{body}<title>{shape.label || '未命名'} · {shape.shape_type}</title></g>;
      })}
      {labelEntries.map((entry, index) => {
        const layout = labelLayouts[index];
        if (!layout) return null;
        const text = fitCanvasLabelText(entry.shape.label || '未命名');
        return <g key={`${item.visualization_id}-label-${entry.index}`} className={`shape-canvas-label ${entry.selected ? 'selected' : ''}`} style={{ ...annotationCategoryStyle(entry.shape.label, annotationCategoryColorOverrides), opacity: entry.opacity }} data-shape-index={entry.index} data-placement={layout.placement} transform={`translate(${layout.x} ${layout.y}) scale(${layout.unit})`} aria-hidden="true"><title>{entry.shape.label || '未命名'}</title><rect x={0} y={0} width={layout.width / layout.unit} height={layout.height / layout.unit} rx={layout.radius / layout.unit} /><text x={layout.paddingX / layout.unit} y={layout.height / layout.unit / 2} dy="0.34em" style={{ fontSize: layout.fontSize / layout.unit }}>{text}</text></g>;
      })}
    </svg>;
  };

  const renderPipelineSharedCrosshair = (item: Pick<PipelineVisualizationResult, 'visualization_id' | 'width' | 'height' | 'coordinate_mapping'>, screenAligned = false) => {
    if (!pipelineSharedCursor) return null;
    const inferred = item.coordinate_mapping ? null : resolvePipelineCoordinateTransform(
      pipelineRequestNodes,
      currentFile?.width ?? 0,
      currentFile?.height ?? 0,
      item.visualization_id,
    );
    const mapping = item.coordinate_mapping ?? pipelineCoordinateMappingFromTransform(inferred, currentFile?.width ?? 0, currentFile?.height ?? 0);
    const point = pipelineSharedCursorPointForPane(pipelineSharedCursor, item.visualization_id, item.width, item.height, mapping);
    if (!point) return null;
    const metrics = screenAligned ? readPipelinePaneMetrics(item, false) : null;
    const style = metrics ? { left: metrics.display.left, top: metrics.display.top, right: 'auto', bottom: 'auto', width: metrics.display.width, height: metrics.display.height } : undefined;
    return <svg className={`pipeline-shared-cursor ${screenAligned ? 'screen-aligned' : ''}`} aria-hidden="true" viewBox={`0 0 ${item.width} ${item.height}`} preserveAspectRatio="xMidYMid meet" style={style}><line className="horizontal" x1="0" x2={item.width} y1={point.y} y2={point.y} /><line className="vertical" x1={point.x} x2={point.x} y1="0" y2={item.height} /><circle cx={point.x} cy={point.y} r={Math.max(2, Math.min(item.width, item.height) * .006)} /></svg>;
  };

  const activeFileIndex = validFiles.findIndex((file) => file.id === currentFile?.id);
  const activeIndex = activeFileIndex >= 0 ? activeFileIndex + 1 : 0;
  const canStepPrevious = activeFileIndex > 0;
  const canStepNext = activeFileIndex >= 0 && (
    activeFileIndex < validFiles.length - 1
    || (isRealDataset && backend.mode === 'online' && !backend.assetSearchLoadingMore && backend.assetSearch.data.next_cursor != null)
  );
  const localServiceDown = backend.mode === 'offline';
  const backendLabel = backend.mode === 'online'
    ? `本地服务 v${backend.health.data?.version ?? '0.1'} · ${backend.models.data.models.length} 模型`
    : backend.mode === 'probing'
      ? '正在连接本地服务'
      : '本地服务离线';
  const scanSummary = backend.mode === 'online' && backend.scan.data
    ? backend.scan.data.summary ?? backend.scan.data.streamed_summary
    : { valid: 0, hidden_image_only: 0, duplicate_match: 0, orphan_annotation: 0, corrupt_image: 0, corrupt_annotation: 0 };
  const scanSessionState = backend.scan.data?.state;
  const scanIsLoading = backend.mode === 'online' && (backend.scan.phase === 'loading' || scanSessionState === 'queued' || scanSessionState === 'running');
  const scanReady = backend.mode === 'online' && scanSessionState === 'succeeded' && backend.scan.phase === 'ready';
  const modelIsLoaded = backend.mode === 'online'
    ? selectedModel.runtimeState === 'loaded' || (backend.runtime.data?.model_id === selectedModel.id && backend.runtime.data.state === 'loaded')
    : modelLoaded;
  const modelActionKind = selectedModel.availability === 'missing_weights' ? 'download' : modelLoadError ? 'retry' : modelIsLoaded ? 'loaded' : 'load';
  const modelActionLabel = modelDownloadActionPending ? '准备下载…' : selectedModelDownloadActive ? selectedModelDownloadProgress === null ? '下载中…' : `${selectedModelDownloadProgress}%` : backend.runtime.phase === 'loading' ? '加载中…' : modelActionKind === 'download' ? '下载' : modelActionKind === 'retry' ? '重试' : modelActionKind === 'loaded' ? '✓ 已加载' : '加载';
  const realImageUrl = dataset.id && currentFile?.imagePath ? backend.assetUrl(dataset.id, currentFile.id, 'image') : null;
  const draftShapes = annotationDraft?.document.shapes ?? [];
  const knownAnnotationLabelValues = [
    ...draftShapes.map((shape) => shape.label),
    ...(isRealDataset ? realFiles : dataset.files).flatMap((file) => file.labels ?? []),
  ];
  const manualLabelChoices = buildAnnotationLabelChoices(knownAnnotationLabelValues, lastManualLabel);
  const manualShapeLabelValid = normalizeAnnotationLabel(manualShapeLabel);
  const manualShapeLabelKnown = Boolean(manualShapeLabelValid && manualLabelChoices.includes(manualShapeLabelValid));
  const freehandGuideLabel = lastManualLabel || manualLabelChoices[0] || 'object';
  const freehandGuideStyle = annotationCategoryStyle(freehandGuideLabel, annotationCategoryColorOverrides);
  const pendingManualShapeStyle = annotationCategoryStyle(manualShapeLabelValid || lastManualLabel || manualLabelChoices[0] || 'object', annotationCategoryColorOverrides);
  const manualLabelMenuPosition = pendingManualShape && typeof window !== 'undefined'
    ? positionFloatingLabelMenu(
      pendingManualShape.anchor,
      {
        width: manualLabelMenuGeometry.viewportWidth || window.innerWidth,
        height: manualLabelMenuGeometry.viewportHeight || window.innerHeight,
      },
      { width: manualLabelMenuGeometry.width, height: manualLabelMenuGeometry.height },
    )
    : null;
  const annotationObjects = draftShapes;
  const normalizedAnnotationObjectLabels = annotationObjects.map((shape) => normalizeAnnotationCategory(shape.label));
  const annotationCategoryCounts = normalizedAnnotationObjectLabels.reduce((counts, label) => counts.set(label, (counts.get(label) ?? 0) + 1), new Map<string, number>());
  const annotationCategories = [...annotationCategoryCounts].sort(([left, leftCount], [right, rightCount]) => rightCount - leftCount || left.localeCompare(right, 'zh-Hans-CN')).map(([label]) => label);
  const directInferenceMatchesCurrent = Boolean(
    currentFile?.imagePath
    && backend.inference.data?.image_path === currentFile.imagePath
    && backend.inference.data.model_id === selectedModel.id
    && completedInferenceSignature
    && completedInferenceSignature === currentInferenceRequestSignature
  );
  const currentBatchInferenceItem = currentBatchInferenceJob && currentFile?.id ? backend.jobItemSnapshots[`${currentBatchInferenceJob.job_id}:${currentFile.id}`] : undefined;
  const currentBatchInferenceResult = currentBatchInferenceItem?.state === 'succeeded' ? inferenceResultFromJobItem(currentBatchInferenceItem.result) : null;
  const batchInferenceMatchesCurrent = Boolean(currentBatchInferenceResult && currentFile?.imagePath && currentBatchInferenceResult.image_path === currentFile.imagePath && currentBatchInferenceResult.model_id === selectedModel.id);
  const currentInferenceResult = directInferenceMatchesCurrent ? backend.inference.data : batchInferenceMatchesCurrent ? currentBatchInferenceResult : null;
  const inferenceMatchesCurrent = Boolean(currentInferenceResult);
  const currentPredictions = currentInferenceResult?.annotations ?? [];
  const currentAnnotationsAreSegmentation = inferenceAnnotationsAreSegmentation(selectedModel.task, selectedModel.adapter);
  const currentDetectionPredictions = currentAnnotationsAreSegmentation ? [] : currentPredictions;
  const currentSegmentationContours = currentAnnotationsAreSegmentation ? currentPredictions : [];
  const promotedPredictionKeys = new Set(draftShapes.flatMap((shape) => shape.source === 'ai_promoted' && shape.model_id === selectedModel.id && typeof shape.prediction_key === 'string' ? [shape.prediction_key] : []));
  const availablePredictionEntries = currentDetectionPredictions.flatMap((prediction, index) => {
    const key = inferencePredictionKey(selectedModel.id, currentInferenceResult?.image_path ?? '', prediction);
    return promotedPredictionKeys.has(key) ? [] : [{ prediction, index, key, label: prediction.label.trim() || '未命名预测' }];
  });
  const aiPredictionCategories = groupInferencePredictionsByCategory(availablePredictionEntries.map((entry) => entry.prediction));
  const aiPredictionCategoryRows = aiPredictionCategories.map((category) => {
    const entries = availablePredictionEntries.filter((entry) => entry.label === category.label);
    const visibleCount = hiddenPredictionCategories.has(category.label) ? 0 : entries.filter((entry) => !hiddenPredictionKeys.has(entry.key)).length;
    return { ...category, entries, visibleCount };
  });
  const aiPredictionCategoryKey = `${currentFile?.id ?? ''}:${selectedModel.id}:${availablePredictionEntries.map((entry) => entry.key).join('\u001f')}`;
  useEffect(() => {
    setHiddenPredictionCategories(new Set());
    setHiddenPredictionKeys(new Set());
    setSelectedPredictionIndex(null);
    if (aiPredictionCategories.length === 0) setObjectSourceTab('manual');
  }, [aiPredictionCategories.length, aiPredictionCategoryKey]);
  const visibleCurrentPredictionEntries = availablePredictionEntries.filter((entry) => !hiddenPredictionCategories.has(entry.label) && !hiddenPredictionKeys.has(entry.key));
  const allAiPredictionObjectsVisible = availablePredictionEntries.length > 0 && visibleCurrentPredictionEntries.length === availablePredictionEntries.length;
  const visibleCurrentPredictions = currentAnnotationsAreSegmentation
    ? showMasks ? currentSegmentationContours : []
    : visibleCurrentPredictionEntries.map((entry) => entry.prediction);
  const visiblePredictionCanvasEntries = currentAnnotationsAreSegmentation
    ? visibleCurrentPredictions.map((prediction, index) => ({ prediction, index }))
    : visibleCurrentPredictionEntries;
  const currentClassifications = currentInferenceResult?.classifications ?? [];
  const currentArtifacts = currentInferenceResult?.artifacts ?? [];
  const selectedFeatureArtifact = selectedLayer
    ? currentArtifacts.find((artifact) => artifact.layer_id === selectedLayer.id)
    : undefined;
  const selectedFeaturePreviewUrl = selectedFeatureArtifact?.preview_available
    ? backend.artifactPreviewUrl(selectedFeatureArtifact.id)
    : null;
  const currentRasters = currentInferenceResult?.rasters ?? [];
  const showInferenceResult = backend.mode === 'online' && inferenceMatchesCurrent && (currentAnnotationsAreSegmentation || currentClassifications.length > 0 || currentArtifacts.length > 0 || currentRasters.length > 0);
  const pipelineBelongsToCurrentAsset = Boolean(
    backend.pipeline.data
    && backend.pipeline.data.dataset_id === dataset.id
    && backend.pipeline.data.asset_id === currentFile?.id,
  );
  const pipelineMatchesCurrent = pipelineBelongsToCurrentAsset
    && completedPipelineSignature === currentPipelineExecutionSignature;
  const pipelineVisualizationResults = useMemo<PipelineVisualizationResult[]>(() => {
    const result = backend.pipeline.data;
    if (!result) return [];
    if (result.visualizations?.length) return result.visualizations.slice(0, MAX_PIPELINE_VISUALIZATIONS);
    return [{
      visualization_id: visualizations[0]?.id ?? 'legacy-visualization',
      label: String(visualizations[0]?.parameters.label ?? '显示'),
      artifact_id: result.artifact_id,
      width: result.width,
      height: result.height,
      media_type: result.media_type,
      annotation_document: result.annotation_document,
      operator_timings_ms: result.operator_timings_ms,
    }];
  }, [backend.pipeline.data, visualizations]);
  const pipelinePreviewItems = useMemo(() => pipelineBelongsToCurrentAsset
    ? pipelineVisualizationResults.map((item) => ({ ...item, url: pipelineArtifactUrl(item.artifact_id) })).filter((item): item is PipelineVisualizationResult & { url: string } => Boolean(item.url))
    : [], [pipelineArtifactUrl, pipelineBelongsToCurrentAsset, pipelineVisualizationResults]);
  const pipelineDisplaySlots = useMemo(
    () => stablePipelineDisplaySlots(visualizations, pipelinePreviewItems, MAX_PIPELINE_VISUALIZATIONS),
    [pipelinePreviewItems, visualizations],
  );
  const primaryPipelineVisualization = pipelinePreviewItems[0];
  const pipelineImageUrl = primaryPipelineVisualization?.url ?? null;
  const pipelineDisplayItems = useMemo(() => pipelineDisplaySlots.map((slot) => {
    const item = slot.result;
    const attempt = item ? pipelineImageAttempts[item.url] ?? 0 : 0;
    return {
      ...slot,
      width: item?.width ?? currentFile?.width ?? 1,
      height: item?.height ?? currentFile?.height ?? 1,
      overlay_compatible: item?.overlay_compatible ?? true,
      coordinate_mapping: item?.coordinate_mapping,
      coordinate_space_id: item?.coordinate_mapping?.coordinate_space_id,
      url: item?.url ?? null,
      displayUrl: item && !pipelineImageRetryExhausted(attempt)
        ? pipelineArtifactDisplayUrl(item.url, pipelineImageEpoch, attempt)
        : null,
    };
  }), [currentFile?.height, currentFile?.width, pipelineDisplaySlots, pipelineImageAttempts, pipelineImageEpoch]);
  const pipelineCoordinateMappingForItem = (item: (typeof pipelineDisplayItems)[number]): PipelineCoordinateMappingLike | null => {
    if (item.coordinate_mapping) return item.coordinate_mapping;
    const inferred = resolvePipelineCoordinateTransform(
      pipelineRequestNodes,
      currentFile?.width ?? 0,
      currentFile?.height ?? 0,
      item.visualization_id,
    );
    return pipelineCoordinateMappingFromTransform(inferred, currentFile?.width ?? 0, currentFile?.height ?? 0);
  };
  const topVisiblePipelineItem = [...pipelineDisplayItems].reverse().find((item) => (visualizationLayerState[item.visualization_id]?.visible ?? true)) ?? pipelineDisplayItems[0];
  const singlePipelineSourceOptions = useMemo(() => [
    { value: 'source', label: '原图' },
    ...pipelineDisplayItems.map((item, index) => ({ value: item.visualization_id, label: `D${index + 1} · ${item.label}${item.result ? '' : ' · 计算中'}` })),
  ], [pipelineDisplayItems]);
  const selectedSinglePipelineItem = singlePipelineSource === 'source'
    ? null
    : pipelineDisplayItems.find((item) => item.visualization_id === singlePipelineSource) ?? null;
  const effectivePipelineImageUrl = pipelineDisplayItems[0]?.displayUrl ?? null;
  const selectedOperatorInputTransform = selectedOperator ? resolvePipelineCoordinateTransform(
    pipelineRequestNodes,
    currentFile?.width ?? 0,
    currentFile?.height ?? 0,
    selectedOperator.id,
  ) : null;
  const activePipelineCoordinateTransform = resolvePipelineCoordinateTransform(
    pipelineRequestNodes,
    currentFile?.width ?? 0,
    currentFile?.height ?? 0,
    primaryPipelineVisualization?.visualization_id,
  );
  const livePipelineAnnotationShapes = activePipelineCoordinateTransform?.topologySafe
    ? draftShapes.map((shape) => transformCanvasShape(shape, activePipelineCoordinateTransform))
    : primaryPipelineVisualization?.annotation_document.shapes;
  const canvasPresentation = resolveCanvasPresentation({
    sourceImageUrl: realImageUrl,
    pipelineImageUrl: visualizationDisplayMode === 'source' ? null : effectivePipelineImageUrl,
    pipelineEnabled,
    pipelineScope,
    annotationShapes: draftShapes,
    pipelineAnnotationShapes: livePipelineAnnotationShapes,
    sourceWidth: currentFile?.width,
    sourceHeight: currentFile?.height,
    pipelineWidth: primaryPipelineVisualization?.width,
    pipelineHeight: primaryPipelineVisualization?.height,
  });
  const canvasEditCoordinateTransform: CanvasCoordinateTransform | null = canvasPresentation.showingPipelineImage ? activePipelineCoordinateTransform : null;
  const pipelineDisplaySlotsReady = pipelineDisplaySlots.length > 1 && pipelineDisplaySlots.every((slot) => slot.result !== null);
  const pipelineOverlayCompatibility = pipelineDisplaySlotsReady
    ? visualizationOverlayCompatibility(pipelineDisplayItems)
    : { allowed: false, reason: '等待当前图的全部显示结果后确认尺寸' };
  const effectiveVisualizationDisplayMode = resolvePipelineDisplayMode(
    visualizationDisplayMode,
    pipelineDisplaySlots.length,
    pipelineDisplaySlotsReady,
    pipelineOverlayCompatibility.allowed,
  );
  const showPipelineViewControls = pipelineEnabled && Boolean(currentFile?.id) && pipelineDisplaySlots.length > 0;
  const showingMultiplePipelineViews = showPipelineViewControls && effectiveVisualizationDisplayMode !== 'source';
  const showingSinglePipelineView = showPipelineViewControls && effectiveVisualizationDisplayMode === 'source' && singlePipelineSource !== 'source';
  const showingPipelinePaneViews = showingMultiplePipelineViews || showingSinglePipelineView;
  const pipelineCanvasItems = showingSinglePipelineView && selectedSinglePipelineItem ? [selectedSinglePipelineItem] : pipelineDisplayItems;
  const showingPipelineImage = canvasPresentation.showingPipelineImage || showingPipelinePaneViews;
  const pipelineSharedPaneTransform = pipelinePaneTransform(view, stageRef.current?.clientWidth ?? 840, stageRef.current?.clientHeight ?? 592);
  const pipelineVisualizationResultKey = pipelineDisplaySlots.map((item) => item.visualization_id).join(':');
  useEffect(() => {
    setVisualizationLayerState((old) => Object.fromEntries(pipelineDisplaySlots.map((item, index) => [item.visualization_id, old[item.visualization_id] ?? { visible: true, opacity: index === 0 ? 100 : 64 }])));
  }, [pipelineDisplaySlots, pipelineVisualizationResultKey]);
  const setVisualizationLayerOpacity = (visualizationId: string, opacityValue: number) => {
    setVisualizationLayerState((old) => updatePipelineLayerOpacity(old, visualizationId, opacityValue));
  };
  const setVisualizationLayerVisible = (visualizationId: string, visible: boolean) => {
    setVisualizationLayerState((old) => updatePipelineLayerVisibility(old, visualizationId, visible));
  };
  useEffect(() => {
    if (effectiveVisualizationDisplayMode !== visualizationDisplayMode) setVisualizationDisplayMode(effectiveVisualizationDisplayMode);
  }, [effectiveVisualizationDisplayMode, visualizationDisplayMode]);
  useEffect(() => {
    if (singlePipelineSource !== 'source' && !pipelineDisplaySlots.some((slot) => slot.visualization_id === singlePipelineSource)) setSinglePipelineSource('source');
  }, [pipelineDisplaySlots, singlePipelineSource]);
  const pipelinePreviewDirty = pipelineBelongsToCurrentAsset && !pipelineMatchesCurrent;
  const pipelineImageLoadFailed = Boolean(pipelineImageUrl && pipelineImageRetryExhausted(pipelineImageAttempts[pipelineImageUrl] ?? 0));
  const sourceCompatibleRasters = currentRasters.filter((raster) => inferenceRasterMatchesSource(raster, currentFile?.width, currentFile?.height));
  const displayableRasters = showingPipelineImage ? [] : sourceCompatibleRasters;
  const incompatibleRasterCount = currentRasters.length - sourceCompatibleRasters.length;
  const pixelResultDisplayBlockedReason = showingPipelineImage
    ? '处理流底图开启时不叠加推理像素结果'
    : currentSegmentationContours.length > 0
      ? null
      : !currentFile?.width || !currentFile.height
        ? '当前源图缺少有效尺寸，不能对齐像素结果'
        : sourceCompatibleRasters.length === 0 && currentRasters.length > 0
          ? `结果尺寸与当前源图 ${currentFile.width} × ${currentFile.height} 不一致，不能叠加`
          : null;
  const pixelResultSummary = pixelResultDisplayBlockedReason ?? [
    currentSegmentationContours.length > 0 ? `${currentSegmentationContours.length} 个分割轮廓` : '',
    sourceCompatibleRasters.length > 0 ? `${sourceCompatibleRasters.length} 个可叠加像素图` : '',
    sourceCompatibleRasters.length > 0 ? `透明度 ${rasterOpacity}%` : '',
    incompatibleRasterCount > 0 ? `${incompatibleRasterCount} 个尺寸不匹配` : '',
  ].filter(Boolean).join(' · ');
  const activeRaster = showMasks ? displayableRasters.find((raster) => raster.id === selectedRasterId) ?? displayableRasters[0] : undefined;
  const activeRasterUrl = activeRaster ? backend.artifactContentUrl(activeRaster.id) : null;
  const displayedImageUrl = canvasPresentation.imageUrl;
  const displayedShapes = canvasPresentation.shapes;
  const displayedWidth = canvasPresentation.width;
  const displayedHeight = canvasPresentation.height;
  const objectListShapes = annotationObjects;
  const annotationObjectIndexes = objectListShapes.map((_shape, index) => index);
  const visibleAnnotationObjectCount = objectListShapes.reduce((count, _shape, index) => count + (hiddenShapeIndexes.has(index) ? 0 : 1), 0);
  const hiddenAnnotationObjectCount = objectListShapes.length - visibleAnnotationObjectCount;
  const labelVisibleBounds = canvasVisibleImageBounds(
    displayedWidth ?? currentFile?.width ?? 1,
    displayedHeight ?? currentFile?.height ?? 1,
    imageRef.current?.clientWidth ?? 0,
    imageRef.current?.clientHeight ?? 0,
    stageRef.current?.clientWidth ?? 0,
    stageRef.current?.clientHeight ?? 0,
    view,
  );
  const canvasLabelUnit = imageUnitsPerScreenPixel();
  const annotationOpticalScale = canvasAnnotationOpticalScale(view.scale);
  const annotationSurfaceWidth = imageRef.current?.clientWidth ?? 0;
  const annotationSurfaceHeight = imageRef.current?.clientHeight ?? 0;
  const annotationStageWidth = stageRef.current?.clientWidth ?? 0;
  const annotationStageHeight = stageRef.current?.clientHeight ?? 0;
  const annotationLayerWidth = annotationSurfaceWidth * view.scale;
  const annotationLayerHeight = annotationSurfaceHeight * view.scale;
  const sourcePixelWidthOnScreen = sourcePixelScreenSize(displayedWidth ?? 0, annotationSurfaceWidth, view.scale);
  const sourcePixelHeightOnScreen = sourcePixelScreenSize(displayedHeight ?? 0, annotationSurfaceHeight, view.scale);
  const sourcePixelScreenMinimum = Math.min(sourcePixelWidthOnScreen || Number.POSITIVE_INFINITY, sourcePixelHeightOnScreen || Number.POSITIVE_INFINITY);
  const pixelGridVisible = isSourcePixelGridVisible(showPixel, sourcePixelWidthOnScreen, sourcePixelHeightOnScreen);
  const pixelGridRevealScaleX = displayedWidth && annotationSurfaceWidth
    ? canvasScaleForSourcePixelSize(displayedWidth, annotationSurfaceWidth, MIN_SOURCE_PIXEL_GRID_SIZE)
    : 0;
  const pixelGridRevealScaleY = displayedHeight && annotationSurfaceHeight
    ? canvasScaleForSourcePixelSize(displayedHeight, annotationSurfaceHeight, MIN_SOURCE_PIXEL_GRID_SIZE)
    : 0;
  const pixelGridRevealScale = pixelGridRevealScaleX && pixelGridRevealScaleY ? Math.max(pixelGridRevealScaleX, pixelGridRevealScaleY) : null;
  const pixelGridBounds = pixelGridVisible && labelVisibleBounds && displayedWidth && displayedHeight
    ? {
      x: Math.max(0, Math.floor(labelVisibleBounds.left)),
      y: Math.max(0, Math.floor(labelVisibleBounds.top)),
      right: Math.min(displayedWidth, Math.ceil(labelVisibleBounds.right)),
      bottom: Math.min(displayedHeight, Math.ceil(labelVisibleBounds.bottom)),
    }
    : null;
  const annotationSvgStyle = {
    opacity: opacity / 100,
    left: annotationStageWidth / 2 - annotationLayerWidth / 2 + view.x,
    top: annotationStageHeight / 2 - annotationLayerHeight / 2 + view.y,
    width: annotationLayerWidth,
    height: annotationLayerHeight,
    visibility: annotationLayerWidth > 0 && annotationLayerHeight > 0 ? 'visible' : 'hidden',
    '--annotation-optical-scale': annotationOpticalScale,
    '--annotation-stroke': `${1.1 * annotationOpticalScale}px`,
    '--annotation-selected-stroke': `${1.45 * annotationOpticalScale}px`,
    '--annotation-prediction-stroke': `${annotationOpticalScale}px`,
    '--annotation-control-stroke': '1.6px',
    '--annotation-control-hover-stroke': '2.2px',
    '--annotation-vertex-stroke': `${CANVAS_VERTEX_CONTROL_DIAMETER_PX}px`,
    '--annotation-shadow-blur': `${annotationOpticalScale}px`,
    '--annotation-control-shadow-blur': '3px',
  } as React.CSSProperties;
  const realCanvasLabelEntries: Array<{ key: string; shape: AnnotationShape; index: number; prediction: boolean; priority: number }> = [
    ...(showGT ? displayedShapes.flatMap((shape, index) => hiddenShapeIndexes.has(index) || canvasLabelOpacity(view.scale, selectedShapeIndex === index) <= 0 ? [] : [{ key: `gt-${index}`, shape: shape as AnnotationShape, index, prediction: false, priority: selectedShapeIndex === index ? 2 : 1 }]) : []),
    ...(!showingPipelineImage && canvasLabelOpacity(view.scale) > 0 ? visiblePredictionCanvasEntries.map(({ prediction: shape, index }) => ({ key: `pred-${index}`, shape: shape as AnnotationShape, index, prediction: true, priority: selectedPredictionIndex === index ? 2 : 0 })) : []),
  ];
  const realCanvasLabelLayoutValues = canvasLabelLayouts(
    realCanvasLabelEntries.map((entry) => ({ points: canvasLabelPointsForShape(entry.shape), text: fitCanvasLabelText(`${entry.shape.label || '未命名'}${typeof entry.shape.score === 'number' ? ` · ${(entry.shape.score * 100).toFixed(0)}%` : ''}`), priority: entry.priority, anchor: canvasLabelAnchorForShape(entry.shape) })),
    displayedWidth ?? currentFile?.width ?? 1,
    displayedHeight ?? currentFile?.height ?? 1,
    canvasLabelUnit,
    labelVisibleBounds ?? undefined,
  );
  const realCanvasLabelLayouts = Object.fromEntries(realCanvasLabelEntries.map((entry, index) => [entry.key, realCanvasLabelLayoutValues[index]])) as Record<string, CanvasLabelLayout | null>;
  const polygonScreenUnit = polygonDraft.length > 0 ? imageUnitsPerScreenPixel() : 1;
  const polygonPreviewPoints = polygonDraft.length > 0
    ? [...polygonDraft, ...(polygonPointer ? [polygonPointer] : [])]
    : [];
  const shouldRequestTiles = Boolean(
    backend.mode === 'online'
    && dataset.id
    && currentFile?.id
    && currentFile.imagePath
    && !showingPipelineImage
    && displayedWidth
    && displayedHeight
    && displayedWidth * displayedHeight >= LARGE_IMAGE_TILE_THRESHOLD_PIXELS,
  );
  const tileAssetKey = shouldRequestTiles && dataset.id && currentFile
    ? `${dataset.id}:${currentFile.id}`
    : null;
  const activeTileMetadata = tileAssetKey
    && tileMetadataState.assetKey === tileAssetKey
    && tileMetadataState.phase === 'ready'
    && tileMetadataState.data?.width === displayedWidth
    && tileMetadataState.data?.height === displayedHeight
    ? tileMetadataState.data
    : null;
  const tileMetadataLoading = Boolean(
    tileAssetKey
    && (tileMetadataState.assetKey !== tileAssetKey || tileMetadataState.phase === 'loading'),
  );
  const tilePlaceholderUrl = shouldRequestTiles && dataset.id && currentFile
    ? backend.assetUrl(dataset.id, currentFile.id, 'thumbnail', '?max_size=2048&format=webp')
    : null;
  const currentTileUrl = useCallback((level: number, x: number, y: number, format: string) => {
    if (!dataset.id || !currentFile) return null;
    return buildTileUrl(dataset.id, currentFile.id, level, x, y, format);
  }, [buildTileUrl, currentFile, dataset.id]);

  useEffect(() => {
    if (!tileAssetKey || !dataset.id || !currentFile) {
      setTileMetadataState((current) => current.assetKey === null && current.phase === 'idle'
        ? current
        : { assetKey: null, phase: 'idle', data: null });
      return;
    }
    const controller = new AbortController();
    setTileMetadataState({ assetKey: tileAssetKey, phase: 'loading', data: null });
    void loadTileMetadata(dataset.id, currentFile.id, controller.signal).then((metadata) => {
      if (controller.signal.aborted) return;
      const valid = metadata.width === displayedWidth
        && metadata.height === displayedHeight
        && metadata.tile_size > 0
        && metadata.max_level >= 0
        && Boolean(metadata.source_etag);
      setTileMetadataState({
        assetKey: tileAssetKey,
        phase: valid ? 'ready' : 'error',
        data: valid ? metadata : null,
      });
    }).catch(() => {
      if (!controller.signal.aborted) {
        setTileMetadataState({ assetKey: tileAssetKey, phase: 'error', data: null });
      }
    });
    return () => controller.abort();
  }, [currentFile, dataset.id, displayedHeight, displayedWidth, loadTileMetadata, tileAssetKey]);
  const selectedDraftShape = selectedShapeIndex === null ? null : draftShapes[selectedShapeIndex] ?? null;
  const selectedPredictionShape = selectedPredictionIndex === null ? null : currentDetectionPredictions[selectedPredictionIndex] ?? null;
  const selectedPanelShape = selectedDraftShape ?? selectedPredictionShape;
  useEffect(() => {
    if (rightTab !== 'layers' || selectedShapeIndex === null) return;
    const frame = window.requestAnimationFrame(() => {
      const list = annotationObjectListRef.current;
      const row = list?.querySelector<HTMLElement>(`.annotation-object-row[data-shape-index="${selectedShapeIndex}"]`);
      if (!list || !row) return;
      const listRect = list.getBoundingClientRect();
      const rowRect = row.getBoundingClientRect();
      const top = rowRect.top < listRect.top
        ? list.scrollTop - (listRect.top - rowRect.top)
        : rowRect.bottom > listRect.bottom
          ? list.scrollTop + (rowRect.bottom - listRect.bottom)
          : null;
      if (top === null) return;
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      list.scrollTo({ top, behavior: reduceMotion ? 'auto' : 'smooth' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [rightTab, selectedShapeIndex]);
  const changeAnnotationIndexesVisibility = (indexes: number[], visible: boolean) => {
    setHiddenShapeIndexes((hidden) => setAnnotationIndexesVisible(hidden, indexes, visible));
    if (!visible) setSelectedShapeIndex((selected) => selected !== null && indexes.includes(selected) ? null : selected);
  };
  const changeShapeVisibility = (index: number, visible: boolean) => changeAnnotationIndexesVisibility([index], visible);
  const revealShapeFromList = (shape: AnnotationShape) => {
    const sourceWidth = displayedWidth ?? 0;
    const sourceHeight = displayedHeight ?? 0;
    const surfaceWidth = imageRef.current?.clientWidth ?? 0;
    const surfaceHeight = imageRef.current?.clientHeight ?? 0;
    if (!sourceWidth || !sourceHeight || !surfaceWidth || !surfaceHeight || !labelVisibleBounds) return;
    const points = canvasLabelPointsForShape(shape);
    if (!points.length) return;
    const minX = Math.min(...points.map((point) => point[0]));
    const minY = Math.min(...points.map((point) => point[1]));
    const maxX = Math.max(...points.map((point) => point[0]));
    const maxY = Math.max(...points.map((point) => point[1]));
    const padding = 24 * canvasLabelUnit;
    const safelyVisible = minX >= labelVisibleBounds.left + padding
      && minY >= labelVisibleBounds.top + padding
      && maxX <= labelVisibleBounds.right - padding
      && maxY <= labelVisibleBounds.bottom - padding;
    if (safelyVisible) return;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    setView((current) => ({
      ...current,
      x: (0.5 - centerX / sourceWidth) * surfaceWidth * current.scale,
      y: (0.5 - centerY / sourceHeight) * surfaceHeight * current.scale,
    }));
  };
  const selectShapeFromList = (shape: AnnotationShape, index: number) => {
    setSelectedShapeIndex(index);
    setSelectedShape(shape.shape_type);
    setTool('select');
    revealShapeFromList(shape);
  };
  const canUndoAnnotation = annotationHistoryVersion >= 0 && Boolean(annotationHistoryRef.current?.past.length);
  const canRedoAnnotation = annotationHistoryVersion >= 0 && Boolean(annotationHistoryRef.current?.future.length);
  const annotationPersistenceText = annotationEditActive
    ? '编辑事务进行中'
    : annotationPersistence.message ?? (annotationDirty ? '未保存草稿' : '已与服务端同步');
  const saveModeLabel = annotationSaving
    ? '保存中'
    : annotationPersistence.phase === 'error'
      ? '保存失败'
      : annotationDirty && ['local', 'offline'].includes(annotationPersistence.phase)
        ? '本机草稿'
        : annotationDirty
          ? '未保存'
          : '已保存';
  const saveModeTitle = annotationSaving
    ? annotationPersistenceText
    : `${annotationPersistenceText} · 当前图自动保存${annotationAutoSave ? '已开启；切换图片后关闭' : '已关闭；点击开启'}`;
  const rotationPreviewShape = drawPreview && tool === 'rotation'
    ? createDragShape('rotation', [drawPreview.x1, drawPreview.y1], [drawPreview.x2, drawPreview.y2])
    : null;
  const rotationPreviewScreenUnit = rotationPreviewShape ? imageUnitsPerScreenPixel() : 1;
  const rotationPreviewCorner = rotationPreviewShape ? rotationCornerHandle(rotationPreviewShape.points) : null;
  const agentCapabilitySections: Array<{ id: AgentCapability['group']; title: string; description: string; capabilities: AgentCapability[] }> = [
    { id: 'inspect', title: '数据检查', description: '只读执行并记录审计', capabilities: backend.agentStatus.data?.capabilities.filter((item) => item.group === 'inspect') ?? [] },
    { id: 'prepare', title: '界面与草案', description: '生成提案，确认后交给界面', capabilities: backend.agentStatus.data?.capabilities.filter((item) => item.group === 'prepare') ?? [] },
    { id: 'run', title: '后台执行', description: '确认后创建持久任务', capabilities: backend.agentStatus.data?.capabilities.filter((item) => item.group === 'run') ?? [] },
  ];
  const agentUnavailableTitle = backend.mode !== 'online'
    ? '本地服务未连接'
    : backend.agentStatus.phase === 'loading' || backend.agentStatus.phase === 'idle'
      ? '正在检查 Agent 后端'
      : backend.agentStatus.phase === 'error'
        ? '无法读取 Agent 配置'
        : 'Agent 后端未配置';
  const agentUnavailableMessage = backend.mode !== 'online'
    ? 'Agent 依赖本地服务与已配置的工具规划后端。恢复本地服务后再检查配置。'
    : backend.agentStatus.phase === 'error'
      ? backend.agentStatus.error?.message ?? '读取 Agent 状态失败，请重试。'
      : backend.agentStatus.data?.message ?? '配置一个 OpenAI-compatible 工具规划服务后才能使用 Agent。';
  const openRightTab = (tab: RightTab) => {
    rightSidebarRef.current?.scrollTo({ top: 0 });
    setRightTab(tab);
  };
  const openInferenceFeatureConfiguration = (node: FlowNode) => {
    const modelId = typeof node.parameters.model_id === 'string' ? node.parameters.model_id : '';
    if (modelId && modelId !== selectedModel.id && displayedModelCatalog.some((model) => model.id === modelId)) chooseModel(modelId);
    openRightTab('inference');
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      inferenceFeatureCardRef.current?.focus({ preventScroll: true });
    }));
  };
  const canvasCursorMode = resolveCanvasCursorMode(tool, {
    temporaryPan: spaceDown,
    panning: canvasPanning,
    drawingEnabled: Boolean(!pendingManualShape),
  });
  const currentNavigatorMetrics = readNavigatorMetrics() ?? {
    navigatorWidth: 132,
    navigatorHeight: 70,
    imageWidth: imageRef.current?.offsetWidth ?? 840,
    imageHeight: imageRef.current?.offsetHeight ?? 592,
    viewportWidth: stageRef.current?.clientWidth ?? 840,
    viewportHeight: stageRef.current?.clientHeight ?? 592,
  };
  const navigatorBox = navigatorViewport(view, currentNavigatorMetrics);
  const backgroundTaskControl = taskIconVisible ? <section ref={taskActivityRef} className={`background-task-control ${taskStreamVisible ? 'visible' : ''} ${taskStreamOpen ? 'pinned' : ''} ${activeTaskCount ? 'has-active' : ''} ${attentionTaskCount ? 'needs-attention' : ''}`} aria-label="后台任务入口" onMouseEnter={() => setTaskStreamHovered(true)} onMouseLeave={() => setTaskStreamHovered(false)} onFocusCapture={() => setTaskStreamFocused(true)} onBlurCapture={(event) => { const next = event.relatedTarget; if (!(next instanceof Node) || !event.currentTarget.contains(next)) setTaskStreamFocused(false); }}>
    <button ref={taskActivityButtonRef} type="button" className={`top-action-button background-task-button ${taskStreamVisible ? 'active' : ''}`} aria-label={taskButtonLabel} aria-haspopup="dialog" aria-expanded={taskStreamVisible} aria-pressed={taskStreamOpen} aria-controls="background-activity-stream" onClick={() => setTaskStreamOpen((open) => !open)}><BackgroundTasksIcon attention={attentionTaskCount > 0} complete={taskIconComplete} />{activeTaskCount > 0 && <span className={`background-task-progress ${latestTaskDownloadView?.percent === null ? 'indeterminate' : ''}`} aria-hidden="true"><i style={{ width: latestTaskDownloadView?.percent === null ? '42%' : `${latestTaskPercent}%` }} /></span>}{activeTaskCount + attentionTaskCount > 1 || attentionTaskCount > 0 ? <span className={`background-task-badge ${attentionTaskCount > 0 ? 'attention' : ''}`} aria-hidden="true">{activeTaskCount + attentionTaskCount}</span> : null}<span className="task-progress-announcement" aria-live="polite">{taskButtonLabel}</span></button>
    {taskStreamVisible && <div id="background-activity-stream" className="background-activity" role="dialog" aria-label="后台任务">
      <header className="activity-popover-heading"><div><strong>后台任务</strong><small>{backend.mode !== 'online' ? '离线 · 显示最后快照' : latestBackgroundTask ? `${backgroundTaskTitle(latestBackgroundTask)} · ${backgroundTaskStateLabel(latestBackgroundTask.state)}` : '批处理、推理与模型下载'}</small></div>{latestBackgroundTask && ACTIVE_TASK_STATES.has(latestBackgroundTask.state) && <b>{latestTaskDownloadView?.percent === null ? '…' : `${latestTaskPercent}%`}</b>}{attentionTaskCount > 0 && <em>{attentionTaskCount} 项需处理</em>}</header>
      {latestBackgroundTask && ACTIVE_TASK_STATES.has(latestBackgroundTask.state) && <div className={`activity-summary-progress ${latestTaskDownloadView?.percent === null ? 'indeterminate' : ''}`} role="progressbar" aria-label={`${backgroundTaskTitle(latestBackgroundTask)}进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={latestTaskDownloadView?.percent === null ? undefined : latestTaskPercent}><span style={{ width: latestTaskDownloadView?.percent === null ? '35%' : `${latestTaskPercent}%` }} /></div>}
      <div className="activity-history-toolbar"><div role="group" aria-label="最近任务时间范围">{BACKGROUND_TASK_HISTORY_HOURS.map((hours) => <button key={hours} type="button" className={taskHistoryHours === hours ? 'active' : ''} aria-pressed={taskHistoryHours === hours} onClick={() => changeTaskHistoryHours(hours)}>{hours === 1 ? '1 小时' : hours === 24 ? '24 小时' : '7 天'}</button>)}</div><div><button type="button" disabled={clearableTaskIds.length === 0} onClick={clearCompletedBackgroundTasks}>清除已完成{clearableTaskIds.length ? ` ${clearableTaskIds.length}` : ''}</button>{Object.keys(dismissedTaskVersions).length > 0 && <button type="button" onClick={restoreHiddenBackgroundTasks}>恢复已隐藏</button>}</div><small>最多显示服务端返回的最近 100 项；清除仅影响本机列表</small></div>
      <div className="activity-stream" role="region" aria-label="后台活动流"><header><span>{backend.jobEvents.mode === 'realtime' ? '● 实时' : backend.jobEvents.mode === 'connecting' ? '◌ 连接中' : backend.jobEvents.mode === 'offline' ? '○ 离线' : '○ 轮询'}</span><small>最近 {taskStreamJobs.length} 项</small><button onClick={() => void backend.refreshJobs()}>刷新</button></header>{taskStreamJobs.map((job) => {
        const terminal = job.completed + job.failed + job.canceled;
        const liveProgress = Object.values(backend.jobItemProgress).find((progress) => progress.job_id === job.job_id);
        const liveView = liveProgress ? formatDownloadProgress(liveProgress.received_bytes, liveProgress.total_bytes) : null;
        const percent = liveView?.percent ?? backgroundTaskPercent(job);
        const pending = taskActionPending?.startsWith(job.job_id);
        const selected = selectedJobId === job.job_id;
        return <article key={job.job_id} className={`activity-row ${job.state} ${selected ? 'selected' : ''}`}><button type="button" className="activity-row-main" aria-pressed={selected} onClick={() => void inspectJob(job.job_id)}><span className={`task-kind ${job.kind}`}>{job.kind === 'pipeline' ? 'FLOW' : job.kind === 'model_download' ? 'GET' : job.kind === 'category_rename' ? 'REN' : 'AI'}</span><span><strong>{backgroundTaskTitle(job)}</strong><small>{backgroundTaskStateLabel(job.state)} · {job.kind === 'model_download' ? `${job.total} 个文件${liveView ? ` · ${liveView.label}` : ''}` : `${job.total} 张`}</small></span><b>{liveView?.percent === null ? '…' : `${percent}%`}</b></button>{ACTIVE_TASK_STATES.has(job.state) && <div className={`task-progress ${liveView?.percent === null ? 'indeterminate' : ''}`} role="progressbar" aria-label={`${backgroundTaskTitle(job)}进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={liveView?.percent === null ? undefined : percent}><span style={{ width: liveView?.percent === null ? '35%' : `${percent}%` }} /></div>}{BACKGROUND_TASK_ATTENTION_STATES.has(job.state) && <div className="activity-attention-actions"><span title={job.error}>{job.error ?? '该任务需要处理'}</span><button type="button" disabled={pending} onClick={() => retryBackgroundTask(job)}>{pending ? '处理中…' : '重试失败项'}</button><button type="button" onClick={() => ignoreBackgroundTask(job)}>忽略提醒</button></div>}{selected && <div className="activity-row-details"><div className="task-counts"><span>成功 {job.completed}</span><span>失败 {job.failed}</span><span>取消 {job.canceled}</span><span>剩余 {Math.max(0, job.total - terminal)}</span></div><footer>{job.kind !== 'model_download' && ['queued','running'].includes(job.state) && <button disabled={pending} onClick={() => void controlTask(job.job_id, 'pause')}>暂停</button>}{job.state === 'paused' && <button disabled={pending} onClick={() => void controlTask(job.job_id, 'resume')}>继续</button>}{!['succeeded','succeeded_with_errors','failed','canceled'].includes(job.state) && <button className="cancel" disabled={pending} onClick={() => void controlTask(job.job_id, 'cancel')}>取消</button>}</footer></div>}</article>;
      })}{taskStreamJobs.length === 0 && <div className="activity-empty"><strong>当前没有后台任务</strong><small>运行全部图像处理、批量推理或下载模型后，进度会自动出现在这里。</small></div>}{selectedJobId && backend.jobItems.phase === 'ready' && backend.jobItems.data && <section className="task-items"><div className="section-title"><h2>任务项状态</h2><span>{backend.jobItems.data.total} 项</span></div>{backend.jobItems.data.items.slice(0, 8).map((item) => { const progress = backend.jobItemProgress[`${selectedJobId}:${item.asset_id}`]; const progressView = progress ? formatDownloadProgress(progress.received_bytes, progress.total_bytes) : null; return <div key={item.asset_id} className={`task-item ${item.state}`}><span>{item.position + 1}</span><div><strong>{item.asset_id.slice(0, 18)}</strong><small>{item.error ?? progressView?.label ?? `attempt ${item.attempts}`}</small>{progressView && <i><span style={{ width: progressView.percent === null ? '35%' : `${progressView.percent}%` }} /></i>}</div><b>{progressView?.percent === null ? '下载中' : progressView?.percent !== undefined ? `${progressView.percent}%` : item.state}</b></div>; })}</section>}</div>
    </div>}
  </section> : null;

  return (
    <main key={uiLanguage} ref={appShellRef} className={`app-shell cursor-${canvasCursorMode} ${settingsOpen ? 'settings-mode' : ''}`}>
      <UiLanguageBridge language={uiLanguage} />
      <GlobalTooltip />
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden="true" /><span>LabelOne</span><span className="prototype-pill">本地工作台</span></div>
        <div className="pathbar" role="group" aria-label="当前图片位置和复制操作"><span className={localServiceDown ? 'status-dot offline' : 'status-dot'} /><span className={`backend-badge ${backend.mode}`}>{backendLabel}</span>{openedDataset && currentFile ? <div className={`current-file-location ${canCopyAbsoluteImagePath ? 'has-full-path' : 'static-path'}`}>{canCopyAbsoluteImagePath ? <button type="button" className="path-copy-segment path-copy-full" aria-label="复制当前图片完整绝对路径" title={`点击复制完整绝对路径：${currentFilePath.fullPath}`} onClick={() => void copyTextToClipboard(currentFilePath.fullPath, '已复制完整路径')}><span className="path-copy-directory-text">{currentFilePath.directoryLabel}</span><span className="path-copy-filename-reserve" aria-hidden="true">{currentFilePath.fileName}</span></button> : <span className="path-copy-static path-copy-directory" title="当前图片没有可用的本地绝对路径"><span>本地路径不可用 · {currentFilePath.directoryLabel}</span></span>}<button type="button" className="path-copy-segment path-copy-filename" aria-label="复制当前图片文件名" disabled={!currentFilePath.fileName} title={currentFilePath.fileName ? `点击复制文件名：${currentFilePath.fileName}` : '当前没有可复制的文件名'} onClick={() => void copyTextToClipboard(currentFilePath.fileName, `已复制文件名：${currentFilePath.fileName}`)}><span>{currentFilePath.fileName || '无当前文件'}</span></button></div> : <span className="empty-window-label">未打开项目</span>}</div>
        <div className="top-actions">
          {backgroundTaskControl}
          <button type="button" className="top-language-button" aria-label={uiLanguage === 'zh-CN' ? '切换到英文界面' : 'Switch to Chinese'} title={uiLanguage === 'zh-CN' ? '切换到英文界面' : 'Switch to Chinese'} onClick={toggleUiLanguage}><span lang={uiLanguage === 'zh-CN' ? 'en' : 'zh-CN'}>{uiLanguage === 'zh-CN' ? 'EN' : '中'}</span></button>
          <button ref={settingsButtonRef} type="button" className={`top-action-button ${settingsOpen ? 'active' : ''}`} aria-label="设置" title="全局设置" aria-haspopup="dialog" aria-expanded={settingsOpen} aria-controls="global-settings-page" onClick={openGlobalSettings}><SettingsIcon /></button>
          <button type="button" className={`top-action-button ${isFullscreen ? 'active' : ''}`} aria-label={isFullscreen ? '退出全屏' : '进入全屏'} title={isFullscreen ? '退出全屏' : '全屏'} aria-pressed={isFullscreen} onClick={() => void toggleFullscreen()}><FullscreenIcon active={isFullscreen} /></button>
        </div>
      </header>
      <span className="language-announcement" aria-live="polite" aria-atomic="true">{languageAnnouncement}</span>

      {openedDataset && currentFile ? <section className="workspace">
        <aside className="sidebar left-sidebar">
          <div className="panel-heading dataset-heading">
            <div><span className="eyebrow">数据集</span><h1>{dataset.name}</h1></div>
            <div className="dataset-actions"><button className="open-dataset-button" disabled={directoryPickerPending === 'image' || scanRegistering} onClick={() => void pickImageDataset()}>{directoryPickerPending === 'image' ? '选择中…' : '＋ 打开'}</button></div>
          </div>
          <div className="dataset-summary"><span><strong>{validTotal.toLocaleString()}</strong> 张可标注图片</span></div>
          <div className="smart-search-wrap">
            <div className={`smart-search ${searchError ? 'has-error' : ''}`}><CustomSelect className="search-mode" ariaLabel="搜索模式" value={searchMode} options={[{ value: 'smart', label: '智能' }, { value: 'text', label: '文本' }, { value: 'regex', label: '正则' }, { value: 'query', label: '条件' }]} onChange={(mode) => setSearchMode(mode as SearchMode)} /><input aria-label="搜索文件" placeholder={searchMode === 'regex' ? '^wafer_.*\\.(tif|png)$' : searchMode === 'query' ? 'class:scratch AND size>8k' : '文件名、class:scratch、size>8k…'} value={search} onChange={(event) => setSearch(event.target.value)} onKeyDown={(event) => { if (event.key === 'Escape') setSearchHelp(false); }} /><button type="button" className={`regex-toggle ${searchMode === 'regex' ? 'active' : ''}`} aria-label="切换正则表达式搜索" aria-pressed={searchMode === 'regex'} title={searchMode === 'regex' ? '退出正则搜索' : '启用正则搜索'} onClick={() => setSearchMode((mode) => mode === 'regex' ? 'smart' : 'regex')}>.*</button><button className="search-help-button" aria-label="搜索语法提示" aria-expanded={searchHelp} aria-controls="search-hint-panel" title={searchHelp ? '关闭搜索提示' : '显示搜索提示'} onClick={() => setSearchHelp((open) => !open)}>?</button></div>
            {searchError && <div className="search-error">{searchError}</div>}
            {searchHelp && <div id="search-hint-panel" className="search-language" role="region" aria-label="搜索语法提示">
              <header><strong>结构化搜索</strong><button onClick={() => setSearchHelp(false)}>×</button></header>
              <button onClick={() => { setSearchMode('query'); setSearch('class:scratch AND size>8k'); setSearchHelp(false); }}>class:scratch AND size&gt;8k <span>AND 条件</span></button>
              <button onClick={() => { setSearchMode('query'); setSearch('annotated:false OR type:rotation'); setSearchHelp(false); }}>annotated:false OR type:rotation <span>OR 条件</span></button>
              <button onClick={() => { setSearchMode('smart'); setSearch('type:rotation has:annotation'); setSearchHelp(false); }}>type:rotation <span>旋转框 + 标注</span></button>
              <button onClick={() => { setSearchMode('regex'); setSearch('^(wafer|chip)_.*\\.(tif|png)$'); setSearchHelp(false); }}>/^(wafer|chip)_…/ <span>正则表达式</span></button>
              <footer>支持 path:、class:、type:、annotated:、has:、size&gt; 与 AND / OR / NOT: 条件</footer>
            </div>}
          </div>
          <div className="list-label"><span>文件</span><div className="file-list-tools"><span>{isRealDataset && backend.assetSearch.phase === 'loading' ? '查询中…' : `${listedFiles.length} 已载入 / ${matchedTotal.toLocaleString()} 结果`}{isRealDataset && backend.assetSearch.data.index_revision > 0 ? ` · r${backend.assetSearch.data.index_revision}` : ''}{isRealDataset && (backend.assetSearch.stale || backend.mode !== 'online') ? ' · 可能已过期' : ''}</span><CustomSelect className={`annotation-filter-select ${filter !== 'all' ? 'filtered' : ''}`} ariaLabel="筛选文件" title={`筛选：${fileAnnotationFilterLabels[filter]}`} value={filter} options={fileAnnotationFilters.map((option) => ({ value: option, label: fileAnnotationFilterLabels[option] }))} onChange={(next) => setFilter(next as FileAnnotationFilter)} /></div></div>
          <VirtualFileList
            key={realSearchKey}
            items={listedFiles}
            total={matchedTotal}
            itemKey={fileItemKey}
            activeItemKey={currentFile?.id ?? null}
            hasMore={isRealDataset && backend.assetSearch.data.next_cursor != null}
            loadingMore={isRealDataset && backend.assetSearchLoadingMore}
            onEndReached={isRealDataset ? loadMoreRealFiles : undefined}
            emptyState={<div className="empty-list">{isRealDataset && backend.assetSearch.phase === 'loading' ? '正在查询真实本地索引…' : '没有符合条件的项目'}</div>}
            renderItem={(file) => {
              const disabled = isRealDataset ? file.selectable !== true || file.rawStatus !== 'valid' : file.status !== 'valid';
              const jobItem = fileProgressJob ? backend.jobItemSnapshots[`${fileProgressJob.job_id}:${file.id}`] : undefined;
              const statusView = resolveFileStatusIndicator(file.annotationFileExists, file.annotations, fileProgressJob?.kind, jobItem);
              const statusStyle = statusView.progress === null ? undefined : ({ '--file-progress-angle': `${statusView.progress * 360}deg` } as React.CSSProperties);
              const thumbnailUrl = isRealDataset && dataset.id && file.selectable === true
                ? backend.assetUrl(dataset.id, file.id, 'thumbnail', '?max_size=96&format=webp')
                : null;
              return <button className={`file-row ${currentFile?.id === file.id ? 'active' : ''} ${disabled ? 'disabled' : ''}`} aria-current={currentFile?.id === file.id ? 'true' : undefined} disabled={disabled} onClick={() => selectFile(file)}>
                <span className={`thumb ${file.variant}`} aria-hidden="true" style={thumbnailUrl ? { backgroundImage: `url(${thumbnailUrl})`, backgroundSize: 'cover', backgroundPosition: 'center' } : undefined} /><span className="file-copy"><strong>{file.name}</strong><small>{file.meta}</small></span>
                <span className="file-status-slot" data-tooltip={statusView.kind !== 'empty' ? statusView.label : undefined} data-tooltip-anchor="pointer">{statusView.kind !== 'empty' && <span className={`file-status-indicator ${statusView.kind}`} style={statusStyle} role="img" aria-label={statusView.label}>{statusView.kind === 'check' ? '✓' : statusView.kind === 'failed' ? '!' : statusView.kind === 'canceled' ? '–' : <i aria-hidden="true" />}</span>}</span>
              </button>;
            }}
          />
          <div className="scan-state"><span className={scanIsLoading ? 'spinner' : 'scan-ready-dot'} /><div><strong>本地数据索引</strong><small>{scanIsLoading ? `扫描会话 ${backend.scan.data?.persisted_items ?? 0} 项已落盘…` : scanSessionState === 'succeeded' ? `${scanSummary.valid} 有效 · ${scanSummary.hidden_image_only} 仅图已隐藏` : scanSessionState === 'interrupted' ? `扫描已中断 · 保留 ${backend.scan.data?.persisted_items ?? 0} 项` : scanSessionState === 'failed' ? `扫描失败 · ${backend.scan.data?.error ?? '可继续'}` : backend.mode === 'offline' ? '服务离线 · 保留最后结果' : '索引已就绪'}</small></div><button onClick={() => backend.mode === 'online' ? void backend.refreshHealth() : notify('本地服务默认监听 127.0.0.1:8766')}>{backend.mode === 'online' ? '刷新' : '详情'}</button></div>
        </aside>

        <section className="canvas-column">
          <div className="canvas-toolbar">
            <div className="toolbar-group canvas-position-counter"><span className="counter" aria-label={`当前第 ${activeIndex} 张，共 ${(isRealDataset ? matchedTotal : validTotal).toLocaleString()} 张`}>{activeIndex} / {(isRealDataset ? matchedTotal : validTotal).toLocaleString()}</span></div>
            <div className="toolbar-group centered" aria-label="标注工具栏">
              <div className="tool-mode-group" role="group" aria-label="画布操作">
                {([['select', '↖', '选择'], ['pan', '✥', '平移']] as [AnnotationTool, string, string][]).map(([id, icon, label]) => { const action = toolShortcutActions[id]!; const definition = shortcutDefinitions.find((item) => item.id === action)!; const shortcut = displayShortcut(shortcuts[action], useMacShortcutSymbols); const guidance = id === 'pan' ? `${definition.description}；按住 Space 可临时平移；画布聚焦后可用` : `${definition.description}；画布聚焦后可用`; return <button key={id} className={`tool ${tool === id ? 'active' : ''}`} aria-pressed={tool === id} aria-keyshortcuts={shortcutAriaLabel(shortcuts[action], useMacShortcutSymbols)} data-tooltip-title={definition.label} data-tooltip={guidance} data-shortcut={shortcut} onClick={() => activateTool(id)}><span className="tool-glyph" aria-hidden="true">{icon}</span><span>{label}</span></button>; })}
              </div>
              <div className="tool-mode-group drawing-tools" role="group" aria-label="绘制标注">
                {([['rect', '矩形'], ['rotation', '旋转框'], ['polygon', '多边形'], ['point', '点'], ['line', '直线'], ['circle', '圆'], ['brush', '自由线']] as [AnnotationTool, string][]).map(([id, label]) => { const action = toolShortcutActions[id]!; const definition = shortcutDefinitions.find((item) => item.id === action)!; const shortcut = displayShortcut(shortcuts[action], useMacShortcutSymbols); const guidance = id === 'polygon' ? `${definition.description}；Enter 完成，Backspace 撤回顶点，Esc 取消；画布聚焦后可用` : `${definition.description}；画布聚焦后可用`; return <button key={id} className={`tool ${tool === id ? 'active' : ''}`} aria-pressed={tool === id} aria-keyshortcuts={shortcutAriaLabel(shortcuts[action], useMacShortcutSymbols)} data-tooltip-title={definition.label} data-tooltip={guidance} data-shortcut={shortcut} onClick={() => activateTool(id)}><ShapeTypeIcon shapeType={id === 'brush' ? 'linestrip' : id} /><span>{label}</span></button>; })}
              </div>
            </div>
            <div className="toolbar-group zoom-controls"><IconButton label="缩小" description="以画布中心为锚点缩小；画布聚焦后可用" shortcut={displayShortcut(shortcuts['canvas.zoomOut'], useMacShortcutSymbols)} ariaShortcut={shortcutAriaLabel(shortcuts['canvas.zoomOut'], useMacShortcutSymbols)} onClick={() => zoomAtCenter(1 / CANVAS_ZOOM_STEP)}>−</IconButton><label className={`zoom-value-control${zoomEditing ? ' is-editing' : ''}`} data-tooltip-title={zoomEditing ? '输入缩放比例' : `缩放 ${Math.round(view.scale * 100)}%`} data-tooltip={`单击修改 · 双击恢复 100% · 范围 25%–${Math.round(maximumCanvasScale() * 100)}%`} onDoubleClick={(event) => { event.preventDefault(); resetZoomPercent(); }}><input ref={zoomValueInputRef} type="text" inputMode="decimal" role="spinbutton" value={zoomEditing ? zoomDraft : String(Math.round(view.scale * 100))} readOnly={!zoomEditing} aria-readonly={!zoomEditing} aria-label={zoomEditing ? '输入缩放百分比' : `缩放 ${Math.round(view.scale * 100)}%；单击修改，双击恢复 100%`} aria-valuemin={25} aria-valuemax={Math.round(maximumCanvasScale() * 100)} aria-valuenow={Number((zoomEditing ? zoomDraft : String(Math.round(view.scale * 100))).replace(/%$/, '')) || Math.round(view.scale * 100)} onClick={(event) => { if (event.detail === 1 && !zoomEditing) beginZoomEditing(); }} onChange={(event) => setZoomDraft(event.target.value)} onBlur={() => { if (zoomEditing) commitZoomEditing(); }} onKeyDown={(event) => { event.stopPropagation(); if (zoomEditing && event.key === 'Enter') { event.preventDefault(); commitZoomEditing(true); } else if (zoomEditing && event.key === 'Escape') { event.preventDefault(); cancelZoomEditing(true); } else if (!zoomEditing && (event.key === 'Enter' || event.key === 'F2')) { event.preventDefault(); beginZoomEditing(); } }} /><span aria-hidden="true">%</span></label><IconButton label="放大" description="以画布中心为锚点放大；画布聚焦后可用" shortcut={displayShortcut(shortcuts['canvas.zoomIn'], useMacShortcutSymbols)} ariaShortcut={shortcutAriaLabel(shortcuts['canvas.zoomIn'], useMacShortcutSymbols)} onClick={() => zoomAtCenter(CANVAS_ZOOM_STEP)}>＋</IconButton><button className="fit-button" aria-keyshortcuts={shortcutAriaLabel(shortcuts['canvas.fit'], useMacShortcutSymbols)} data-tooltip-title="适应窗口" data-tooltip="居中并适应当前可用空间；画布聚焦后可用" data-shortcut={displayShortcut(shortcuts['canvas.fit'], useMacShortcutSymbols)} onClick={() => setView({ scale: .92, x: 0, y: 0 })}>适应</button><button className="fit-button" aria-keyshortcuts={shortcutAriaLabel(shortcuts['canvas.actualSize'], useMacShortcutSymbols)} data-tooltip-title="实际大小" data-tooltip="恢复到 1:1 显示；画布聚焦后可用" data-shortcut={displayShortcut(shortcuts['canvas.actualSize'], useMacShortcutSymbols)} onClick={zoomOneToOne}>1:1</button></div>
          </div>

          <div ref={stageRef} className="canvas-stage" role="region" aria-label="图像标注画布" aria-keyshortcuts={`${Object.values(shortcuts).map((binding) => shortcutAriaLabel(binding, useMacShortcutSymbols)).join(' ')} ArrowLeft ArrowRight`} tabIndex={0} onDragStart={preventCanvasNativeDrag} onPointerDownCapture={focusCanvasFromPointer} onPointerDown={handlePointerDown} onPointerMoveCapture={updateCanvasCrosshair} onPointerMove={handlePointerMove} onMouseMoveCapture={updateCanvasCrosshair} onMouseMove={handleMouseMove} onPointerLeave={hideCanvasGuides} onPointerUp={handlePointerUp} onPointerCancel={handlePointerUp} onDoubleClick={() => zoomAtCenter(1.5)}>
            <nav className="canvas-page-navigation" aria-label="切换图片" onDoubleClick={(event) => event.stopPropagation()}><button className="canvas-page-button previous" disabled={!canStepPrevious} aria-label={`上一张，快捷键 ${displayShortcut(shortcuts['navigation.previous'], useMacShortcutSymbols)} 或左方向键`} aria-keyshortcuts={`${shortcutAriaLabel(shortcuts['navigation.previous'], useMacShortcutSymbols)} ArrowLeft`} data-tooltip-title="上一张图片" data-tooltip="切换到上一张可标注图片；画布聚焦后可用" data-shortcut={`${displayShortcut(shortcuts['navigation.previous'], useMacShortcutSymbols)} / ←`} onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); stageRef.current?.focus(); }} onClick={(event) => { event.stopPropagation(); stepFile(-1); }}><span className="canvas-page-arrow" aria-hidden="true">‹</span></button><button className="canvas-page-button next" disabled={!canStepNext} aria-label={`下一张，快捷键 ${displayShortcut(shortcuts['navigation.next'], useMacShortcutSymbols)} 或右方向键`} aria-keyshortcuts={`${shortcutAriaLabel(shortcuts['navigation.next'], useMacShortcutSymbols)} ArrowRight`} data-tooltip-title="下一张图片" data-tooltip="切换到下一张可标注图片；画布聚焦后可用" data-shortcut={`${displayShortcut(shortcuts['navigation.next'], useMacShortcutSymbols)} / →`} onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); stageRef.current?.focus(); }} onClick={(event) => { event.stopPropagation(); stepFile(1); }}><span className="canvas-page-arrow" aria-hidden="true">›</span></button></nav>
            <div className={`pipeline-summary ${pipelineSummaryOpen ? 'open' : ''}`} onMouseEnter={() => setPipelineSummaryOpen(true)} onMouseLeave={() => setPipelineSummaryOpen(false)} onFocusCapture={() => setPipelineSummaryOpen(true)} onBlurCapture={(event) => { const next = event.relatedTarget; if (!(next instanceof Node) || !event.currentTarget.contains(next)) setPipelineSummaryOpen(false); }} onKeyDown={(event) => { if (event.key === 'Escape') { event.stopPropagation(); setPipelineSummaryOpen(false); } }} onPointerDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
              <button type="button" className={`pipeline-chip ${pipelineEnabled ? 'enabled' : ''} ${pipelineSummaryOpen ? 'summary-open' : ''}`} aria-label={pipelineEnabled ? '关闭处理流' : '开启处理流'} aria-pressed={pipelineEnabled} aria-haspopup="true" aria-expanded={pipelineSummaryOpen} aria-controls="pipeline-summary-popover" onClick={(event) => { event.stopPropagation(); changePipelineEnabled(!pipelineEnabled); }}><span className="pipeline-chip-switch" aria-hidden="true"><span /></span><b>处理流</b><small aria-live="polite"><span className="pipeline-chip-state">{pipelineEnabled ? `当前图实时 · 后台预计算${pipelineScope === 'all' ? '已开' : '已关'} · ${Math.max(0, nodes.length - 1)} 算子 · ${visualizations.length} 显示` : '已关闭'}</span><span className="pipeline-chip-action">{pipelineEnabled ? '点击关闭处理流' : '点击开启处理流'}</span></small><i aria-hidden="true">{pipelineSummaryOpen ? '⌃' : '⌄'}</i></button>
              {pipelineSummaryOpen && <section id="pipeline-summary-popover" className="pipeline-summary-popover" role="region" aria-label="处理流步骤预览"><header><strong>流程步骤</strong><small>{nodes.length} 个主节点 · {visualizations.length} 个显示 · 只读</small></header><ol>{pipelineLinearNodes.map((node) => { const isDisplay = node.kind === 'visualize'; const mainIndex = nodes.findIndex((candidate) => candidate.id === node.id); const displayIndex = visualizations.findIndex((candidate) => candidate.id === node.id); return <li key={node.id} className={`${isDisplay ? 'visualization-summary' : ''} ${node.enabled ? '' : 'disabled-node'}`}><span>{isDisplay ? `D${displayIndex + 1}` : mainIndex + 1}</span><div><b title={node.name}>{node.name}</b><small>{node.kind === 'source' ? '输入图像' : isDisplay ? '显示上游输出' : 'Image → Image'}</small></div><em>{node.kind === 'source' ? '输入' : isDisplay ? '显示' : node.enabled ? '启用' : '停用'}</em></li>; })}</ol><footer>执行顺序自上而下；顶部固定原图像，底部固定显示。</footer></section>}
            </div>
            {showPipelineViewControls && <div className="pipeline-view-controls" onPointerDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}><div role="radiogroup" aria-label="处理流显示布局"><div className={`pipeline-single-choice ${effectiveVisualizationDisplayMode === 'source' ? 'active' : ''}`} role="radio" aria-checked={effectiveVisualizationDisplayMode === 'source'}><CustomSelect className="pipeline-single-source-select" ariaLabel="单画面来源" value={singlePipelineSource} options={singlePipelineSourceOptions} onChange={(value) => { setSinglePipelineSource(value); setVisualizationDisplayMode('source'); }} /></div><button className={effectiveVisualizationDisplayMode === 'split' ? 'active' : ''} role="radio" aria-checked={effectiveVisualizationDisplayMode === 'split'} disabled={pipelineDisplaySlots.length < 2} onClick={() => setVisualizationDisplayMode('split')}>分屏</button><button className={effectiveVisualizationDisplayMode === 'overlay' ? 'active' : ''} role="radio" aria-checked={effectiveVisualizationDisplayMode === 'overlay'} disabled={!pipelineOverlayCompatibility.allowed && effectiveVisualizationDisplayMode !== 'overlay'} title={pipelineOverlayCompatibility.reason} onClick={() => setVisualizationDisplayMode('overlay')}>叠加</button></div></div>}
            {pipelineFailureDetail && <button type="button" className="pipeline-canvas-failure" role="alert" onClick={() => openRightTab('pipeline')}><span aria-hidden="true">!</span><div><strong>处理流程失败</strong><small>{pipelineFailureDetail}</small></div><i aria-hidden="true">查看 →</i></button>}
            {pipelinePreviewDirty && <div className="pipeline-stale-notice" role="status">参数已更新 · 保留上一版分屏，正在重新计算</div>}
            {pipelineImageLoadFailed && <div className="pipeline-stale-notice" role="status">处理流图像暂时不可读取 · 已回退原图，画布仍可编辑</div>}
            <div className={`canvas-hint ${polygonCloseReady ? 'close-ready' : ''}`}>{tool === 'select'
              ? '点击选择标注 · 连续点击可切换重叠或相互包含的框 · 拖动移动'
              : tool === 'brush'
              ? !annotationDraft
                ? '自由线 · 先载入可编辑标注'
                : brushPreview.length > 0
                  ? '正在绘制开放连续线 · 松开完成并选择标签'
                  : '自由线 · 按住左键拖动绘制 · 松开完成 · 不会闭合'
              : tool === 'line'
              ? drawPreview
                ? '移动指针预览 · 点击确定终点 · Esc 取消'
                : '点击确定直线起点 · 再点击确定终点'
              : tool === 'polygon'
              ? polygonDraft.length === 0
                ? '点击添加多边形顶点 · Enter / 双击完成 · Esc 取消'
                : polygonDraft.length < 3
                  ? `已添加 ${polygonDraft.length} 个点 · 至少还需 ${3 - polygonDraft.length} 个点`
                  : polygonCloseReady
                    ? '● 已吸附起点 · 点击即可闭合'
                    : '靠近起点点击闭合 · Enter / 双击也可完成'
              : '滚轮/双指平移 · 捏合或 ⌘/Ctrl + 滚轮缩放 · 聚焦画布后用 + / − / 0 / 1'}</div>
            {localServiceDown && <div className="service-banner"><span className="spinner" /><div><strong>本地服务连接中断</strong><small>保留最后成功数据；真实扫描与模型操作已暂停。</small></div></div>}
            {!showingPipelinePaneViews && <div ref={canvasCrosshairRef} className="canvas-crosshair" aria-hidden="true"><i /></div>}
            {showingPipelinePaneViews ? <div className={`pipeline-preview-surface mode-${showingSinglePipelineView ? 'single' : effectiveVisualizationDisplayMode} count-${pipelineCanvasItems.length}`} aria-label={showingSinglePipelineView ? `单画面显示 ${selectedSinglePipelineItem?.label ?? '处理结果'}` : `${pipelineDisplaySlots.length} 个同步处理流显示`}>
              {effectiveVisualizationDisplayMode !== 'overlay' ? <div className="pipeline-preview-grid">{pipelineCanvasItems.map((item, index) => <figure key={item.visualization_id} className={`pipeline-preview-pane ${item.result && readPipelinePaneMetrics(item.result).pixelGridVisible ? 'source-pixels-visible' : ''}`}><div ref={index === 0 ? imageRef : undefined} className="pipeline-preview-pane-canvas" data-pipeline-preview-pane data-pipeline-visualization-id={item.visualization_id} data-pipeline-width={item.width} data-pipeline-height={item.height} data-pipeline-label={item.label}><div className="pipeline-preview-pane-content" style={{ transform: pipelineSharedPaneTransform }}>{item.displayUrl ? <>{/* eslint-disable-next-line @next/next/no-img-element -- local pipeline artifact bypasses image optimization */}
                    <img className="pipeline-preview-image" src={item.displayUrl} alt={`${item.label} 处理流显示`} crossOrigin="anonymous" draggable={false} onError={() => { if (item.url) handlePipelineImageError(item.url); }} /></> : <div className="image-load-empty" role="status">{item.result ? '该处理流图像暂时不可读取' : '正在计算此显示…'}</div>}</div>{item.result && item.result.overlay_compatible !== false && <PipelinePixelGridCanvas imageWidth={item.width} imageHeight={item.height} view={view} referenceWidth={stageRef.current?.clientWidth ?? 840} referenceHeight={stageRef.current?.clientHeight ?? 592} enabled={showPixel} />}{item.result && item.result.overlay_compatible !== false && renderPipelinePreviewAnnotationLayer(item.result, true)}{item.result && renderPipelineSharedCrosshair(item.result, true)}</div><figcaption className="pipeline-preview-label"><strong>{item.label}</strong><span>{item.result ? `${item.width} × ${item.height}${item.result.overlay_compatible === false ? ' · 非空间特征' : ''}` : '加载中'}</span></figcaption></figure>)}</div> : <div ref={imageRef} className="pipeline-overlay-stack" data-pipeline-preview-pane data-pipeline-visualization-id={topVisiblePipelineItem?.visualization_id} data-pipeline-width={topVisiblePipelineItem?.width} data-pipeline-height={topVisiblePipelineItem?.height} data-pipeline-label={topVisiblePipelineItem?.label}><div className="pipeline-preview-pane-content" style={{ transform: pipelineSharedPaneTransform }}>{pipelineDisplayItems.map((item) => { const layer = visualizationLayerState[item.visualization_id] ?? { visible: true, opacity: 100 }; return <figure key={item.visualization_id} className={layer.visible ? 'visible' : 'hidden'} style={{ opacity: layer.visible ? layer.opacity / 100 : 0 }}>{item.displayUrl ? <>{/* eslint-disable-next-line @next/next/no-img-element -- local pipeline artifact bypasses image optimization */}
                    <img className="pipeline-preview-image" src={item.displayUrl} alt={`${item.label} 叠加层`} crossOrigin="anonymous" draggable={false} onError={() => { if (item.url) handlePipelineImageError(item.url); }} />{item.result && item.result.overlay_compatible !== false && renderPipelinePreviewAnnotationLayer(item.result)}</> : null}</figure>; })}{!pipelineDisplaySlotsReady && <div className="image-load-empty pipeline-overlay-loading" role="status">正在加载当前图的 {pipelineDisplaySlots.length} 个显示结果</div>}{topVisiblePipelineItem && renderPipelineSharedCrosshair(topVisiblePipelineItem)}</div><span className="pipeline-preview-label pipeline-overlay-label">叠加 · {pipelineDisplaySlots.length} 层{pipelineDisplaySlotsReady ? '' : ' · 加载中'}</span></div>}
            </div> : <div ref={imageRef} className={`image-surface real ${pixelGridVisible ? 'source-pixels-visible' : ''}`} style={{ transform: `translate(calc(-50% + ${view.x}px), calc(-50% + ${view.y}px)) scale(${view.scale})`, ...(displayedWidth && displayedHeight ? { aspectRatio: `${displayedWidth} / ${displayedHeight}` } : {}) }}>
              {displayedImageUrl ? <>
                {activeTileMetadata && tileAssetKey ? (
                  <TiledImage
                    key={`${tileAssetKey}:${activeTileMetadata.source_etag}`}
                    assetKey={tileAssetKey}
                    alt={currentFile?.name ?? 'Dataset image'}
                    metadata={activeTileMetadata}
                    placeholderUrl={tilePlaceholderUrl}
                    tileUrl={currentTileUrl}
                    view={view}
                    viewportRef={stageRef}
                  />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element -- local source must bypass cloud image optimization
                  <img className="real-image" src={tileMetadataLoading && tilePlaceholderUrl ? tilePlaceholderUrl : displayedImageUrl} alt={currentFile?.name ?? 'Dataset image'} crossOrigin="anonymous" draggable={false} onError={() => { if (showingPipelineImage && pipelineImageUrl) handlePipelineImageError(pipelineImageUrl); }} />
                )}
                {activeRasterUrl && activeRaster && <>
                  {/* eslint-disable-next-line @next/next/no-img-element -- local raster artifact bypasses cloud image optimization */}
                  <img className={`raster-overlay role-${activeRaster.role.toLowerCase().replace(/[^a-z0-9_-]+/g, '-')}`} src={activeRasterUrl} alt={`${activeRaster.role} raster overlay`} draggable={false} style={{ opacity: rasterOpacity / 100 }} />
                </>}
              </> : <div className="image-load-empty" role="status">当前图像暂时不可读取</div>}
            </div>}
            {displayedImageUrl && !showingPipelinePaneViews && <svg className={`real-annotation-layer screen-annotation-layer ${isSamModel && samPromptMode ? 'sam-prompt-active' : ''}`} viewBox={`0 0 ${displayedWidth ?? 1} ${displayedHeight ?? 1}`} preserveAspectRatio="none" style={annotationSvgStyle} onPointerDown={startCanvasPointer} onPointerMove={moveCanvasPointer} onPointerUp={endCanvasPointer} onPointerCancel={cancelCanvasPointer} onDoubleClick={handleCanvasDoubleClick}>
                  <defs><pattern id="source-pixel-grid" width="1" height="1" patternUnits="userSpaceOnUse"><path className="source-pixel-grid-line" d="M 1 0 H 0 V 1" /></pattern></defs>
                  {pixelGridBounds && <rect className="source-pixel-grid" x={pixelGridBounds.x} y={pixelGridBounds.y} width={pixelGridBounds.right - pixelGridBounds.x} height={pixelGridBounds.bottom - pixelGridBounds.y} fill="url(#source-pixel-grid)" data-cell-x={sourcePixelWidthOnScreen} data-cell-y={sourcePixelHeightOnScreen} aria-hidden="true" />}
                  {showGT && displayedShapes.map((shape, index) => hiddenShapeIndexes.has(index) ? null : renderRealShape(shape, index, false, 'geometry'))}
                  {!showingPipelineImage && visiblePredictionCanvasEntries.length > 0 && <g className={currentAnnotationsAreSegmentation ? 'inference-segmentation-contours' : 'inference-detections'}>{visiblePredictionCanvasEntries.map(({ prediction: shape, index }) => renderRealShape(shape as AnnotationShape, index, true, 'geometry'))}</g>}
                  {realCanvasLabelEntries.map((entry) => renderRealShape(entry.shape, entry.index, entry.prediction, 'label'))}
                  {showGT && displayedShapes.map((shape, index) => hiddenShapeIndexes.has(index) ? null : renderRealShape(shape, index, false, 'controls'))}
                  {isSamModel && samBoxes.map((box, index) => <rect key={`sam-box-${index}`} className="sam-prompt-box" x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} />)}
                  {isSamModel && samBoxPreview && <rect className="sam-prompt-box preview" x={samBoxPreview[0]} y={samBoxPreview[1]} width={samBoxPreview[2] - samBoxPreview[0]} height={samBoxPreview[3] - samBoxPreview[1]} />}
                  {isSamModel && samPoints.map((point, index) => <g key={`sam-point-${index}`} className={`sam-prompt-point ${point.label === 1 ? 'positive' : 'negative'}`}><circle cx={point.x} cy={point.y} r={Math.max(5, (displayedWidth ?? 1000) / 320)} /><text x={point.x} y={point.y} dy="0.34em" textAnchor="middle">{point.label === 1 ? '+' : '−'}</text></g>)}
                  {polygonPreviewPoints.length > 0 && <polyline className={`draw-preview freeform shape-polygon polygon-rubberband ${polygonCloseReady ? 'ready' : ''}`} points={polygonPreviewPoints.map((point) => point.join(',')).join(' ')} fill="none" />}
                  {brushPreview.length > 0 && <polyline className="draw-preview freeform shape-linestrip freehand-line-preview" points={brushPreview.map((point) => point.join(',')).join(' ')} fill="none" style={freehandGuideStyle} />}
                  {drawPreview && tool === 'line' && <line className="draw-preview shape-line" x1={drawPreview.x1} y1={drawPreview.y1} x2={drawPreview.x2} y2={drawPreview.y2} />}
                  {drawPreview && tool === 'circle' && <circle className="draw-preview shape-circle" cx={drawPreview.x1} cy={drawPreview.y1} r={Math.hypot(drawPreview.x2 - drawPreview.x1, drawPreview.y2 - drawPreview.y1)} />}
                  {drawPreview && tool === 'rect' && <rect className="draw-preview shape-rectangle" x={Math.min(drawPreview.x1, drawPreview.x2)} y={Math.min(drawPreview.y1, drawPreview.y2)} width={Math.abs(drawPreview.x2 - drawPreview.x1)} height={Math.abs(drawPreview.y2 - drawPreview.y1)} />}
                  {rotationPreviewShape && <g className="rotation-draw-preview shape-rotation"><polygon className="draw-preview shape-rotation" points={rotationPreviewShape.points.map((point) => point.join(',')).join(' ')} />{rotationPreviewCorner && <circle className="rotation-preview-corner shape-rotation" cx={rotationPreviewCorner[0]} cy={rotationPreviewCorner[1]} r={CANVAS_CONTROL_POINT_RADIUS_PX * rotationPreviewScreenUnit} />}</g>}
                  {pendingManualShape && renderPendingManualShape(pendingManualShape.shape, pendingManualShapeStyle)}
                  {polygonDraft.length > 0 && <g className={`polygon-close-target ${polygonDraft.length >= 3 ? 'available' : ''} ${polygonCloseReady ? 'ready' : ''}`}><title>{polygonDraft.length >= 3 ? '点击闭合多边形' : '至少添加 3 个点后可闭合'}</title><circle className="polygon-close-hit" cx={polygonDraft[0][0]} cy={polygonDraft[0][1]} r={14 * polygonScreenUnit} onPointerDown={(event) => { if (polygonDraftRef.current.length < 3) return; event.preventDefault(); event.stopPropagation(); finishPolygonDraft({ x: event.clientX, y: event.clientY }); }} /><circle className="polygon-close-halo" cx={polygonDraft[0][0]} cy={polygonDraft[0][1]} r={9 * polygonScreenUnit} /><circle className="polygon-close-dot" cx={polygonDraft[0][0]} cy={polygonDraft[0][1]} r={4 * polygonScreenUnit} /></g>}
            </svg>}
            {showClassifications && currentClassifications.length > 0 && <section className="canvas-classification-overlay" aria-label="当前图分类 Top-K"><header><span>分类 Top-K</span><strong>{currentClassifications[0].label}</strong></header>{currentClassifications.slice(0, 5).map((item) => <div key={`${item.rank}:${item.label}`}><b>#{item.rank}</b><span>{item.label}</span><strong>{(item.score * 100).toFixed(2)}%</strong></div>)}</section>}
            {activeRaster && activeRasterUrl && <div className="raster-overlay-controls" onPointerDown={(event) => event.stopPropagation()}><div><span>像素结果</span><strong>{activeRaster.role}</strong></div><label><span>透明度</span><input type="range" min="0" max="100" value={rasterOpacity} onChange={(event) => setRasterOpacity(Number(event.target.value))} /><b>{rasterOpacity}%</b></label><button onClick={() => setShowMasks(false)}>关闭</button></div>}
            <div className="navigator"><div ref={navigatorImageRef} className="navigator-image image" role="application" aria-label="画布导航器，点击或拖动定位" onPointerDown={startNavigatorDrag} onPointerMove={moveNavigatorDrag} onPointerUp={endNavigatorDrag} onPointerCancel={endNavigatorDrag} style={dataset.id && currentFile ? { backgroundImage: `url(${backend.assetUrl(dataset.id, currentFile.id, 'thumbnail', '?max_size=256')})`, backgroundSize: 'contain', backgroundPosition: 'center', backgroundRepeat: 'no-repeat' } : undefined}><span className="viewport-box" style={{ left: navigatorBox.left, top: navigatorBox.top, width: navigatorBox.width, height: navigatorBox.height }} /></div><span>导航器 · 点击或拖动定位</span></div>
          </div>

          <footer className="statusbar"><div className="statusbar-group statusbar-context"><span className="statusbar-source">{displayedWidth ?? 0} × {displayedHeight ?? 0}{showingPipelineImage ? ' · 处理流底图' : activeTileMetadata ? ` · ${activeTileMetadata.backend === 'pyvips' ? 'libvips' : 'Pillow'} 金字塔瓦片` : ' · 真实图像'}</span><span className="healthy">● {backend.pipeline.phase === 'loading' ? '正在执行处理流' : tileMetadataLoading ? '正在准备大图瓦片' : backend.annotation.phase === 'loading' ? '正在加载标注' : annotationDraft ? '标签已载入' : '画布就绪'}</span></div><div className="statusbar-group statusbar-pixels">{pipelineSharedCursor && <span className="pipeline-active-display" title="当前像素读数来源">{pipelineSharedCursor.label}</span>}<PixelReadout cursor={cursor} /></div><div className="statusbar-group statusbar-actions"><label className={`statusbar-grid-toggle ${showPixel ? 'active' : ''} ${pixelGridVisible ? 'visible' : ''}`} title={!showPixel ? '点击开启真实像素网格' : pixelGridVisible ? `真实像素网格正在显示 · 每格对应 1 个源像素 · 当前 ${sourcePixelScreenMinimum.toFixed(1)}px/像素` : pixelGridRevealScale ? `真实像素网格已启用 · 放大到约 ${Math.ceil(pixelGridRevealScale * 100)}% 后显示` : '真实像素网格已启用 · 图像尺寸就绪后显示'}><input type="checkbox" aria-label="显示真实像素网格" checked={showPixel} onChange={(event) => setShowPixel(event.target.checked)} /><span className="statusbar-grid-check" aria-hidden="true" /><span>{pixelGridVisible ? '真实网格已显示' : '真实网格待显示'}</span></label><button type="button" className={`statusbar-autosave ${annotationAutoSave ? 'active' : ''} ${annotationDirty ? 'dirty' : ''} ${annotationPersistence.phase === 'error' ? 'error' : ''}`} role="switch" aria-label="当前图自动保存" aria-describedby="current-image-save-status" aria-checked={annotationAutoSave} disabled={annotationSaving} title={saveModeTitle} onClick={() => changeAnnotationAutoSave(!annotationAutoSave)}><span>当前图自动保存</span><b id="current-image-save-status" aria-live="polite">{saveModeLabel}</b><i aria-hidden="true"><span /></i></button></div></footer>
        </section>

        <aside className={`sidebar right-sidebar ${rightTab}-tab`}>
          <div className="right-tabs">{([['layers', '对象'], ['pipeline', '处理流'], ['inference', '推理'], ['agent', 'Agent']] as [RightTab, string][]).map(([id, label]) => <button key={id} className={`${rightTab === id ? 'active' : ''} ${id === 'agent' && !agentBackendReady ? 'agent-unavailable' : ''}`} title={id === 'agent' && !agentBackendReady ? 'Agent 后端未配置' : undefined} onClick={() => openRightTab(id)}>{label}</button>)}</div>
          <div ref={rightSidebarRef} className={`right-panel-scroll tab-${rightTab}`}>
          {rightTab === 'layers' && <div className="layers-panel">
            <section className="side-section annotation-category-panel" aria-labelledby="annotation-category-list-title">
              <div className="annotation-category-header"><div><span className="eyebrow">当前图类别</span><h2 id="annotation-category-list-title">类别 <b>{annotationCategories.length + aiPredictionCategories.length}</b></h2><small>人工 {annotationCategories.length} 类 · AI 预测 {aiPredictionCategories.length} 类，分组管理</small></div></div>
              <div className="object-source-tabs" role="tablist" aria-label="对象来源"><button type="button" role="tab" aria-selected={objectSourceTab === 'manual'} className={objectSourceTab === 'manual' ? 'active' : ''} onClick={() => { setObjectSourceTab('manual'); setSelectedPredictionIndex(null); }}>人工标注 <b>{objectListShapes.length}</b></button><button type="button" role="tab" aria-selected={objectSourceTab === 'ai'} className={objectSourceTab === 'ai' ? 'active' : ''} disabled={availablePredictionEntries.length === 0} onClick={() => { setObjectSourceTab('ai'); setSelectedShapeIndex(null); }}>AI 预测 <b>{availablePredictionEntries.length}</b></button></div>
              <div className={`annotation-category-groups source-${objectSourceTab}`}>
              {objectSourceTab === 'manual' && (annotationCategories.length ? <div className="annotation-category-list" role="list" aria-label="当前图类别">
                {annotationCategories.map((category) => {
                  const categoryIndexes = annotationIndexesForCategory(normalizedAnnotationObjectLabels, category);
                  const categoryVisibleCount = categoryIndexes.reduce((count, index) => count + (hiddenShapeIndexes.has(index) ? 0 : 1), 0);
                  const categoryAllVisible = categoryVisibleCount === categoryIndexes.length;
                  const categoryVisibilityState = categoryVisibleCount === 0 ? false : categoryAllVisible ? true : 'mixed';
                  const categoryColors = annotationCategoryColors(category, Object.hasOwn(annotationCategoryColorOverrides, category) ? annotationCategoryColorOverrides[category] : undefined);
                  const editingCategory = pendingCategoryLabelEdit?.category === category;
                  return <div key={category} className={`annotation-category-row ${categoryVisibleCount === 0 ? 'hidden-category' : ''}`} role="listitem" style={annotationCategoryStyle(category, annotationCategoryColorOverrides)}>
                    <div className="annotation-category-icon">
                      <label className="annotation-category-color" title={`设置「${category}」的标注颜色`}><input type="color" disabled={editingCategory || categoryRenameCreating} value={categoryColors.stroke} aria-label={`设置类别 ${category} 的颜色`} onChange={(event) => setAnnotationCategoryColor(category, event.target.value)} /><span aria-hidden="true" /></label>
                      <button className={`annotation-category-visibility ${categoryVisibleCount > 0 ? 'on' : ''}`} disabled={editingCategory || categoryRenameCreating} role="checkbox" aria-checked={categoryVisibilityState} aria-label={`${categoryAllVisible ? '隐藏' : '显示'}类别 ${category} 的全部 ${categoryIndexes.length} 个标注框`} title={categoryAllVisible ? '隐藏该类别的全部框' : '显示该类别的全部框'} onClick={() => changeAnnotationIndexesVisibility(categoryIndexes, !categoryAllVisible)}><VisibilityEyeIcon visible={categoryVisibleCount > 0} /></button>
                    </div>
                    {editingCategory ? <form className="annotation-inline-label-editor batch" onSubmit={(event) => { event.preventDefault(); void commitPendingCategoryLabelEdit(); }} onBlur={(event) => { if (!categoryRenameCreating && !event.currentTarget.contains(event.relatedTarget)) closePendingLabelEditor(); }}><input autoFocus aria-label={`全数据集重命名类别 ${category}`} value={manualShapeLabel} maxLength={128} onFocus={(event) => event.currentTarget.select()} onChange={(event) => setManualShapeLabel(event.target.value)} onKeyDown={(event) => { event.stopPropagation(); if (!event.isComposing && event.key === 'Escape') { event.preventDefault(); closePendingLabelEditor(); } }} /><small>{categoryRenameCreating ? '正在创建全数据集任务…' : '全数据集 · 将修改所有图片中的同名类别；已有名称会合并'}</small></form> : <button type="button" className="annotation-category-main" disabled={!annotationDraft || annotationSaving || categoryRenameCreating || Boolean(activeCategoryRenameJob)} aria-label={`类别 ${category}，${categoryIndexes.length} 个框；双击或按 ${displayShortcut(shortcuts['edit.changeCategory'], useMacShortcutSymbols)} 修改名称`} aria-keyshortcuts={shortcutAriaLabel(shortcuts['edit.changeCategory'], useMacShortcutSymbols)} data-tooltip-title="修改类别" data-tooltip={activeCategoryRenameJob ? '当前数据集已有类别重命名任务在运行' : '双击或使用快捷键修改全数据集同名类别'} data-shortcut={displayShortcut(shortcuts['edit.changeCategory'], useMacShortcutSymbols)} title={!annotationDraft ? '标注尚未载入' : undefined} onClick={(event) => { if (event.detail === 0) openCategoryLabelEditor(category, categoryIndexes.length); }} onDoubleClick={(event) => { event.preventDefault(); event.stopPropagation(); openCategoryLabelEditor(category, categoryIndexes.length); }} onKeyDown={(event) => { if (resolveShortcutAction(event.nativeEvent, shortcuts, 'app') === 'edit.changeCategory') { event.preventDefault(); event.stopPropagation(); openCategoryLabelEditor(category, categoryIndexes.length); } }}><strong>{category}</strong><small>{categoryIndexes.length} 个框</small></button>}
                    <button className="annotation-category-delete" disabled={!annotationDraft || annotationSaving || editingCategory || categoryRenameCreating} aria-label={`删除当前图类别 ${category} 及其 ${categoryIndexes.length} 个标注框`} title={!annotationDraft ? '标注尚未载入' : `删除当前图「${category}」及 ${categoryIndexes.length} 个框 · 可撤销`} onClick={() => deleteAnnotationCategory(category)}>×</button>
                  </div>;
                })}
              </div> : <div className="annotation-category-empty">当前图还没有人工标注类别</div>)}
              {objectSourceTab === 'ai' && (aiPredictionCategoryRows.length > 0 ? <div className="annotation-category-list prediction-category-list" role="list" aria-label="当前图 AI 预测类别">{aiPredictionCategoryRows.map((category) => {
                const categoryAllVisible = category.visibleCount === category.entries.length;
                const categoryVisibilityState = category.visibleCount === 0 ? false : categoryAllVisible ? true : 'mixed';
                return <div key={`ai-category-${category.label}`} className={`annotation-category-row prediction-category-row ${category.visibleCount === 0 ? 'hidden-category' : ''}`} role="listitem" style={annotationCategoryStyle(category.label, annotationCategoryColorOverrides)}>
                  <div className="annotation-category-icon"><span className="annotation-category-color prediction-readonly-color" title={`AI 预测类别 ${category.label}`} aria-hidden="true" /><button type="button" className={`annotation-category-visibility ${category.visibleCount > 0 ? 'on' : ''}`} role="checkbox" aria-checked={categoryVisibilityState} aria-label={`${categoryAllVisible ? '隐藏' : '显示'} AI 预测类别 ${category.label} 的全部 ${category.entries.length} 个框`} onClick={() => { setHiddenPredictionCategories((current) => { const next = new Set(current); if (categoryAllVisible) next.add(category.label); else next.delete(category.label); return next; }); setHiddenPredictionKeys((current) => { const next = new Set(current); for (const entry of category.entries) { if (categoryAllVisible) next.add(entry.key); else next.delete(entry.key); } return next; }); }}><VisibilityEyeIcon visible={category.visibleCount > 0} /></button></div>
                  <div className="annotation-category-main prediction-category-main"><strong title={category.label}>{category.label}</strong><small>{category.count} 个框 · 最高 {(category.maxScore * 100).toFixed(1)}%</small></div>
                  <span className="annotation-category-source-badge" title="只读 AI 预测类别">AI</span>
                </div>;
              })}</div> : <div className="annotation-category-empty">当前图没有 AI 预测类别</div>)}
              </div>
            </section>
            <section className={`side-section annotation-object-panel source-${objectSourceTab}`} aria-labelledby="annotation-object-list-title">
              <div className="annotation-list-header"><div><span className="eyebrow">当前图对象</span><h2 id="annotation-object-list-title">标注框 <b>{objectSourceTab === 'manual' ? objectListShapes.length : availablePredictionEntries.length}</b></h2><small aria-live="polite">{objectSourceTab === 'manual' ? showGT ? `${visibleAnnotationObjectCount}/${objectListShapes.length} 个画布显示` : `0/${objectListShapes.length} 个画布显示` : `${visibleCurrentPredictionEntries.length}/${availablePredictionEntries.length} 个 AI 框画布显示`}</small></div>{objectSourceTab === 'manual' ? <div className="annotation-list-tools"><button className={`annotation-master-visibility ${showGT ? 'on' : ''}`} role="switch" aria-checked={showGT} aria-label={showGT ? '隐藏画布中的全部标注' : '显示画布中的全部标注'} title={showGT ? '画布标注已显示；点击隐藏' : '画布标注已隐藏；点击显示'} onClick={() => setShowGT((visible) => !visible)}><VisibilityEyeIcon visible={showGT} /></button><div className="annotation-list-actions"><button disabled={!annotationObjectIndexes.length || hiddenAnnotationObjectCount === 0} onClick={() => changeAnnotationIndexesVisibility(annotationObjectIndexes, true)}>全开</button><button disabled={!annotationObjectIndexes.length || visibleAnnotationObjectCount === 0} onClick={() => changeAnnotationIndexesVisibility(annotationObjectIndexes, false)}>全关</button></div></div> : <div className="annotation-list-tools"><button className={`annotation-master-visibility ${allAiPredictionObjectsVisible ? 'on' : ''}`} role="switch" aria-checked={allAiPredictionObjectsVisible} aria-label={allAiPredictionObjectsVisible ? '隐藏画布中的全部 AI 预测框' : '显示画布中的全部 AI 预测框'} onClick={() => { if (allAiPredictionObjectsVisible) setHiddenPredictionKeys(new Set(availablePredictionEntries.map((entry) => entry.key))); else { setHiddenPredictionCategories(new Set()); setHiddenPredictionKeys(new Set()); } }}><VisibilityEyeIcon visible={visibleCurrentPredictionEntries.length > 0} /></button><div className="annotation-list-actions"><button disabled={allAiPredictionObjectsVisible} onClick={() => { setHiddenPredictionCategories(new Set()); setHiddenPredictionKeys(new Set()); }}>全开</button><button disabled={visibleCurrentPredictionEntries.length === 0} onClick={() => setHiddenPredictionKeys(new Set(availablePredictionEntries.map((entry) => entry.key)))}>全关</button></div></div>}</div>
              {objectSourceTab === 'ai' && (availablePredictionEntries.length ? <div className="annotation-object-list prediction-object-list" role="list" aria-label="当前图 AI 预测框">{availablePredictionEntries.map(({ prediction, index, key, label }) => { const visible = !hiddenPredictionCategories.has(label) && !hiddenPredictionKeys.has(key); const selected = selectedPredictionIndex === index; return <div key={`ai-object-${key}`} className={`annotation-object-row prediction-object-row ${selected ? 'selected' : ''} ${visible ? '' : 'hidden-object'}`} role="listitem" data-prediction-index={index} aria-current={selected ? 'true' : undefined} style={annotationCategoryStyle(label, annotationCategoryColorOverrides)}><div className="annotation-object-icon"><span className={`annotation-object-shape ${annotationShapeClass(prediction.shape_type)}`} title={`${shapeTypeLabels[prediction.shape_type] ?? prediction.shape_type} · AI ${(prediction.score * 100).toFixed(1)}%`} aria-hidden="true"><ShapeTypeIcon shapeType={prediction.shape_type} /></span><button type="button" className={`annotation-object-visibility ${visible ? 'on' : ''}`} role="switch" aria-checked={visible} aria-label={`${visible ? '隐藏' : '显示'} AI 预测框 ${index + 1}：${label}`} onClick={() => { if (visible) setHiddenPredictionKeys((current) => new Set(current).add(key)); else { setHiddenPredictionCategories((current) => { const next = new Set(current); next.delete(label); return next; }); setHiddenPredictionKeys((current) => { const next = new Set(current); next.delete(key); return next; }); } }}><VisibilityEyeIcon visible={visible} /></button></div><button type="button" className="annotation-object-main" aria-label={`${label}，AI 预测框 ${index + 1}，置信度 ${(prediction.score * 100).toFixed(1)}%`} onClick={() => { setSelectedShapeIndex(null); setSelectedPredictionIndex(index); }}><small className="annotation-object-index">AI #{String(index + 1).padStart(2, '0')}</small><strong title={`${label} · ${(prediction.score * 100).toFixed(1)}%`}>{label}</strong></button><button type="button" className="prediction-promote-button" aria-label={`将 AI 预测框 ${index + 1}：${label} 转为人工标注框`} title="转为人工标注框" onClick={() => promotePredictionToManual(index)}>✓</button></div>; })}</div> : <div className="annotation-list-empty"><span>◇</span><strong>当前图没有 AI 预测框</strong><small>运行检测模型后会以标注框形式显示在这里。</small></div>)}
              {objectSourceTab === 'manual' && (objectListShapes.length ? <div ref={annotationObjectListRef} className={`annotation-object-list ${!showGT ? 'master-hidden' : ''}`} role="list" aria-label="当前图全部标注框">
                {objectListShapes.map((shape, index) => {
                  const visible = !hiddenShapeIndexes.has(index);
                  const rowSelected = selectedShapeIndex === index;
                  const label = shape.label?.trim() || '未命名';
                  const editingShape = pendingShapeLabelEdit?.index === index;
                  return <div key={`annotation-object-${index}`} className={`annotation-object-row ${rowSelected ? 'selected' : ''} ${visible ? '' : 'hidden-object'}`} role="listitem" data-shape-index={index} aria-current={rowSelected ? 'true' : undefined} style={annotationCategoryStyle(shape.label, annotationCategoryColorOverrides)}>
                    <div className="annotation-object-icon">
                      <span className={`annotation-object-shape ${annotationShapeClass(shape.shape_type)}`} title={shapeTypeLabels[shape.shape_type] ?? shape.shape_type} aria-hidden="true"><ShapeTypeIcon shapeType={shape.shape_type} /></span>
                      <button className={`annotation-object-visibility ${visible ? 'on' : ''}`} disabled={editingShape} role="switch" aria-checked={visible} aria-label={`${visible ? '隐藏' : '显示'}标注框 ${index + 1}：${label}`} title={visible ? '画布中已显示；点击隐藏' : '画布中已隐藏；点击显示'} onClick={() => changeShapeVisibility(index, !visible)}><VisibilityEyeIcon visible={visible} /></button>
                    </div>
                    {editingShape ? <form className="annotation-inline-label-editor object" onSubmit={(event) => { event.preventDefault(); commitPendingShapeLabelEdit(); }} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) closePendingLabelEditor(); }}><input autoFocus aria-label={`修改标注框 ${index + 1} 的类别`} value={manualShapeLabel} maxLength={128} onFocus={(event) => event.currentTarget.select()} onChange={(event) => setManualShapeLabel(event.target.value)} onKeyDown={(event) => { event.stopPropagation(); if (!event.isComposing && event.key === 'Escape') { event.preventDefault(); closePendingLabelEditor(); } }} /><small>仅修改此标注框 · Enter 确认 · Esc 取消</small></form> : <button className="annotation-object-main" aria-label={`${label}，标注 ${index + 1}，${shapeTypeLabels[shape.shape_type] ?? shape.shape_type}`} aria-keyshortcuts={annotationDraft ? shortcutAriaLabel(shortcuts['edit.changeCategory'], useMacShortcutSymbols) : undefined} data-tooltip-title={annotationDraft ? '修改对象类别' : undefined} data-tooltip={annotationDraft ? '先选中对象；双击或使用快捷键修改类别' : undefined} data-shortcut={annotationDraft ? displayShortcut(shortcuts['edit.changeCategory'], useMacShortcutSymbols) : undefined} title={annotationDraft ? undefined : `选择 ${label}`} onClick={() => selectShapeFromList(shape, index)} onDoubleClick={(event) => { event.preventDefault(); event.stopPropagation(); openShapeLabelEditor(index); }} onKeyDown={(event) => { if (annotationDraft && resolveShortcutAction(event.nativeEvent, shortcuts, 'app') === 'edit.changeCategory') { event.preventDefault(); event.stopPropagation(); openShapeLabelEditor(index); } }}><small className="annotation-object-index" aria-label={`标注编号 ${index + 1}`}>#{String(index + 1).padStart(2, '0')}</small><strong title={label}>{label}</strong></button>}
                    <button className="shape-delete-button" disabled={!annotationDraft || annotationSaving || editingShape} aria-label={`删除标注框 ${index + 1}：${label}`} title={!annotationDraft ? '标注尚未载入' : `删除 ${label} · 可撤销`} onClick={() => deleteShapeAtIndex(index)}>×</button>
                  </div>;
                })}
              </div> : <div className="annotation-list-empty"><span>□</span><strong>当前图没有标注框</strong><small>使用画布工具新建后会立即出现在这里。</small></div>)}
            </section>
            <section className="side-section object-card">
              <span className="eyebrow">{objectSourceTab === 'ai' ? `AI 预测 · ${availablePredictionEntries.length} 个只读框` : annotationDraft ? `真实标注 · ${draftShapes.length} 对象 · ${annotationDirty ? '未保存' : '已同步'}` : '标注尚未载入'}</span>
              <div className="object-title">{objectSourceTab === 'manual' && selectedDraftShape ? <button className="object-category-button" onClick={openSelectedShapeLabelEditor} aria-label={`选择对象类别，当前为 ${selectedDraftShape.label}`}><small>类别</small><strong>{selectedDraftShape.label}</strong><i>⌄</i></button> : <strong>{selectedPanelShape?.label ?? '未选择'}</strong>}<span>{selectedPanelShape?.shape_type ?? selectedShape}</span></div>
              {selectedPredictionShape && selectedPredictionIndex !== null && <button type="button" className="promote-selected-prediction" onClick={() => promotePredictionToManual(selectedPredictionIndex)}>确认此 AI 框 · 转为人工框</button>}
              <div className="property-grid"><span>X1 <strong>{Math.round(selectedPanelShape?.points[0]?.[0] ?? 0).toLocaleString()}</strong></span><span>Y1 <strong>{Math.round(selectedPanelShape?.points[0]?.[1] ?? 0).toLocaleString()}</strong></span><span>点数 <strong>{selectedPanelShape?.points.length ?? 0}</strong></span><span>类型 <strong>{selectedPanelShape?.shape_type ?? selectedShape}</strong></span>{selectedPredictionShape && <span>置信度 <strong>{(selectedPredictionShape.score * 100).toFixed(1)}%</strong></span>}{selectedPanelShape?.shape_type === 'rotation' && <><span className="angle-property">弧度 <strong>{Number(selectedPanelShape.direction ?? 0).toFixed(3)}</strong></span><span>角度 <strong>{Number((selectedPanelShape.direction ?? 0) * 180 / Math.PI).toFixed(1)}°</strong></span></>}</div>
              <label className="opacity-control object-opacity-control"><span>标注透明度</span><input type="range" min="20" max="100" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /><b>{opacity}%</b></label>
              {objectSourceTab === 'manual' && annotationDraft && <div className="annotation-history-actions"><button disabled={!canUndoAnnotation || annotationSaving} aria-keyshortcuts={shortcutAriaLabel(shortcuts['edit.undo'], useMacShortcutSymbols)} data-tooltip-title="撤销" data-tooltip="撤销上一次标注修改" data-shortcut={displayShortcut(shortcuts['edit.undo'], useMacShortcutSymbols)} onClick={undoAnnotation}>↶ 撤销 <kbd>{displayShortcut(shortcuts['edit.undo'], useMacShortcutSymbols)}</kbd></button><button disabled={!canRedoAnnotation || annotationSaving} aria-keyshortcuts={shortcutAriaLabel(shortcuts['edit.redo'], useMacShortcutSymbols)} data-tooltip-title="重做" data-tooltip="恢复上一次撤销的修改" data-shortcut={displayShortcut(shortcuts['edit.redo'], useMacShortcutSymbols)} onClick={redoAnnotation}>↷ 重做 <kbd>{displayShortcut(shortcuts['edit.redo'], useMacShortcutSymbols)}</kbd></button></div>}
            </section>
            {objectSourceTab === 'manual' && annotationDraft && selectedDraftShape?.shape_type === 'rotation' && <section className="real-rotation-editor"><span>旋转框角度</span><div><button onClick={() => rotateSelectedShape(-1)}>−1°</button><button onClick={() => rotateSelectedShape(-0.1)}>−0.1°</button><button onClick={() => rotateSelectedShape(0.1)}>＋0.1°</button><button onClick={() => rotateSelectedShape(1)}>＋1°</button></div><small>图像坐标系，顺时针为正；保存时由四点重新计算 direction。</small></section>}
          </div>}

          {rightTab === 'pipeline' && <section className="pipeline-panel">
            <div className="panel-intro pipeline-intro"><div><span className="eyebrow">处理数据流</span><h2>{pipelineEnabled ? '流程已开启' : '流程已关闭'}</h2><p>每个节点严格遵循 Image → Image，并声明标注变换。</p></div><label className="switch"><input type="checkbox" checked={pipelineEnabled} onChange={(event) => changePipelineEnabled(event.target.checked)} /><span /></label></div>
            <div className="pipeline-background-toggle"><div><strong>后台预计算全部图像</strong><small>关闭时仍实时处理当前图</small></div><label className="switch" title="当前图始终实时处理；此开关只控制是否在后台预计算全库"><input type="checkbox" aria-label="后台预计算全部图像" checked={pipelineScope === 'all'} onChange={(event) => changePipelineScope(event.target.checked ? 'all' : 'current')} /><span /></label></div>
            {pipelineFailureDetail && <div className="pipeline-editor-failure" role="alert"><span aria-hidden="true">!</span><div><strong>处理流程失败</strong><p>{pipelineFailureDetail}</p></div>{pipelineFailureNeedsModel && pipelineFeatureNode && <button type="button" onClick={() => openInferenceFeatureConfiguration(pipelineFeatureNode)}>前往推理加载模型</button>}</div>}
            <div className="flow-canvas" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); addNode(event.dataTransfer.getData('text/plain'), pipelineGaps.at(-1)); }}>
              <div className="flow-sequence">{pipelineLinearNodes.map((node, sequenceIndex) => {
                const isVisualization = node.kind === 'visualize';
                const nodeContract = pipelineContracts.find((contract) => contract.kind === node.kind);
                const nodeTiming = pipelineTimingByNode[node.id] ?? pipelineTimingByNode[node.kind];
                const nodeTimingText = formatPipelineTiming(nodeTiming?.milliseconds, nodeTiming?.samples);
                const nodeTimingStatus = backend.pipeline.phase === 'loading' ? nodeTimingText ?? '计算中' : nodeTimingText;
                const gap = pipelineGaps[sequenceIndex];
                const displayIndex = isVisualization ? visualizations.findIndex((candidate) => candidate.id === node.id) : -1;
                const mainIndex = isVisualization ? -1 : nodes.findIndex((candidate) => candidate.id === node.id);
                const visualizationDisabledByNeighbor = Boolean(gap && !gap.visualizationTapAfterNodeId);
                const visualizationDisabled = visualizations.length >= MAX_PIPELINE_VISUALIZATIONS || visualizationDisabledByNeighbor;
                const visualizationTitle = visualizationDisabledByNeighbor
                  ? '相邻节点已有显示，不能连续添加显示'
                  : visualizations.length >= MAX_PIPELINE_VISUALIZATIONS
                    ? `最多 ${MAX_PIPELINE_VISUALIZATIONS} 个显示节点`
                    : `显示 ${node.name} 的结果`;
                return <section key={node.id} className="flow-stage">
                  <div className="flow-stage-row">
                    <div className={`flow-node ${node.kind === 'source' ? 'source-node' : isVisualization ? 'visualization-node output-node' : 'operator-node'} ${selectedNode === node.id ? 'selected' : ''} ${!node.enabled ? 'disabled-node' : ''}`} onClick={() => setSelectedNode(node.id)}>
                      <span className="node-index">{isVisualization ? `D${displayIndex + 1}` : mainIndex + 1}</span><div className="node-copy"><strong>{node.name}</strong><small>{node.kind === 'source' ? '固定原图输入' : isVisualization ? '显示上游节点的输出' : `图像变换 · ${nodeContract?.size_behavior ?? '尺寸未知'}`}{isVisualization && node.parameters.label && node.parameters.label !== node.name ? ` · 结果 ${String(node.parameters.label)}` : ''}{nodeTimingStatus ? ` · ${nodeTimingStatus}` : ''}</small></div>{isVisualization ? <button className="node-delete" disabled={visualizations.length <= 1} title={visualizations.length <= 1 ? '至少保留一个显示节点' : `删除 ${node.name}`} aria-label={`删除 ${node.name}`} onClick={(event) => { event.stopPropagation(); deleteVisualization(node.id); }}>×</button> : node.kind !== 'source' ? <button className="node-delete" aria-label={`删除 ${node.name}`} title={`删除 ${node.name}`} onClick={(event) => { event.stopPropagation(); deleteNode(node.id); }}>×</button> : <span className="node-delete-space" aria-hidden="true" />}
                    </div>
                  </div>
                  {gap && <div className={`flow-insert-connector ${pipelineInsertGapKey === gap.key ? 'open' : ''}`}>
                    <button aria-label={`在 ${gap.upstream.name} 和 ${gap.downstream.name} 之间添加算子`} aria-haspopup="dialog" aria-expanded={pipelineInsertGapKey === gap.key} aria-controls={pipelineInsertGapKey === gap.key ? `pipeline-insert-${sequenceIndex}` : undefined} onClick={(event) => { const opening = pipelineInsertGapKey !== gap.key; pipelineInsertAnchorRef.current = opening ? event.currentTarget : null; if (opening) setOperatorSearch(''); setPipelineInsertGapKey(opening ? gap.key : null); }}>＋</button>
                    {pipelineInsertGapKey === gap.key && <PipelineInsertPopover id={`pipeline-insert-${sequenceIndex}`} anchor={pipelineInsertAnchorRef.current} search={operatorSearch} operators={filteredPipelinePalette} showVisualization={visualizationMatchesOperatorSearch} visualizationDisabled={visualizationDisabled} visualizationName={visualizationName} visualizationTitle={visualizationTitle} visualizationSourceName={gap.upstream.name} onSearchChange={setOperatorSearch} onAddOperator={(kind) => addNode(kind, gap)} onAddVisualization={() => { if (gap.visualizationTapAfterNodeId) addVisualization(gap.visualizationTapAfterNodeId); }} onClose={() => setPipelineInsertGapKey(null)} />}
                  </div>}
                </section>;
              })}</div>
              <span className={`pipeline-validation-indicator ${pipelineValidationIndicatorState}`} role={pipelineValidationIndicatorState === 'invalid' ? 'alert' : 'status'} aria-live="polite" aria-label={`处理流校验：${pipelineValidationIndicatorText}`} tabIndex={0} data-tooltip-title="处理流校验" data-tooltip={pipelineValidationIndicatorText}>{pipelineValidationIndicatorIcon}</span>
            </div>
            <PipelineParameterEditor node={selectedOperator} contract={selectedOperatorContract} registryHash={backend.pipelineRegistry.data?.registry_hash} onChange={(name, value) => updateNodeParameter(selectedOperator.id, name, value)} onEnabledChange={selectedOperator.kind === 'source' || selectedVisualization ? undefined : (enabled) => setNodes((old) => old.map((item) => item.id === selectedOperator.id ? { ...item, enabled } : item))} models={displayedModelCatalog} featureLayers={pipelineFeatureRuntimeLayers} featureRuntimeModelId={backend.runtime.data?.model_id} onOpenInference={selectedOperator.kind === 'model_feature' ? () => openInferenceFeatureConfiguration(selectedOperator) : undefined} inputWidth={selectedOperatorInputTransform?.width} inputHeight={selectedOperatorInputTransform?.height} />
            {showPipelineViewControls && <section className={`visualization-layer-controls ${effectiveVisualizationDisplayMode}`}><header><div><span className="eyebrow">多显示布局</span><strong title={effectiveVisualizationDisplayMode === 'overlay' ? '列表按实际遮挡顺序排列；顶层为 100% 时会完全遮住下面的图层' : undefined}>{effectiveVisualizationDisplayMode === 'source' ? singlePipelineSource === 'source' ? '单画面 · 原图' : `单画面 · ${selectedSinglePipelineItem?.label ?? '等待结果'}` : effectiveVisualizationDisplayMode === 'overlay' ? `叠加 · ${pipelineDisplaySlots.length} 层 · 顶层优先` : `${pipelineDisplaySlots.length} 个同步分屏`}</strong></div><div><button className={effectiveVisualizationDisplayMode === 'source' ? 'active' : ''} aria-pressed={effectiveVisualizationDisplayMode === 'source'} onClick={() => setVisualizationDisplayMode('source')}>单画面</button><button className={effectiveVisualizationDisplayMode === 'split' ? 'active' : ''} aria-pressed={effectiveVisualizationDisplayMode === 'split'} disabled={pipelineDisplaySlots.length < 2} onClick={() => setVisualizationDisplayMode('split')}>分屏</button><button className={effectiveVisualizationDisplayMode === 'overlay' ? 'active' : ''} aria-pressed={effectiveVisualizationDisplayMode === 'overlay'} disabled={!pipelineOverlayCompatibility.allowed && effectiveVisualizationDisplayMode !== 'overlay'} title={pipelineOverlayCompatibility.reason} onClick={() => setVisualizationDisplayMode('overlay')}>叠加</button></div></header>{effectiveVisualizationDisplayMode !== 'overlay' && <p className={pipelineOverlayCompatibility.allowed ? 'ready' : 'blocked'}>{effectiveVisualizationDisplayMode === 'source' ? '单画面来源可在中心画布顶部选择；处理流和后台预计算不会暂停。' : pipelineOverlayCompatibility.allowed ? '结果尺寸一致，可切换为像素对齐叠加。' : `${pipelineOverlayCompatibility.reason}；继续使用同步分屏。`}</p>}{effectiveVisualizationDisplayMode === 'overlay' && <div className="visualization-alpha-mixer" role="group" aria-label="叠加图层透明度，从顶层到底层排列">{pipelineDisplayItems.slice().reverse().map((item, stackIndex) => { const layer = visualizationLayerState[item.visualization_id] ?? { visible: true, opacity: 100 }; const canHide = canHidePipelineLayer(item.visualization_id, visualizationLayerState); const topLayer = stackIndex === 0; const labelId = `visualization-alpha-${item.visualization_id}`; return <div className={`visualization-alpha-row ${layer.visible ? 'visible' : 'hidden'} ${topLayer ? 'top-layer' : ''}`} key={item.visualization_id}><button type="button" className="visualization-alpha-visibility" role="switch" aria-checked={layer.visible} aria-label={`${layer.visible ? '隐藏' : '显示'} ${item.label}`} disabled={layer.visible && !canHide} title={layer.visible && !canHide ? '至少保留一个可见层' : `${layer.visible ? '隐藏' : '显示'} ${item.label}`} onClick={() => { if (!layer.visible || canHide) setVisualizationLayerVisible(item.visualization_id, !layer.visible); }}><VisibilityEyeIcon visible={layer.visible} /></button><span id={labelId} className="visualization-alpha-name"><b>{topLayer ? '顶' : `D${pipelineDisplayItems.length - stackIndex}`}</b><strong title={item.label}>{item.label}{item.result ? '' : ' · 加载中'}</strong></span><input type="range" min="0" max="100" step="1" disabled={!layer.visible} aria-labelledby={labelId} aria-valuetext={`${layer.opacity}%`} value={layer.opacity} onChange={(event) => setVisualizationLayerOpacity(item.visualization_id, Number(event.currentTarget.value))} /><output>{layer.opacity}%</output></div>; })}</div>}</section>}
          </section>}

          {rightTab === 'inference' && <section className="inference-panel">
            <div className="panel-intro model-intro"><div><span className="eyebrow">模型推理</span><h2>模型</h2><p>{backend.mode === 'online' ? `${displayedModelCatalog.length} 个可选模型 · ${backend.health.data?.model_registry.adapters ?? 0} 个可预测适配器` : '本地服务离线，模型目录暂不可用'}</p></div></div>
            <section className={`inference-model-browser compact ${modelActionKind}`} aria-labelledby="inference-model-browser-title"><div className="inference-model-current"><button ref={modelPickerTriggerRef} type="button" className="inference-model-trigger" aria-haspopup="dialog" aria-expanded={modelPickerOpen} onClick={toggleModelPicker}><span>{selectedModel.badge.slice(0, 2)}</span><div><strong id="inference-model-browser-title">{selectedModel.name}</strong><small>{selectedModel.task} · {modelStatusRefreshing ? '刷新状态…' : modelIsLoaded ? '已加载' : selectedModel.availability === 'missing_weights' ? '缺少权重' : modelLoadError ? '加载失败' : selectedModel.availability === 'available' ? '已下载 · 选择后自动加载' : '不可用'}</small></div><svg className="model-picker-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.75 6 7.5l3-2.75" /></svg></button>{modelActionKind === 'download' || modelActionKind === 'retry' ? <button type="button" className={`model-action-button ${selectedModelDownloadActive ? `downloading ${selectedModelDownloadProgress === null ? 'indeterminate' : 'determinate'}` : ''}`} style={selectedModelDownloadActive && selectedModelDownloadProgress !== null ? ({ '--model-download-progress': `${selectedModelDownloadProgress}%` } as React.CSSProperties) : undefined} disabled={backend.mode !== 'online' || !selectedModel.id || backend.runtime.phase === 'loading' || modelDownloadActionPending || selectedModelDownloadActive} onClick={() => modelActionKind === 'download' ? void downloadSelectedModel() : void loadSelectedModel()}>{modelActionLabel}</button> : <span className={`model-state-pill ${modelIsLoaded ? 'loaded' : backend.runtime.phase === 'loading' ? 'loading' : 'idle'}`}>{backend.runtime.phase === 'loading' ? '加载中…' : modelIsLoaded ? '已加载' : '未加载'}</span>}</div></section>
            {modelPickerOpen && <ModelPickerDialog models={displayedModelCatalog} tasks={modelTasks} selectedTask={modelTask} selectedModelId={selectedModel.id} refreshing={modelStatusRefreshing} downloadPending={modelDownloadActionPending} downloadActive={selectedModelDownloadActive} downloadProgress={selectedModelDownloadActive ? selectedModelDownloadProgress : null} onTaskChange={chooseModelTask} onSelect={(modelId) => chooseModel(modelId)} onDownload={() => void downloadSelectedModel()} onClose={closeModelPicker} />}
            {isSamModel && <section className={`sam-prompt-panel ${samPromptMode ? 'active' : ''}`}><header><div><span className="eyebrow">SAM Prompts</span><strong>当前图交互提示</strong></div><b>{samPromptCount} 个</b></header><div className="sam-prompt-tools"><button className={samPromptMode && samPromptTool === 'positive' ? 'active positive' : 'positive'} onClick={() => { drawRef.current = null; setDrawPreview(null); setSamPromptMode(true); setSamPromptTool('positive'); }}>＋ 正点</button><button className={samPromptMode && samPromptTool === 'negative' ? 'active negative' : 'negative'} onClick={() => { drawRef.current = null; setDrawPreview(null); setSamPromptMode(true); setSamPromptTool('negative'); }}>− 负点</button><button className={samPromptMode && samPromptTool === 'box' ? 'active box' : 'box'} onClick={() => { drawRef.current = null; setDrawPreview(null); setSamPromptMode(true); setSamPromptTool('box'); }}>▭ 框选</button><button disabled={samPromptCount === 0} onClick={() => { samBoxDragRef.current = null; setSamPoints([]); setSamBoxes([]); setSamBoxPreview(null); }}>清空</button></div><div className="sam-prompt-counts"><span>正点 {samPoints.filter((point) => point.label === 1).length}</span><span>负点 {samPoints.filter((point) => point.label === 0).length}</span><span>框 {samBoxes.length}</span></div>{samPromptMode ? <button className="sam-exit" onClick={() => { samBoxDragRef.current = null; setSamBoxPreview(null); setSamPromptMode(false); }}>退出提示模式 · 恢复普通标注工具</button> : <p>选择提示类型后在当前图点击或拖框。提示仅作为推理参数，不会写入原标注。</p>}</section>}
            <section className="side-section compact inference-parameter-card"><div className="section-title"><div><h2>推理参数</h2><small>{Object.keys(selectedModel.parametersSchema).length ? `${Object.keys(selectedModel.parametersSchema).length} 个模型参数 · 修改后自动刷新当前图` : '该模型未声明额外推理参数'}</small></div><button disabled={!Object.keys(selectedModel.parametersSchema).length} onClick={resetInferenceParameters}>恢复默认</button></div>{Object.keys(selectedModel.parametersSchema).length ? <div className="model-parameter-list">{Object.entries(selectedModel.parametersSchema).map(([name, schema]) => <PipelineParameterControl key={name} name={name} label={schema.title ?? name} schema={schema} value={inferenceParameters[name]} onChange={updateInferenceParameter} />)}</div> : <div className="model-parameter-empty">使用模型适配器的固定默认配置。</div>}<label className="parameter-row runtime-provider"><span>运行设备</span><CustomSelect ariaLabel="运行设备" value={inferenceProvider} options={[{ value: 'CPUExecutionProvider', label: 'CPU · ONNX Runtime' }]} onChange={setInferenceProvider} /></label><small className="provider-note">选择或调整模型参数后，会自动对当前图片执行单图推理；批量任务仍需明确创建。</small></section>
            <section ref={inferenceFeatureCardRef} className="feature-capture-card" tabIndex={-1}>
              <header><div><span className="eyebrow">Layer Visualization</span><strong>中间层可视化</strong><small>{!selectedModel.capture ? '当前适配器不开放内部张量' : !modelIsLoaded ? '加载权重后解析真实 ONNX 图' : selectedLayer ? `已启用 · ${selectedLayer.name}` : `${availableLayers.length} 个可捕获层 · 选择后自动启用`}</small></div></header>
              <div className="feature-layer-select"><span>选择层</span><CustomSelect disabled={!modelIsLoaded || availableLayers.length === 0} ariaLabel="选择可视化层" value={selectedLayer?.id ?? ''} options={availableLayers.map((layer) => ({ value: layer.id, label: `${layer.group} / ${layer.name}` }))} placeholder={!modelIsLoaded ? '请先加载模型' : availableLayers.length ? '选择一个真实层' : '没有可捕获层'} onChange={setSelectedLayerId} /></div>
              {!selectedModel.capture ? <div className="feature-unavailable">{selectedModel.adapter === 'trusted_remote_http' ? '远程黑盒模型不提供内部层。' : '当前模型适配器未声明可捕获的 ONNX 特征层。'}</div>
                : !modelIsLoaded ? <div className="feature-load-state"><p>请先加载模型，随后这里会显示真实 ONNX 图中的可捕获层。</p></div>
                  : availableLayers.length === 0 ? <div className="feature-unavailable">{backend.runtime.data?.capture_warning ?? '这个模型没有可安全捕获的浮点特征图、Token 或向量层。'}</div>
                    : <>{selectedLayer && <>
                      <div className="feature-layer-meta"><code title={selectedLayer.id}>{selectedLayer.id}</code><span>{featureTensorKindLabel(selectedFeatureKind)} · {selectedLayer.dtype ?? 'dtype unknown'} · {selectedLayer.axes.join(' × ') || 'axes unknown'}</span></div>
                      <div className="tensor-shape"><span>原始 Tensor</span><b>{formatTensorShape(selectedLayer.shape)}</b><i>→</i><span>输出 Artifact</span><b>{selectedFeatureArtifact ? formatTensorShape(selectedFeatureArtifact.shape) : featureOutputShape}</b></div>
                      <div className={`feature-preview ${selectedFeaturePreviewUrl ? 'ready' : 'empty'}`}>{selectedFeaturePreviewUrl ? <span className={`feature-preview-image ${selectedFeatureKind}`} role="img" aria-label={`${selectedLayer.name} 的真实${featureTensorKindLabel(selectedFeatureKind)}预览`} style={{ backgroundImage: `url(${selectedFeaturePreviewUrl})` }} /> : <span className="feature-preview-empty">运行后显示</span>}<div><strong>{selectedLayer.name}</strong><small>{selectedFeatureArtifact ? `真实 PNG · ${selectedFeatureArtifact.preview_width ?? '?'} × ${selectedFeatureArtifact.preview_height ?? '?'} · ${formatArtifactBytes(selectedFeatureArtifact.size_bytes)}` : featurePreviewDescription(selectedFeatureKind)}</small></div></div>
                      <div className="feature-controls">
                        {selectedFeatureProjectionOptions.length > 1 && <label><span>{selectedFeatureKind === 'spatial-map' ? '通道投影' : '展示方式'}</span><CustomSelect ariaLabel="特征展示方式" value={projection} options={selectedFeatureProjectionOptions.map((item) => ({ value: item, label: item }))} onChange={setProjection} /></label>}
                        <label><span>数值归一化</span><CustomSelect ariaLabel="数值归一化" value={normalization} options={['Min-Max', 'Z-Score', 'L2', 'None'].map((item) => ({ value: item, label: item }))} onChange={setNormalization} /></label>
                        <label><span>数值截断</span><CustomSelect ariaLabel="特征值截断" value={featureClip} options={[{ value: 'p1p99', label: 'P1–P99' }, { value: 'p5p95', label: 'P5–P95' }, { value: 'none', label: '不截断' }]} onChange={setFeatureClip} /></label>
                      </div>
                      {projection === 'Single Channel' && <label className="feature-channel-control"><span>通道索引</span><input type="number" min="0" max={Math.max(0, selectedFeatureChannelCount - 1)} value={featureChannel} onChange={(event) => setFeatureChannel(Math.max(0, Number(event.target.value) || 0))} /><b>#{featureChannel}</b></label>}
                    </>}</>}
              {backend.runtime.data?.capture_warning && availableLayers.length > 0 && <p className="feature-capture-warning">{backend.runtime.data.capture_warning}</p>}
            </section>
            {(currentSegmentationContours.length > 0 || currentRasters.length > 0 || currentClassifications.length > 0) && <section className="inference-display-controls" aria-label="当前图推理结果显示">
              {(currentSegmentationContours.length > 0 || currentRasters.length > 0) && <div className={`prediction-display-control ${showMasks && !pixelResultDisplayBlockedReason ? 'active' : ''}`}><div><span className="eyebrow">Segmentation / Pixel</span><strong>分割 / 像素结果</strong><small>{pixelResultSummary}</small></div><button role="switch" aria-label="显示分割和像素结果" aria-checked={showMasks && !pixelResultDisplayBlockedReason} disabled={Boolean(pixelResultDisplayBlockedReason)} title={pixelResultDisplayBlockedReason ?? undefined} onClick={() => setShowMasks((visible) => !visible)}>{pixelResultDisplayBlockedReason ? '不可叠加' : showMasks ? '显示中' : '已隐藏'}</button></div>}
              {currentClassifications.length > 0 && <div className={`prediction-display-control ${showClassifications ? 'active' : ''}`}><div><span className="eyebrow">Classification</span><strong>分类浮层</strong><small>Top-{Math.min(5, currentClassifications.length)} · {currentClassifications[0].label}</small></div><button role="switch" aria-label="显示分类 Top-K" aria-checked={showClassifications} onClick={() => setShowClassifications((visible) => !visible)}>{showClassifications ? '显示中' : '已隐藏'}</button></div>}
            </section>}
            {showInferenceResult && <div className="inference-results-region">
              <div className="inference-result-strip" role="status" aria-label="当前图推理摘要"><span><b>{Math.round(currentInferenceResult?.timings_ms.total ?? 0)} ms</b>耗时</span>{currentAnnotationsAreSegmentation && <span><b>{currentPredictions.length}</b>轮廓</span>}<span><b>{currentClassifications.length}</b>分类</span><span><b>{currentArtifacts.length}</b>特征</span>{currentRasters.length > 0 && <span><b>{currentRasters.length}</b>Raster</span>}</div>
              {backend.mode === 'online' && currentRasters.length > 0 && <section className="inference-rasters"><header><div><span className="eyebrow">Pixel Outputs</span><strong>当前图像素 / Mask 结果</strong></div><b>{currentRasters.length}</b></header><div>{currentRasters.map((raster) => {
                const contentUrl = backend.artifactContentUrl(raster.id);
                const compatible = !showingPipelineImage && inferenceRasterMatchesSource(raster, currentFile?.width, currentFile?.height);
                const active = raster.id === activeRaster?.id;
                const mismatchReason = showingPipelineImage ? '处理流底图开启时不可叠加' : `尺寸 ${raster.width} × ${raster.height} 与源图 ${currentFile?.width ?? '?'} × ${currentFile?.height ?? '?'} 不一致`;
                return <button key={raster.id} className={active ? 'active' : ''} aria-pressed={active} disabled={!compatible} title={compatible ? `在画布显示 ${raster.role}` : mismatchReason} onClick={() => { setSelectedRasterId(raster.id); setShowMasks(true); }}>{contentUrl && <span className="raster-preview" aria-hidden="true" style={{ backgroundImage: `url(${contentUrl})` }} />}<span className="raster-copy"><strong>{raster.role}</strong><small>{raster.width} × {raster.height} · {formatArtifactBytes(raster.size_bytes)} · {raster.media_type}</small><em>{compatible ? summarizeRasterMetadata(raster.metadata) : mismatchReason}</em></span><b>{compatible ? active ? '画布显示中' : '显示' : '不可叠加'}</b></button>;
              })}</div></section>}
              {backend.mode === 'online' && currentClassifications.length > 0 && <section className="classification-results"><header><span className="eyebrow">Classification</span><strong>当前图 Top-K</strong></header>{currentClassifications.map((item) => <div key={`${item.rank}:${item.label}`}><b>#{item.rank}</b><span>{item.label}</span><strong>{(item.score * 100).toFixed(2)}%</strong></div>)}</section>}
              {backend.mode === 'online' && currentArtifacts.length > 0 && <section className="feature-result-compact" aria-label="当前图中间层结果">{currentArtifacts.map((artifact) => <details key={artifact.id}><summary><span><strong>{artifact.layer_id}</strong><small>{artifact.dtype} · {formatArtifactBytes(artifact.size_bytes)}</small></span><code>{formatTensorShape(artifact.source_shape)} → {formatTensorShape(artifact.shape)}</code><b>详情</b></summary><div className="feature-result-details"><dl>{Object.entries(artifact.statistics).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatArtifactNumber(value)}</dd></div>)}</dl><code>{artifact.id}</code><p>{summarizeFeatureTransform(artifact.transform)}</p></div></details>)}</section>}
            </div>}
          </section>}

          {rightTab === 'agent' && <section className={`agent-panel ${agentBackendReady ? 'ready' : 'unavailable'}`}>
            <header className="agent-header">
              <div className="agent-mark" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M8 4.5h8a3.5 3.5 0 0 1 3.5 3.5v5A3.5 3.5 0 0 1 16 16.5h-3.2L9 19v-2.5H8A3.5 3.5 0 0 1 4.5 13V8A3.5 3.5 0 0 1 8 4.5Z" /><path d="M8.5 9h7M8.5 12h4.5" /></svg></div>
              <div><strong>LabelOne Agent</strong><small>受控数据工作流助手</small></div>
              <span className={agentBackendReady ? 'live' : 'offline'}>{agentSending ? '处理中' : agentBackendReady ? '已配置' : '不可用'}</span>
            </header>

            {!agentBackendReady ? <div className="agent-unavailable-state" role="status" aria-live="polite">
              <div className="agent-unavailable-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 3.5a8.5 8.5 0 1 0 8.5 8.5A8.5 8.5 0 0 0 12 3.5Z" /><path d="M12 7.5v5.2M12 16.5h.01" /></svg></div>
              <div><span className="eyebrow">Agent unavailable</span><h2>{agentUnavailableTitle}</h2><p>{agentUnavailableMessage}</p></div>
              <div className="agent-unavailable-capabilities" aria-label="配置后可用能力">
                <div><strong>数据检查</strong><small>异常、筛选、标签分布</small></div>
                <div><strong>标注质检</strong><small>越界、退化、重复对象</small></div>
                <div><strong>工作流提案</strong><small>处理流与模型任务，执行前确认</small></div>
              </div>
              <button type="button" className="agent-configure-button" disabled={backend.agentStatus.phase === 'loading' && backend.mode === 'online'} onClick={() => { if (backend.mode !== 'online') void backend.refreshHealth(); else { setSettingsSection('ai'); setSettingsOpen(true); } }}>{backend.mode !== 'online' ? '重试本地服务' : backend.agentStatus.phase === 'loading' ? '正在检查配置…' : '配置 Agent 后端'}</button>
              <small className="agent-unavailable-note">未配置时不会发送请求，也不会执行任何 Agent 工具。</small>
            </div> : <>
              <div className="agent-context-strip" aria-label="本次 Agent 上下文">
                <div><span>项目</span><strong title={dataset.name}>{dataset.name}</strong></div>
                <div><span>当前图</span><strong title={currentFile?.name}>{currentFile?.name ?? '无'}</strong></div>
                <div><span>对象</span><strong>{annotationDraft?.document.shapes.length ?? currentFile?.annotations ?? 0}</strong></div>
              </div>

              <section className="agent-capability-workbench" aria-labelledby="agent-capability-title">
                <header><div><strong id="agent-capability-title">快捷检查</strong><small>只读工具直接执行；界面动作和任务必须确认</small></div><span>{backend.agentStatus.data?.capabilities.length ?? 0} 项能力</span></header>
                <div className="agent-quick-actions">
                  <button type="button" disabled={agentSending || !dataset.id} onClick={() => void sendAgent('检查数据集概况', { tool: 'dataset.stats', arguments: {} })}><strong>数据概况</strong><small>有效与异常条目</small></button>
                  <button type="button" disabled={agentSending || !dataset.id} onClick={() => void sendAgent('查找未标注图片', { tool: 'dataset.search', arguments: { annotated: false, limit: 20 } })}><strong>未标注图片</strong><small>返回前 20 项</small></button>
                  <button type="button" disabled={agentSending || !dataset.id} onClick={() => void sendAgent('统计标签分布', { tool: 'dataset.distribution', arguments: { top_n: 20 } })}><strong>标签分布</strong><small>类别与形状统计</small></button>
                  <button type="button" disabled={agentSending || !currentFile?.id} title={currentFile?.id ? undefined : '需要先选择当前图片'} onClick={() => void sendAgent('检查当前标注', { tool: 'annotation.qa', arguments: {} })}><strong>当前图质检</strong><small>{currentFile?.id ? '检查标注问题' : '需要当前图片'}</small></button>
                </div>
                <details className="agent-capability-details"><summary><span>查看全部受控能力</span><small>权限与上下文要求</small></summary><div>{agentCapabilitySections.map((section) => <section key={section.id}><header><div><strong>{section.title}</strong><small>{section.description}</small></div><span>{section.capabilities.length}</span></header><ul>{section.capabilities.map((capability) => <li key={capability.tool} className={capability.requires_asset && !currentFile?.id ? 'context-disabled' : ''}><div><strong>{capability.title}</strong><small>{capability.description}</small></div><span className={capability.risk}>{capability.risk === 'read' ? '只读' : '需确认'}</span></li>)}</ul></section>)}</div></details>
              </section>

              <div className="agent-run-feed" role="log" aria-live="polite" aria-label="Agent 任务记录">{agentMessages.length === 0 && <div className="agent-run-empty"><strong>还没有任务记录</strong><small>选择上方快捷检查，或输入一个与当前项目有关的任务。</small></div>}{agentMessages.map((message) => <article key={message.id} className={`agent-run-entry ${message.role} ${message.source ?? ''} ${message.run ? 'with-result' : ''}`}>
                <header><span>{message.role === 'user' ? '任务指令' : message.source === 'error' ? '执行失败' : 'Agent 结果'}</span>{message.run && <b className={message.run.state}>{message.run.state === 'completed' ? '已完成' : message.run.state === 'failed' ? '失败' : '待确认'}</b>}</header>
                <p>{message.text}</p>
                {message.context && <small className="agent-context">{message.context.label}</small>}
                {message.run?.tool_results.map((result) => <AgentToolResultView key={`${message.run!.run_id}:${result.tool}`} result={result} />)}
                {message.run?.proposals.map((proposal) => {
                  const key = `${message.run!.run_id}:${proposal.id}`;
                  const pending = agentProposalPending === key;
                  const confirming = agentConfirmingProposal === key;
                  const contextMismatch = message.run!.dataset_id !== dataset.id || Boolean(message.run!.asset_id && message.run!.asset_id !== currentFile?.id);
                  return <section key={proposal.id} className={`agent-proposal ${proposal.risk} ${proposal.executed ? 'executed' : ''} ${contextMismatch ? 'context-mismatch' : ''}`}><header><code>{proposal.tool}</code><div><span className={`risk ${proposal.risk}`}>{proposal.risk === 'write' ? '受控写操作' : '只读'}</span>{proposal.requires_confirmation && <span className="confirm-required">需要确认</span>}</div></header><strong>{proposal.title}</strong><p>{proposal.description}</p>{proposal.executed ? <div className="proposal-executed"><span aria-hidden="true">✓</span><strong>已执行</strong></div> : contextMismatch ? <div className="proposal-context-warning" role="alert">该提案属于之前的项目或图片。请切回原上下文后再确认。</div> : confirming ? <div className="proposal-confirm"><span>确认后才会调用本地工具；服务端返回前不会标记完成。</span><div><button type="button" disabled={pending} onClick={() => setAgentConfirmingProposal(null)}>取消</button><button type="button" className="confirm" disabled={pending} onClick={() => void executeAgentProposal(message.run!.run_id, proposal.id)}>{pending ? '执行中…' : '确认执行'}</button></div></div> : <button type="button" disabled={pending} onClick={() => setAgentConfirmingProposal(key)}>{pending ? '执行中…' : '审核并确认'}</button>}{agentExecutionErrors[key] && <small className="proposal-error" role="alert">{agentExecutionErrors[key]}</small>}</section>;
                })}
              </article>)}</div>

              <div className="agent-composer"><label htmlFor="agent-task-input">任务指令</label><textarea id="agent-task-input" aria-label="给 Agent 的任务指令" disabled={agentSending} value={agentInput} onChange={(event) => setAgentInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendAgent(); } }} placeholder="例如：分析未标注图片，并建议一个可审核的增强流程" /><div><span>仅处理数据检查、标注质检、处理流和模型推理</span><button type="button" disabled={agentSending || !agentInput.trim() || !dataset.id} onClick={() => void sendAgent()}>{agentSending ? '处理中…' : '提交任务'}</button></div></div>
            </>}
          </section>}
          </div>
        </aside>
      </section> : <WelcomeScreen
        backendMode={backend.mode}
        recentProjects={recentProjects}
        openingProjectId={openingRecentDatasetId}
        openingFolder={directoryPickerPending === 'image' || scanRegistering}
        error={welcomeError}
        onOpenProject={() => void pickImageDataset()}
        onOpenRecent={(project) => void openRecentProject(project)}
        onOpenSettings={openGlobalSettings}
        onRetryService={() => void backend.refreshHealth()}
      />}

      {datasetOpen && <div className="recovery-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setDatasetOpen(false); }}>
        <section className="dataset-open-dialog" role="dialog" aria-modal="true" aria-label="打开数据集">
          <header><div><span className="eyebrow">Dataset Scan</span><h2>正在打开数据集</h2><p>正在递归扫描 {rootPath} 中的图像与同名 JSON。关闭此窗口不会停止扫描。</p></div><button onClick={() => setDatasetOpen(false)}>×</button></header>
          {backend.scan.data && <>
            <div className={`scan-progress ${scanIsLoading ? 'indeterminate' : ''}`}><div><span style={{ width: scanIsLoading ? '35%' : scanReady ? '100%' : '0%' }} /></div><small>{scanIsLoading ? `正在后台扫描 · ${backend.scan.data.persisted_items.toLocaleString()} 项已落盘；首批可用后立即打开。` : scanRegistering ? '正在装载首批索引…' : autoOpenError ? `自动打开失败：${autoOpenError}` : scanSessionState === 'interrupted' ? `扫描已中断；保留 ${backend.scan.data.persisted_items.toLocaleString()} 项，可继续重试。` : scanSessionState === 'failed' ? `扫描失败：${backend.scan.data.error ?? '未知错误'}；可继续重试。` : scanReady ? '后台扫描完成，正在合并最终索引。' : '等待扫描状态。'}</small></div>
            <div className="match-preview expanded"><div><span>✓</span><strong>{scanSummary.valid.toLocaleString()}</strong><small>有效配对</small></div><div><span>◌</span><strong>{scanSummary.hidden_image_only.toLocaleString()}</strong><small>仅图像 · 隐藏</small></div><div><span>!</span><strong>{scanSummary.duplicate_match}</strong><small>重复 · 置灰</small></div><div><span>!</span><strong>{scanSummary.orphan_annotation}</strong><small>孤立 JSON</small></div><div><span>!</span><strong>{scanSummary.corrupt_image + scanSummary.corrupt_annotation}</strong><small>损坏文件</small></div></div>
            {backend.mode === 'online' && <div className="scan-stream-results"><header><span>增量扫描结果</span><strong>{backend.scan.data.items.length.toLocaleString()} / {backend.scan.data.persisted_items.toLocaleString()} 已载入</strong></header>{backend.scan.data.items.slice(-5).map((item) => <div key={item.asset_id} className={item.selectable ? 'selectable' : 'disabled'}><span>{item.selectable ? '✓' : '!'}</span><strong>{item.display_path}</strong><small>{item.selectable ? `${item.annotation_count ?? 0} 标注` : item.reason ?? item.status}</small></div>)}</div>}
            <footer className="scan-status-actions">{(scanSessionState === 'queued' || scanSessionState === 'running') && <button className="ghost-button" disabled={scanRegistering} onClick={() => void backend.interruptScan().catch((error) => notify(error instanceof Error ? error.message : '扫描中断失败'))}>中断扫描</button>}{(scanSessionState === 'failed' || scanSessionState === 'interrupted') && <button className="primary-button" disabled={scanRegistering} onClick={() => void backend.resumeScan().catch((error) => notify(error instanceof Error ? error.message : '扫描继续失败'))}>继续扫描</button>}{scanReady && autoOpenError && autoScanIntent && <button className="primary-button" disabled={scanRegistering} onClick={() => void openScannedDataset(autoScanIntent.sessionId, autoScanIntent.name, autoScanIntent.operationId)}>重试自动注册并打开</button>}</footer>
          </>}
        </section>
      </div>}

      {remoteInferenceConfirmation && <div className="recovery-overlay"><section className="remote-inference-confirmation" role="dialog" aria-modal="true" aria-label="远程推理隐私确认"><header><span className="eyebrow">Remote Inference Privacy</span><h2>确认发送图像到远程 HTTPS 服务</h2><p>这是一次性确认，只授权下方这一次推理动作；不会记住授权。</p></header><div className="remote-inference-scope"><span>发送范围</span><strong>{remoteInferenceConfirmation.action === 'current' ? `当前图：${remoteInferenceConfirmation.fileName ?? remoteInferenceConfirmation.asset_id}` : `全部可选图像：${(remoteInferenceConfirmation.selectableCount ?? 0).toLocaleString()} 项`}</strong></div><ul><li>图像数据将发送到用户配置的受信 HTTPS endpoint。</li><li>认证凭据由本地服务从环境变量读取，前端不会展示或保存凭据。</li><li>该适配器是远程黑盒：不提供可选择的中间层或本地特征捕获。</li></ul><footer><button className="ghost-button" onClick={() => setRemoteInferenceConfirmation(null)}>取消 · 不加载、不发送</button><button className="primary-button" onClick={confirmRemoteInference}>确认并仅执行这一次</button></footer></section></div>}

      {pendingManualShape && manualLabelMenuPosition && <form ref={manualLabelMenuRef} className="annotation-label-popover" role="dialog" aria-labelledby="annotation-label-title" style={{ left: manualLabelMenuPosition.x, top: manualLabelMenuPosition.y, transformOrigin: manualLabelMenuPosition.transformOrigin }} onPointerDown={(event) => event.stopPropagation()} onSubmit={(event) => { event.preventDefault(); commitPendingManualShape(); }}><header><span className="manual-shape-kind"><i><ShapeTypeIcon shapeType={pendingManualShape.shape.shape_type} /></i><strong id="annotation-label-title">选择类别</strong><small>{shapeTypeLabels[pendingManualShape.shape.shape_type] ?? pendingManualShape.shape.shape_type}</small></span><button type="button" aria-label="取消这次标注" onClick={cancelPendingManualShape}>×</button></header>{manualLabelChoices.length > 0 ? <div className="annotation-label-menu" role="menu" aria-label="已知标签类别">{manualLabelChoices.map((label) => <button type="button" key={label} role="menuitem" onClick={() => commitPendingManualShape(label)}><span>{label.slice(0, 1).toUpperCase()}</span><strong>{label}</strong><small>↵</small></button>)}</div> : <div className="annotation-label-empty compact">暂无类别，可在下方新增</div>}<label className="annotation-label-create" htmlFor="manual-shape-label"><input id="manual-shape-label" autoFocus value={manualShapeLabel} maxLength={128} onChange={(event) => setManualShapeLabel(event.target.value)} placeholder="＋ 新增标签类别" autoComplete="off" spellCheck={false} /><button type="submit" disabled={!manualShapeLabelValid}>{manualShapeLabelKnown ? '选择' : '新增'}</button></label><small className="annotation-label-hint">单击即选 · Enter 确认 · Esc 取消</small></form>}

      {pendingAnnotationNavigation && <div className="recovery-overlay"><section className="annotation-navigation-dialog" role="dialog" aria-modal="true" aria-labelledby="annotation-navigation-title"><header><div><span className="eyebrow">Unsaved Annotation</span><h2 id="annotation-navigation-title">要保留当前标注更改吗？</h2><p>只在切换图片时询问，不会反复占用图层面板空间。</p></div></header><div className="annotation-navigation-route"><span>{currentFile?.name ?? '当前图片'}</span><i>→</i><strong>{pendingAnnotationNavigation.targetLabel}</strong></div><p className="annotation-navigation-copy">{annotationDirty ? backend.mode === 'online' ? '当前图片还有尚未写入标注文件的更改。选择保留会先保存 JSON；选择不保留会清理本机草稿后切换。' : '本地服务离线。选择保留会把更改留在本机草稿中，稍后返回此图时可继续恢复。' : '更改已保存完成，可以继续切换。'}</p>{annotationNavigationError && <p className="annotation-navigation-error" role="alert">{annotationNavigationError}</p>}<footer><button className="ghost-button" disabled={Boolean(annotationNavigationDecision)} onClick={() => closeAnnotationNavigationPrompt(true)}>取消</button>{annotationDirty && <button className="danger-button" disabled={Boolean(annotationNavigationDecision) || annotationSaving} onClick={() => void discardChangesAndNavigate()}>{annotationNavigationDecision === 'discard' ? '正在清理…' : '不保留并切换'}</button>}<button className="primary-button" disabled={Boolean(annotationNavigationDecision) || annotationSaving} onClick={() => void keepChangesAndNavigate()}>{annotationNavigationDecision === 'keep' ? '正在保存…' : annotationDirty ? backend.mode === 'online' ? '保存并切换' : '保留本机并切换' : '继续切换'}</button></footer></section></div>}

      {annotationRecovery && <div className="recovery-overlay"><section className={`annotation-recovery-dialog ${annotationRecovery.kind}`} role="dialog" aria-modal="true" aria-label="意外退出草稿恢复"><header><div><span className="eyebrow">Crash Draft Recovery</span><h2>{annotationRecovery.kind === 'recoverable' ? '发现未同步的本机草稿' : '草稿与服务端版本冲突'}</h2><p>{annotationRecovery.kind === 'recoverable' ? '服务端 revision 未改变，可安全恢复到编辑区；恢复后仍会通过 ETag 保存。' : '服务端标注在草稿创建后已改变。为防止静默覆盖，只能先导出草稿或丢弃。'}</p></div></header><div className="annotation-recovery-revisions"><span>草稿基线 <code>{annotationRecovery.local.base_revision.slice(0, 16)}</code></span><span>服务端 <code>{annotationRecovery.server.revision.slice(0, 16)}</code></span><span>本机保存于 <strong>{new Date(annotationRecovery.local.updated_at).toLocaleString()}</strong></span></div><footer>{annotationRecovery.kind === 'conflict' && <button className="ghost-button" onClick={exportConflictingAnnotation}>导出冲突草稿 JSON</button>}<button className="danger-button" onClick={() => void discardRecoveredAnnotation()}>丢弃本机草稿</button>{annotationRecovery.kind === 'recoverable' && <button className="primary-button" onClick={restoreRecoveredAnnotation}>恢复到编辑区</button>}</footer></section></div>}

      {settingsOpen && <GlobalSettingsPage
        section={settingsSection}
        onSectionChange={(section) => { setSettingsSection(section); setShortcutFeedback(''); setRecordingShortcut(null); if (section === 'operators' && backend.mode === 'online') void refreshPipelineRegistry().catch(() => undefined); }}
        onClose={closeGlobalSettings}
        closeButtonRef={settingsCloseRef}
        isFullscreen={isFullscreen}
        onToggleFullscreen={() => void toggleFullscreen()}
        language={uiLanguage}
        onToggleLanguage={toggleUiLanguage}
        backendMode={backend.mode}
        remoteSettings={backend.applicationSettings.data}
        remoteSettingsLoading={backend.applicationSettings.phase === 'loading'}
        modelWeightsPath={modelWeightsPathInput}
        onModelWeightsPathChange={(value) => { setModelWeightsPathInput(value); setModelSettingsStatus(''); }}
        onPickModelWeightsPath={() => void pickGlobalSettingsDirectory()}
        onSaveModelWeightsPath={() => void saveModelWeightsDirectory()}
        modelWeightsSaving={modelSettingsSaving}
        modelDirectoryPicking={modelDirectoryPicking}
        modelSettingsStatus={modelSettingsStatus}
        modelDownloadSource={modelDownloadSource}
        onModelDownloadSourceChange={(value) => { setModelDownloadSource(value); setModelSettingsStatus(''); }}
        onSaveModelDownloadSource={() => void saveModelDownloadSource()}
        defaultInferenceProvider={inferenceProvider}
        onDefaultInferenceProviderChange={setInferenceProvider}
        currentInferenceModelName={selectedModel.name}
        currentPipelineSummary={`${nodes.filter((node) => node.kind !== 'source').length} 个处理节点 · ${visualizations.length} 个显示节点`}
        workspaceSettingsSaving={workspaceSettingsSaving}
        workspaceSettingsStatus={workspaceSettingsStatus}
        onSaveWorkspaceDefaults={() => void saveWorkspaceDefaults()}
        cloudAiDraft={cloudAiDraft}
        onCloudAiDraftChange={(next) => { setCloudAiDraft(next); setCloudAiStatus(''); }}
        cloudAiSaving={cloudAiSaving}
        cloudAiStatus={cloudAiStatus}
        onSaveCloudAi={() => void saveCloudAiSettings()}
        networkProxyDraft={networkProxyDraft}
        onNetworkProxyDraftChange={(next) => { setNetworkProxyDraft(next); setNetworkProxyStatus(''); }}
        networkProxySaving={networkProxySaving}
        networkProxyStatus={networkProxyStatus}
        onSaveNetworkProxy={() => void saveNetworkProxy()}
        operatorRegistry={backend.pipelineRegistry.data}
        operatorRegistryLoading={backend.pipelineRegistry.phase === 'loading'}
        operatorImporting={operatorImporting}
        operatorStatus={operatorImportStatus}
        operatorInspection={operatorInspection}
        onRefreshOperators={() => void refreshPipelineRegistry().catch((error) => setOperatorImportStatus(error instanceof Error ? error.message : '算子库刷新失败'))}
        onChooseOperatorZip={() => operatorPackageInputRef.current?.click()}
        onConfirmOperatorImport={() => void confirmOperatorPackageImport()}
        onCancelOperatorImport={cancelOperatorPackageImport}
        shortcuts={shortcuts}
        shortcutOverrides={shortcutOverrides}
        recordingShortcut={recordingShortcut}
        shortcutFeedback={shortcutFeedback}
        useMacSymbols={useMacShortcutSymbols}
        onStartRecording={(action) => { setRecordingShortcut(action); setShortcutFeedback('请按下新的快捷键；Esc 取消。'); }}
        onRecordShortcut={recordShortcut}
        onResetShortcut={resetShortcut}
        onResetAllShortcuts={resetAllShortcuts}
      />}
      <input ref={operatorPackageInputRef} hidden type="file" accept=".zip,application/zip" aria-label="从系统导入算子包" onChange={(event) => void importOperatorPackageFile(event.target.files?.[0])} />

      {toast && <div className="toast" role="status"><span>✓</span>{toast}</div>}
    </main>
  );
}
