import { canvasVisibleImageBounds, type CanvasVisibleBounds } from './canvas-label.ts';
import { isSourcePixelGridVisible, sourcePixelScreenSize } from './canvas-viewport.ts';
import type { CanvasCoordinateTransform } from './canvas-interaction.ts';

export type SharedPipelineView = { scale: number; x: number; y: number };
export type PipelineDisplayRect = { left: number; top: number; width: number; height: number };
export type PipelineLayerState = Record<string, { visible: boolean; opacity: number }>;
export type PipelineDisplayMode = 'source' | 'split' | 'overlay';
export type PipelineCoordinateMappingLike = {
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
export type PipelineSharedCursor = {
  activeVisualizationId: string;
  label: string;
  localX: number;
  localY: number;
  activeWidth: number;
  activeHeight: number;
  sourceX: number | null;
  sourceY: number | null;
};

export function resolvePipelineDisplayMode(
  mode: PipelineDisplayMode,
  slotCount: number,
  overlayCompatibilityConfirmed: boolean,
  overlayAllowed: boolean,
): PipelineDisplayMode {
  if (slotCount <= 1) return 'source';
  if (mode === 'overlay' && overlayCompatibilityConfirmed && !overlayAllowed) return 'split';
  return mode;
}
export type PipelinePaneMetrics = {
  contained: PipelineDisplayRect;
  display: PipelineDisplayRect;
  visibleBounds: CanvasVisibleBounds | null;
  imageUnitsPerScreenPixel: number;
  pixelWidthOnScreen: number;
  pixelHeightOnScreen: number;
  pixelGridVisible: boolean;
};

export function snapPipelineGridCoordinate(value: number, devicePixelRatio: number): number {
  const dpr = Math.max(1, Number.isFinite(devicePixelRatio) ? devicePixelRatio : 1);
  return (Math.round(value * dpr) + 0.5) / dpr;
}

export function stablePipelineDisplaySlots<
  TDefinition extends { id: string; parameters?: Record<string, unknown> },
  TResult extends { visualization_id: string; label: string },
>(definitions: TDefinition[], results: TResult[], maximum = 4) {
  const resultById = new Map(results.map((result) => [result.visualization_id, result]));
  return definitions.slice(0, maximum).map((definition, index) => ({
    visualization_id: definition.id,
    label: String(definition.parameters?.label ?? resultById.get(definition.id)?.label ?? `显示 ${index + 1}`),
    result: resultById.get(definition.id) ?? null,
  }));
}

export function updatePipelineLayerOpacity(layers: PipelineLayerState, layerId: string, opacity: number): PipelineLayerState {
  const current = layers[layerId] ?? { visible: true, opacity: 100 };
  const nextOpacity = Math.max(0, Math.min(100, Math.round(opacity)));
  return { ...layers, [layerId]: { ...current, opacity: nextOpacity } };
}

export function updatePipelineLayerVisibility(layers: PipelineLayerState, layerId: string, visible: boolean): PipelineLayerState {
  const current = layers[layerId] ?? { visible: true, opacity: 100 };
  return { ...layers, [layerId]: { ...current, visible } };
}

export function containedPipelineImageRect(
  container: PipelineDisplayRect,
  imageWidth: number,
  imageHeight: number,
): PipelineDisplayRect {
  if (container.width <= 0 || container.height <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { left: container.left, top: container.top, width: 0, height: 0 };
  }
  const scale = Math.min(container.width / imageWidth, container.height / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    left: container.left + (container.width - width) / 2,
    top: container.top + (container.height - height) / 2,
    width,
    height,
  };
}

export function normalizedPipelinePan(view: SharedPipelineView, referenceWidth: number, referenceHeight: number) {
  return {
    x: referenceWidth > 0 ? view.x / referenceWidth : 0,
    y: referenceHeight > 0 ? view.y / referenceHeight : 0,
  };
}

export function pipelinePaneVectorToReference(
  vector: { x: number; y: number },
  paneWidth: number,
  paneHeight: number,
  referenceWidth: number,
  referenceHeight: number,
) {
  return {
    x: paneWidth > 0 && referenceWidth > 0 ? vector.x * referenceWidth / paneWidth : vector.x,
    y: paneHeight > 0 && referenceHeight > 0 ? vector.y * referenceHeight / paneHeight : vector.y,
  };
}

export function pipelineWheelInputToReference(
  input: { deltaX: number; deltaY: number; deltaMode: number; ctrlKey: boolean; metaKey: boolean },
  paneWidth: number,
  paneHeight: number,
  referenceWidth: number,
  referenceHeight: number,
) {
  if (input.ctrlKey || input.metaKey) return input;
  const delta = pipelinePaneVectorToReference(
    { x: input.deltaX, y: input.deltaY },
    paneWidth,
    paneHeight,
    referenceWidth,
    referenceHeight,
  );
  return { ...input, deltaX: delta.x, deltaY: delta.y };
}

export function pipelinePaneTransform(view: SharedPipelineView, referenceWidth: number, referenceHeight: number): string {
  const pan = normalizedPipelinePan(view, referenceWidth, referenceHeight);
  return `translate(${pan.x * 100}%, ${pan.y * 100}%) scale(${view.scale})`;
}

export function pipelinePaneMetrics(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
  view: SharedPipelineView,
  referenceWidth = containerWidth,
  referenceHeight = containerHeight,
  pixelGridEnabled = false,
): PipelinePaneMetrics {
  const contained = containedPipelineImageRect({ left: 0, top: 0, width: containerWidth, height: containerHeight }, imageWidth, imageHeight);
  const pan = normalizedPipelinePan(view, referenceWidth, referenceHeight);
  const paneView = { scale: view.scale, x: pan.x * containerWidth, y: pan.y * containerHeight };
  const displayWidth = contained.width * view.scale;
  const displayHeight = contained.height * view.scale;
  const display = {
    left: containerWidth / 2 - displayWidth / 2 + paneView.x,
    top: containerHeight / 2 - displayHeight / 2 + paneView.y,
    width: displayWidth,
    height: displayHeight,
  };
  const visibleBounds = canvasVisibleImageBounds(imageWidth, imageHeight, contained.width, contained.height, containerWidth, containerHeight, paneView);
  const pixelWidthOnScreen = sourcePixelScreenSize(imageWidth, contained.width, view.scale);
  const pixelHeightOnScreen = sourcePixelScreenSize(imageHeight, contained.height, view.scale);
  const scaledContainedWidth = contained.width * Math.max(view.scale, 0.01);
  const scaledContainedHeight = contained.height * Math.max(view.scale, 0.01);
  const imageUnitsPerScreenPixel = Math.max(
    scaledContainedWidth > 0 ? imageWidth / scaledContainedWidth : 0,
    scaledContainedHeight > 0 ? imageHeight / scaledContainedHeight : 0,
  ) || 1;
  return {
    contained,
    display,
    visibleBounds,
    imageUnitsPerScreenPixel,
    pixelWidthOnScreen,
    pixelHeightOnScreen,
    pixelGridVisible: isSourcePixelGridVisible(pixelGridEnabled, pixelWidthOnScreen, pixelHeightOnScreen),
  };
}

function applyAffine(point: [number, number], transform: [number, number, number, number, number, number]): [number, number] {
  const [a, b, c, d, e, f] = transform;
  return [a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f];
}

export function pipelineCoordinateMappingFromTransform(
  transform: CanvasCoordinateTransform | null,
  sourceWidth: number,
  sourceHeight: number,
): PipelineCoordinateMappingLike | null {
  if (!transform || sourceWidth <= 0 || sourceHeight <= 0) return null;
  const determinant = transform.a * transform.d - transform.b * transform.c;
  if (Math.abs(determinant) <= Number.EPSILON) return null;
  const sourceToOutput: [number, number, number, number, number, number] = [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f];
  const outputToSource: [number, number, number, number, number, number] = [
    transform.d / determinant,
    -transform.b / determinant,
    -transform.c / determinant,
    transform.a / determinant,
    (transform.c * transform.f - transform.d * transform.e) / determinant,
    (transform.b * transform.e - transform.a * transform.f) / determinant,
  ];
  return {
    kind: sourceToOutput.every((value, index) => value === [1, 0, 0, 1, 0, 0][index]) ? 'identity' : 'affine',
    source_width: sourceWidth,
    source_height: sourceHeight,
    output_width: transform.width,
    output_height: transform.height,
    source_to_output: sourceToOutput,
    output_to_source: outputToSource,
    coordinate_space_id: `frontend:${sourceToOutput.join(':')}`,
    topology_safe: transform.topologySafe,
  };
}

export function canvasCoordinateTransformFromPipelineMapping(
  mapping: PipelineCoordinateMappingLike | null | undefined,
): CanvasCoordinateTransform | null {
  if (!mapping || mapping.kind === 'unavailable' || !mapping.source_to_output) return null;
  const [a, b, c, d, e, f] = mapping.source_to_output;
  if (![a, b, c, d, e, f, mapping.output_width, mapping.output_height].every(Number.isFinite)) return null;
  if (mapping.output_width <= 0 || mapping.output_height <= 0 || Math.abs(a * d - b * c) <= Number.EPSILON) return null;
  return {
    a,
    b,
    c,
    d,
    e,
    f,
    width: mapping.output_width,
    height: mapping.output_height,
    topologySafe: mapping.topology_safe,
  };
}

export function createPipelineSharedCursor(
  activeVisualizationId: string,
  label: string,
  localX: number,
  localY: number,
  mapping: PipelineCoordinateMappingLike | null | undefined,
  activeWidth = mapping?.output_width ?? 0,
  activeHeight = mapping?.output_height ?? 0,
): PipelineSharedCursor {
  const sourcePoint = mapping?.kind !== 'unavailable' && mapping?.output_to_source
    ? applyAffine([localX, localY], mapping.output_to_source)
    : null;
  return {
    activeVisualizationId,
    label,
    localX,
    localY,
    activeWidth,
    activeHeight,
    sourceX: sourcePoint?.[0] ?? null,
    sourceY: sourcePoint?.[1] ?? null,
  };
}

export function pipelineSharedCursorPointForPane(
  cursor: PipelineSharedCursor,
  visualizationId: string,
  width: number,
  height: number,
  mapping: PipelineCoordinateMappingLike | null | undefined,
): { x: number; y: number } | null {
  if (cursor.activeVisualizationId !== visualizationId && (cursor.sourceX === null || cursor.sourceY === null)) return null;
  if (mapping?.kind === 'unavailable' && cursor.activeVisualizationId !== visualizationId) return null;
  const mappedPoint = cursor.activeVisualizationId === visualizationId
    ? [cursor.localX, cursor.localY] as [number, number]
    : cursor.sourceX !== null && cursor.sourceY !== null && mapping?.kind !== 'unavailable' && mapping?.source_to_output
      ? applyAffine([cursor.sourceX, cursor.sourceY], mapping.source_to_output)
      : null;
  const point = mappedPoint
    && mappedPoint[0] >= 0 && mappedPoint[1] >= 0 && mappedPoint[0] < width && mappedPoint[1] < height
    ? mappedPoint
    : cursor.activeWidth > 0 && cursor.activeHeight > 0
      ? [cursor.localX / cursor.activeWidth * width, cursor.localY / cursor.activeHeight * height] as [number, number]
      : null;
  if (!point || point[0] < 0 || point[1] < 0 || point[0] >= width || point[1] >= height) return null;
  return { x: point[0], y: point[1] };
}

export function canHidePipelineLayer(layerId: string, layers: Record<string, { visible: boolean }>): boolean {
  if (!layers[layerId]?.visible) return true;
  return Object.values(layers).filter((layer) => layer.visible).length > 1;
}
