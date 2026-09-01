export type CanvasDrawingTool = 'rect' | 'rotation' | 'polygon' | 'point' | 'line' | 'circle' | 'brush';
export type CanvasFlipAxes = { horizontal: boolean; vertical: boolean };

export function resolvePipelineFlipAxes(
  nodes: Array<{ id?: string; kind: string; enabled?: boolean; parameters?: Record<string, unknown> }>,
  visualizationId?: string,
): CanvasFlipAxes | null {
  const axes: CanvasFlipAxes = { horizontal: false, vertical: false };
  for (const node of nodes) {
    if (visualizationId && node.id === visualizationId) break;
    if (node.enabled === false || ['source', 'visualize', 'output', 'color', 'noise'].includes(node.kind)) continue;
    if (node.kind !== 'flip') return null;
    const axis = String(node.parameters?.axis ?? 'horizontal');
    if (axis === 'horizontal') axes.horizontal = !axes.horizontal;
    else if (axis === 'vertical') axes.vertical = !axes.vertical;
    else return null;
  }
  return axes;
}

export function flipCanvasPoint(point: [number, number], width: number, height: number, axes: CanvasFlipAxes): [number, number] {
  return [axes.horizontal ? width - point[0] : point[0], axes.vertical ? height - point[1] : point[1]];
}

export function flipCanvasDelta(delta: [number, number], axes: CanvasFlipAxes): [number, number] {
  return [axes.horizontal ? -delta[0] : delta[0], axes.vertical ? -delta[1] : delta[1]];
}

export function flipCanvasShape<T extends { points: number[][] }>(shape: T, width: number, height: number, axes: CanvasFlipAxes): T {
  return { ...shape, points: shape.points.map((point) => flipCanvasPoint([point[0], point[1]], width, height, axes)) };
}

export type CanvasCoordinateTransform = {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
  width: number;
  height: number;
  topologySafe: boolean;
};

type PipelineTransformNode = { id?: string; kind: string; enabled?: boolean; parameters?: Record<string, unknown> };

function composeCanvasTransform(current: CanvasCoordinateTransform, next: Omit<CanvasCoordinateTransform, 'topologySafe'>, topologySafe = true): CanvasCoordinateTransform {
  return {
    a: next.a * current.a + next.c * current.b,
    b: next.b * current.a + next.d * current.b,
    c: next.a * current.c + next.c * current.d,
    d: next.b * current.c + next.d * current.d,
    e: next.a * current.e + next.c * current.f + next.e,
    f: next.b * current.e + next.d * current.f + next.f,
    width: next.width,
    height: next.height,
    topologySafe: current.topologySafe && topologySafe,
  };
}

function finiteParameter(parameters: Record<string, unknown> | undefined, name: string, fallback: number): number {
  const value = parameters?.[name];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolvePipelineCoordinateTransform(
  nodes: PipelineTransformNode[],
  sourceWidth: number,
  sourceHeight: number,
  visualizationId?: string,
): CanvasCoordinateTransform | null {
  if (![sourceWidth, sourceHeight].every((value) => Number.isFinite(value) && value > 0)) return null;
  let transform: CanvasCoordinateTransform = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0, width: sourceWidth, height: sourceHeight, topologySafe: true };
  for (const node of nodes) {
    if (visualizationId && node.id === visualizationId) break;
    if (node.enabled === false || ['source', 'visualize', 'output', 'color', 'noise'].includes(node.kind) || node.kind.startsWith('opencv.')) continue;
    if (node.kind === 'crop') {
      const margin = finiteParameter(node.parameters, 'margin_ratio', 0.05);
      const x = finiteParameter(node.parameters, 'x', Math.round(transform.width * margin));
      const y = finiteParameter(node.parameters, 'y', Math.round(transform.height * margin));
      const width = finiteParameter(node.parameters, 'width', Math.round(transform.width * (1 - margin * 2)));
      const height = finiteParameter(node.parameters, 'height', Math.round(transform.height * (1 - margin * 2)));
      if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > transform.width || y + height > transform.height) return null;
      transform = composeCanvasTransform(transform, { a: 1, b: 0, c: 0, d: 1, e: -x, f: -y, width, height }, false);
      continue;
    }
    if (node.kind === 'resize') {
      const width = finiteParameter(node.parameters, 'width', transform.width);
      const height = finiteParameter(node.parameters, 'height', transform.height);
      if (width <= 0 || height <= 0) return null;
      const scaleX = width / transform.width;
      const scaleY = height / transform.height;
      transform = composeCanvasTransform(transform, { a: scaleX, b: 0, c: 0, d: scaleY, e: 0, f: 0, width, height }, Math.abs(scaleX - scaleY) <= 1e-12);
      continue;
    }
    if (node.kind === 'flip') {
      const axis = String(node.parameters?.axis ?? 'horizontal');
      if (axis === 'horizontal') transform = composeCanvasTransform(transform, { a: -1, b: 0, c: 0, d: 1, e: transform.width, f: 0, width: transform.width, height: transform.height });
      else if (axis === 'vertical') transform = composeCanvasTransform(transform, { a: 1, b: 0, c: 0, d: -1, e: 0, f: transform.height, width: transform.width, height: transform.height });
      else return null;
      continue;
    }
    if (node.kind === 'rotate') {
      const degrees = ((Math.round(finiteParameter(node.parameters, 'degrees', 90)) % 360) + 360) % 360;
      if (degrees === 0) continue;
      if (degrees === 90) transform = composeCanvasTransform(transform, { a: 0, b: 1, c: -1, d: 0, e: transform.height, f: 0, width: transform.height, height: transform.width });
      else if (degrees === 180) transform = composeCanvasTransform(transform, { a: -1, b: 0, c: 0, d: -1, e: transform.width, f: transform.height, width: transform.width, height: transform.height });
      else if (degrees === 270) transform = composeCanvasTransform(transform, { a: 0, b: -1, c: 1, d: 0, e: 0, f: transform.width, width: transform.height, height: transform.width });
      else return null;
      continue;
    }
    return null;
  }
  return transform;
}

export function transformCanvasPoint(point: [number, number], transform: CanvasCoordinateTransform): [number, number] {
  return [transform.a * point[0] + transform.c * point[1] + transform.e, transform.b * point[0] + transform.d * point[1] + transform.f];
}

export function inverseTransformCanvasPoint(point: [number, number], transform: CanvasCoordinateTransform): [number, number] {
  const determinant = transform.a * transform.d - transform.b * transform.c;
  if (Math.abs(determinant) <= Number.EPSILON) return point;
  const x = point[0] - transform.e;
  const y = point[1] - transform.f;
  return [(transform.d * x - transform.c * y) / determinant, (-transform.b * x + transform.a * y) / determinant];
}

export function inverseTransformCanvasDelta(delta: [number, number], transform: CanvasCoordinateTransform): [number, number] {
  const determinant = transform.a * transform.d - transform.b * transform.c;
  if (Math.abs(determinant) <= Number.EPSILON) return delta;
  return [(transform.d * delta[0] - transform.c * delta[1]) / determinant, (-transform.b * delta[0] + transform.a * delta[1]) / determinant];
}

export function transformCanvasShape<T extends { points: number[][] }>(shape: T, transform: CanvasCoordinateTransform): T {
  return { ...shape, points: shape.points.map((point) => transformCanvasPoint([point[0], point[1]], transform)) };
}

export function inverseTransformCanvasShape<T extends { points: number[][] }>(shape: T, transform: CanvasCoordinateTransform): T {
  return { ...shape, points: shape.points.map((point) => inverseTransformCanvasPoint([point[0], point[1]], transform)) };
}

export function shouldSwitchToSelectAfterBlankClick(input: {
  armed: boolean;
  tool: string;
  button: number;
  spaceDown: boolean;
  blankTarget: boolean;
}): boolean {
  return input.armed
    && input.button === 0
    && !input.spaceDown
    && input.blankTarget
    && ['rect', 'rotation', 'polygon', 'point', 'line', 'circle', 'brush'].includes(input.tool);
}

export function resolveCanvasPresentation<T>(input: {
  sourceImageUrl: string | null;
  pipelineImageUrl: string | null;
  pipelineEnabled: boolean;
  pipelineScope: 'all' | 'current';
  annotationShapes: T[];
  pipelineAnnotationShapes?: T[];
  sourceWidth?: number;
  sourceHeight?: number;
  pipelineWidth?: number;
  pipelineHeight?: number;
}) {
  const pipelineDimensionsValid = Number.isFinite(input.pipelineWidth)
    && Number.isFinite(input.pipelineHeight)
    && Number(input.pipelineWidth) > 0
    && Number(input.pipelineHeight) > 0;
  const showingPipelineImage = Boolean(
    input.pipelineEnabled
    && input.pipelineImageUrl
    && input.pipelineAnnotationShapes !== undefined
    && pipelineDimensionsValid
  );
  return {
    imageUrl: showingPipelineImage ? input.pipelineImageUrl : input.sourceImageUrl,
    showingPipelineImage,
    shapes: showingPipelineImage ? input.pipelineAnnotationShapes ?? [] : input.annotationShapes,
    width: showingPipelineImage ? input.pipelineWidth : input.sourceWidth,
    height: showingPipelineImage ? input.pipelineHeight : input.sourceHeight,
  };
}
