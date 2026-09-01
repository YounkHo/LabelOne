export type AnnotationPoint = [number, number];

export type EditableAnnotationShape = {
  label: string;
  shape_type: string;
  points: number[][];
  direction?: number;
  [key: string]: unknown;
};

const supportedShapePoints: Record<string, { minimum: number; maximum?: number }> = {
  point: { minimum: 1, maximum: 1 },
  rectangle: { minimum: 2, maximum: 4 },
  rotation: { minimum: 4, maximum: 4 },
  quadrilateral: { minimum: 4, maximum: 4 },
  line: { minimum: 2, maximum: 2 },
  circle: { minimum: 2, maximum: 2 },
  polygon: { minimum: 3 },
  linestrip: { minimum: 2 },
  cuboid: { minimum: 8, maximum: 8 },
};

export function clampAnnotationPoint(point: AnnotationPoint, width: number, height: number): AnnotationPoint {
  return [Math.max(0, Math.min(width, point[0])), Math.max(0, Math.min(height, point[1]))];
}

export function rectangleCorners(first: AnnotationPoint, second: AnnotationPoint): number[][] {
  const left = Math.min(first[0], second[0]);
  const right = Math.max(first[0], second[0]);
  const top = Math.min(first[1], second[1]);
  const bottom = Math.max(first[1], second[1]);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

export function rotationDirection(points: number[][]): number {
  if (points.length < 2) return 0;
  const direction = Math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]);
  return direction < 0 ? direction + Math.PI * 2 : direction;
}

export function canClosePolygonAtPoint(points: AnnotationPoint[], pointer: AnnotationPoint, tolerance: number): boolean {
  if (points.length < 3 || !Number.isFinite(tolerance) || tolerance <= 0) return false;
  return Math.hypot(pointer[0] - points[0][0], pointer[1] - points[0][1]) <= tolerance;
}

export function annotationShapeClass(shapeType: string): string {
  const normalized = shapeType.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
  return `shape-${normalized || 'unknown'}`;
}

export function editableControlPointIndexes(shape: Pick<EditableAnnotationShape, 'shape_type' | 'points'>): number[] {
  if (shape.shape_type === 'rectangle') {
    if (shape.points.length === 2) return [0, 1];
    if (shape.points.length >= 4) return [0, 2];
  }
  if (shape.shape_type === 'rotation' && shape.points.length >= 4) return [0, 2];
  return shape.points.map((_, index) => index);
}

export function rotationCenter(points: number[][]): AnnotationPoint | null {
  if (points.length !== 4) return null;
  return [(points[0][0] + points[2][0]) / 2, (points[0][1] + points[2][1]) / 2];
}

export function rotationCornerHandle(points: number[][]): AnnotationPoint | null {
  if (points.length !== 4) return null;
  return [points[1][0], points[1][1]];
}

export function polygonVertexControlPath(points: number[][], segmentLength: number): string {
  if (!Number.isFinite(segmentLength) || segmentLength <= 0) return '';
  return points.map(([x, y]) => `M${x} ${y}h${segmentLength}`).join(' ');
}

export function rotateRotationShape(shape: EditableAnnotationShape, radians: number): EditableAnnotationShape {
  if (shape.shape_type !== 'rotation' || shape.points.length !== 4 || !Number.isFinite(radians)) return shape;
  const centerX = (shape.points[0][0] + shape.points[2][0]) / 2;
  const centerY = (shape.points[0][1] + shape.points[2][1]) / 2;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const points = shape.points.map(([x, y]) => {
    const dx = x - centerX;
    const dy = y - centerY;
    return [centerX + dx * cosine - dy * sine, centerY + dx * sine + dy * cosine];
  });
  return { ...shape, points, direction: rotationDirection(points) };
}

export function createDragShape(
  tool: 'rect' | 'rotation' | 'line' | 'circle',
  start: AnnotationPoint,
  end: AnnotationPoint,
  label = 'object',
): EditableAnnotationShape | null {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  if (Math.hypot(dx, dy) < 2) return null;
  if (tool === 'line') return { label, shape_type: 'line', points: [start, end] };
  if (tool === 'circle') return { label, shape_type: 'circle', points: [start, end] };
  const points = rectangleCorners(start, end);
  if (Math.abs(dx) < 2 || Math.abs(dy) < 2) return null;
  return {
    label,
    shape_type: tool === 'rotation' ? 'rotation' : 'rectangle',
    points,
    ...(tool === 'rotation' ? { direction: rotationDirection(points) } : {}),
  };
}

export function compactFreehandPoints(points: AnnotationPoint[], minimumDistance = 1): AnnotationPoint[] {
  const compacted: AnnotationPoint[] = [];
  for (const point of points) {
    const previous = compacted.at(-1);
    if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= minimumDistance) compacted.push(point);
  }
  return compacted;
}

export function polygonArea(points: number[][]): number {
  if (points.length < 3) return 0;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current[0] * next[1] - next[0] * current[1];
  }
  return Math.abs(twiceArea) / 2;
}

export type AnnotationShapeHit = {
  zone: 'edge' | 'interior';
  distance: number;
  area: number;
};

function pointDistance(first: AnnotationPoint, second: AnnotationPoint): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

function pointToSegmentDistance(point: AnnotationPoint, start: AnnotationPoint, end: AnnotationPoint): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= Number.EPSILON) return pointDistance(point, start);
  const amount = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
  return pointDistance(point, [start[0] + amount * dx, start[1] + amount * dy]);
}

function pathDistance(points: AnnotationPoint[], point: AnnotationPoint, closed: boolean): number {
  if (points.length === 0) return Number.POSITIVE_INFINITY;
  if (points.length === 1) return pointDistance(points[0], point);
  let distance = Number.POSITIVE_INFINITY;
  const segmentCount = closed ? points.length : points.length - 1;
  for (let index = 0; index < segmentCount; index += 1) {
    distance = Math.min(distance, pointToSegmentDistance(point, points[index], points[(index + 1) % points.length]));
  }
  return distance;
}

function pointInPolygon(point: AnnotationPoint, points: AnnotationPoint[]): boolean {
  let inside = false;
  for (let index = 0, previousIndex = points.length - 1; index < points.length; previousIndex = index, index += 1) {
    const current = points[index];
    const previous = points[previousIndex];
    const crosses = (current[1] > point[1]) !== (previous[1] > point[1])
      && point[0] < (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1]) + current[0];
    if (crosses) inside = !inside;
  }
  return inside;
}

function shapePoints(shape: Pick<EditableAnnotationShape, 'shape_type' | 'points'>): AnnotationPoint[] {
  const points = shape.points.filter(finitePoint).map(([x, y]) => [x, y] as AnnotationPoint);
  if (shape.shape_type === 'rectangle' && points.length === 2) return rectangleCorners(points[0], points[1]) as AnnotationPoint[];
  return points;
}

export function hitTestAnnotationShape(
  shape: Pick<EditableAnnotationShape, 'shape_type' | 'points'>,
  point: AnnotationPoint,
  tolerance: number,
): AnnotationShapeHit | null {
  if (!finitePoint(point)) return null;
  const hitTolerance = Number.isFinite(tolerance) ? Math.max(0, tolerance) : 0;
  const points = shapePoints(shape);
  if (points.length === 0) return null;
  if (shape.shape_type === 'point') {
    const distance = pointDistance(point, points[0]);
    return distance <= hitTolerance ? { zone: 'edge', distance, area: 0 } : null;
  }
  if (shape.shape_type === 'line' || shape.shape_type === 'linestrip') {
    const distance = pathDistance(points, point, false);
    return distance <= hitTolerance ? { zone: 'edge', distance, area: 0 } : null;
  }
  if (shape.shape_type === 'circle' && points.length >= 2) {
    const radius = pointDistance(points[0], points[1]);
    const centerDistance = pointDistance(points[0], point);
    const distance = Math.abs(centerDistance - radius);
    if (distance <= hitTolerance) return { zone: 'edge', distance, area: Math.PI * radius * radius };
    return centerDistance < radius ? { zone: 'interior', distance, area: Math.PI * radius * radius } : null;
  }
  if (points.length < 3) return null;
  const distance = pathDistance(points, point, true);
  const area = polygonArea(points);
  if (distance <= hitTolerance) return { zone: 'edge', distance, area };
  return pointInPolygon(point, points) ? { zone: 'interior', distance, area } : null;
}

export function annotationHitCandidates(
  shapes: Array<Pick<EditableAnnotationShape, 'shape_type' | 'points'>>,
  point: AnnotationPoint,
  tolerance: number,
  hiddenIndexes: ReadonlySet<number> = new Set<number>(),
): number[] {
  return shapes
    .flatMap((shape, index) => {
      if (hiddenIndexes.has(index)) return [];
      const hit = hitTestAnnotationShape(shape, point, tolerance);
      return hit ? [{ index, ...hit }] : [];
    })
    .sort((first, second) => {
      if (first.zone !== second.zone) return first.zone === 'edge' ? -1 : 1;
      if (first.zone === 'edge' && first.distance !== second.distance) return first.distance - second.distance;
      if (first.area !== second.area) return first.area - second.area;
      if (first.distance !== second.distance) return first.distance - second.distance;
      return second.index - first.index;
    })
    .map(({ index }) => index);
}

export function selectAnnotationHitIndex(
  candidates: number[],
  selectedIndex: number | null,
  advance: boolean,
): number | null {
  if (candidates.length === 0) return null;
  if (!advance) return candidates[0];
  const selectedPosition = selectedIndex === null ? -1 : candidates.indexOf(selectedIndex);
  if (selectedPosition < 0) return candidates[0];
  return candidates[(selectedPosition + 1) % candidates.length];
}

export function createFreehandLine(
  inputPoints: AnnotationPoint[],
  imageWidth: number,
  imageHeight: number,
  pointBudget = 10_000,
): EditableAnnotationShape | null {
  if (!Number.isFinite(imageWidth) || !Number.isFinite(imageHeight) || imageWidth <= 0 || imageHeight <= 0 || !Number.isInteger(pointBudget) || pointBudget < 2) return null;
  const compacted = compactFreehandPoints(inputPoints.map((point) => clampAnnotationPoint(point, imageWidth, imageHeight)), 0.5);
  if (compacted.length < 2) return null;
  const points = compacted.length <= pointBudget
    ? compacted
    : Array.from({ length: pointBudget }, (_, index) => compacted[Math.round(index * (compacted.length - 1) / (pointBudget - 1))]);
  const totalLength = points.slice(1).reduce((sum, point, index) => sum + Math.hypot(point[0] - points[index][0], point[1] - points[index][1]), 0);
  if (totalLength < 2) return null;
  return { label: 'object', shape_type: 'linestrip', points };
}

export function translateShapeWithinImage(
  shape: EditableAnnotationShape,
  requestedDx: number,
  requestedDy: number,
  width: number,
  height: number,
): EditableAnnotationShape {
  const xs = shape.points.map((point) => point[0]);
  const ys = shape.points.map((point) => point[1]);
  const dx = Math.max(-Math.min(...xs), Math.min(width - Math.max(...xs), requestedDx));
  const dy = Math.max(-Math.min(...ys), Math.min(height - Math.max(...ys), requestedDy));
  return { ...shape, points: shape.points.map(([x, y]) => [x + dx, y + dy]) };
}

export function moveShapeControlPoint(
  shape: EditableAnnotationShape,
  pointIndex: number,
  point: AnnotationPoint,
  width: number,
  height: number,
): EditableAnnotationShape {
  if (pointIndex < 0 || pointIndex >= shape.points.length) return shape;
  const target = clampAnnotationPoint(point, width, height);
  if (shape.shape_type === 'rectangle' && shape.points.length === 4) {
    const opposite = shape.points[(pointIndex + 2) % 4] as AnnotationPoint;
    return { ...shape, points: rectangleCorners(target, opposite) };
  }
  if (shape.shape_type === 'rotation' && shape.points.length === 4) {
    const oppositeIndex = (pointIndex + 2) % 4;
    const nextIndex = (pointIndex + 1) % 4;
    const previousIndex = (pointIndex + 3) % 4;
    const opposite = shape.points[oppositeIndex] as AnnotationPoint;
    const nextVector: AnnotationPoint = [shape.points[nextIndex][0] - shape.points[pointIndex][0], shape.points[nextIndex][1] - shape.points[pointIndex][1]];
    const previousVector: AnnotationPoint = [shape.points[previousIndex][0] - shape.points[pointIndex][0], shape.points[previousIndex][1] - shape.points[pointIndex][1]];
    const nextLength = Math.hypot(...nextVector);
    const previousLength = Math.hypot(...previousVector);
    if (nextLength < 1e-6 || previousLength < 1e-6) return shape;
    const nextUnit: AnnotationPoint = [nextVector[0] / nextLength, nextVector[1] / nextLength];
    const previousUnit: AnnotationPoint = [previousVector[0] / previousLength, previousVector[1] / previousLength];
    const diagonal: AnnotationPoint = [target[0] - opposite[0], target[1] - opposite[1]];
    const nextProjection = diagonal[0] * nextUnit[0] + diagonal[1] * nextUnit[1];
    const previousProjection = diagonal[0] * previousUnit[0] + diagonal[1] * previousUnit[1];
    if (Math.abs(nextProjection) < 2 || Math.abs(previousProjection) < 2) return shape;
    const points = shape.points.map((value) => [...value]);
    points[pointIndex] = target;
    points[oppositeIndex] = [...opposite];
    points[nextIndex] = [opposite[0] + previousProjection * previousUnit[0], opposite[1] + previousProjection * previousUnit[1]];
    points[previousIndex] = [opposite[0] + nextProjection * nextUnit[0], opposite[1] + nextProjection * nextUnit[1]];
    return { ...shape, points, direction: rotationDirection(points) };
  }
  const points = shape.points.map((value, index) => index === pointIndex ? [...target] : [...value]);
  return {
    ...shape,
    points,
    ...(shape.shape_type === 'rotation' ? { direction: rotationDirection(points) } : {}),
  };
}

function finitePoint(value: unknown): value is AnnotationPoint {
  return Array.isArray(value)
    && value.length === 2
    && value.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate));
}

export function validateImportedAnnotationDocument(payload: unknown, width: number, height: number): Record<string, unknown> & { shapes: EditableAnnotationShape[] } {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('标注 JSON 根节点必须是 object');
  const document = payload as Record<string, unknown>;
  if (!Array.isArray(document.shapes)) throw new Error('标注 JSON 必须包含 shapes 数组');
  if (document.imageWidth !== undefined && document.imageWidth !== width) throw new Error(`imageWidth 与当前图不一致：${String(document.imageWidth)} ≠ ${width}`);
  if (document.imageHeight !== undefined && document.imageHeight !== height) throw new Error(`imageHeight 与当前图不一致：${String(document.imageHeight)} ≠ ${height}`);
  const shapes = document.shapes.map((raw, shapeIndex) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error(`shapes[${shapeIndex}] 必须是 object`);
    const shape = raw as EditableAnnotationShape;
    if (typeof shape.label !== 'string' || !shape.label.trim()) throw new Error(`shapes[${shapeIndex}].label 不能为空`);
    const rule = supportedShapePoints[shape.shape_type];
    if (!rule) throw new Error(`shapes[${shapeIndex}] 使用不支持的 shape_type：${String(shape.shape_type)}`);
    if (!Array.isArray(shape.points) || shape.points.some((point) => !finitePoint(point))) throw new Error(`shapes[${shapeIndex}].points 必须是有限数值坐标`);
    if (shape.points.length < rule.minimum || (rule.maximum !== undefined && shape.shape_type !== 'rectangle' && shape.points.length !== rule.maximum) || (shape.shape_type === 'rectangle' && ![2, 4].includes(shape.points.length))) {
      throw new Error(`shapes[${shapeIndex}] 点数不符合 ${shape.shape_type} 要求`);
    }
    if (shape.points.some(([x, y]) => x < 0 || y < 0 || x > width || y > height)) throw new Error(`shapes[${shapeIndex}] 存在越界坐标`);
    const uniquePoints = new Set(shape.points.map(([x, y]) => `${x}:${y}`));
    if (shape.shape_type === 'polygon' && uniquePoints.size < 3) throw new Error(`shapes[${shapeIndex}] 多边形至少需要 3 个不同坐标`);
    if (shape.shape_type === 'rotation' && Math.hypot(shape.points[1][0] - shape.points[0][0], shape.points[1][1] - shape.points[0][1]) < 1) throw new Error(`shapes[${shapeIndex}] 旋转框首边无效`);
    if (shape.shape_type === 'circle' && Math.hypot(shape.points[1][0] - shape.points[0][0], shape.points[1][1] - shape.points[0][1]) < 1) throw new Error(`shapes[${shapeIndex}] 圆半径无效`);
    if (shape.shape_type === 'line' && shape.points[0][0] === shape.points[1][0] && shape.points[0][1] === shape.points[1][1]) throw new Error(`shapes[${shapeIndex}] 直线长度无效`);
    const points = shape.points.map(([x, y]) => [x, y]);
    return {
      ...shape,
      label: shape.label.trim(),
      points,
      ...(shape.shape_type === 'rotation' ? { direction: rotationDirection(points) } : {}),
    };
  });
  return { ...document, imageWidth: width, imageHeight: height, shapes };
}
