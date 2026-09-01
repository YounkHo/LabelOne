import assert from 'node:assert/strict';
import test from 'node:test';

import {
  annotationHitCandidates,
  annotationShapeClass,
  canClosePolygonAtPoint,
  compactFreehandPoints,
  createDragShape,
  createFreehandLine,
  editableControlPointIndexes,
  moveShapeControlPoint,
  polygonVertexControlPath,
  rotateRotationShape,
  rotationCenter,
  rotationCornerHandle,
  selectAnnotationHitIndex,
  translateShapeWithinImage,
  validateImportedAnnotationDocument,
} from './annotation-tools.ts';

test('nested annotation hit testing prefers the contained frame independent of paint order', () => {
  const outer = { label: 'outer', shape_type: 'rectangle', points: [[0, 0], [100, 100]] };
  const inner = { label: 'inner', shape_type: 'rectangle', points: [[30, 30], [70, 70]] };
  assert.deepEqual(annotationHitCandidates([inner, outer], [50, 50], 2), [0, 1]);
  assert.deepEqual(annotationHitCandidates([outer, inner], [50, 50], 2), [1, 0]);
  assert.deepEqual(annotationHitCandidates([outer, inner], [10, 10], 2), [0]);
  assert.deepEqual(annotationHitCandidates([outer, inner], [50, 50], 2, new Set([1])), [0]);
});

test('annotation hit testing prioritizes nearby boundaries and supports all primary geometry types', () => {
  const overlapping = [
    { label: 'a', shape_type: 'rectangle', points: [[0, 0], [70, 70]] },
    { label: 'b', shape_type: 'rectangle', points: [[30, 30], [100, 100]] },
  ];
  assert.deepEqual(annotationHitCandidates(overlapping, [45, 45], 2), [1, 0]);
  assert.deepEqual(annotationHitCandidates(overlapping, [50, 50], 2), [1, 0]);
  assert.deepEqual(annotationHitCandidates([
    { label: 'four-point', shape_type: 'rectangle', points: [[0, 0], [20, 0], [20, 20], [0, 20]] },
    { label: 'circle', shape_type: 'circle', points: [[50, 50], [60, 50]] },
    { label: 'line', shape_type: 'line', points: [[70, 10], [90, 10]] },
    { label: 'point', shape_type: 'point', points: [[100, 10]] },
  ], [20, 10], 1), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'line', shape_type: 'line', points: [[70, 10], [90, 10]] }], [80, 12], 2), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'point', shape_type: 'point', points: [[100, 10]] }], [103, 10], 3), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'circle', shape_type: 'circle', points: [[50, 50], [60, 50]] }], [50, 50], 1), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'rotation', shape_type: 'rotation', points: [[50, 30], [70, 50], [50, 70], [30, 50]] }], [50, 50], 1), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'polygon', shape_type: 'polygon', points: [[0, 0], [20, 0], [10, 20]] }], [10, 10], 1), [0]);
  assert.deepEqual(annotationHitCandidates([{ label: 'line', shape_type: 'line', points: [[70, 10], [90, 10]] }], [80, 12.1], 2), []);
});

test('ordinary annotation hits choose the nearest frame and repeated hits cycle', () => {
  const nestedCandidates = [1, 0];
  assert.equal(selectAnnotationHitIndex(nestedCandidates, null, false), 1);
  assert.equal(selectAnnotationHitIndex(nestedCandidates, 0, false), 1);
  assert.equal(selectAnnotationHitIndex(nestedCandidates, 1, true), 0);
  assert.equal(selectAnnotationHitIndex(nestedCandidates, 0, true), 1);
  assert.equal(selectAnnotationHitIndex([], 0, true), null);
});

test('shape presentation helpers expose stable colors and geometry-specific handles', () => {
  assert.equal(annotationShapeClass('Rotated Rectangle'), 'shape-rotated-rectangle');
  assert.equal(annotationShapeClass(''), 'shape-unknown');
  assert.deepEqual(editableControlPointIndexes({ shape_type: 'rectangle', points: [[0, 0], [10, 10]] }), [0, 1]);
  assert.deepEqual(editableControlPointIndexes({ shape_type: 'rectangle', points: [[0, 0], [10, 0], [10, 10], [0, 10]] }), [0, 2]);
  assert.deepEqual(editableControlPointIndexes({ shape_type: 'rotation', points: [[0, 0], [10, 0], [10, 10], [0, 10]] }), [0, 2]);
  assert.deepEqual(editableControlPointIndexes({ shape_type: 'polygon', points: [[0, 0], [10, 0], [10, 10], [3, 8]] }), [0, 1, 2, 3]);
  assert.deepEqual(rotationCenter([[10, 10], [30, 10], [30, 20], [10, 20]]), [20, 15]);
  assert.deepEqual(rotationCornerHandle([[10, 10], [30, 10], [30, 20], [10, 20]]), [30, 10]);
  const manyVertices = Array.from({ length: 10_000 }, (_, index) => [index, index % 37]);
  assert.equal((polygonVertexControlPath(manyVertices, 0.001).match(/M/g) ?? []).length, 10_000);
});

test('polygon closure uses a tolerant first-point target only after three points', () => {
  const points: [number, number][] = [[10, 10], [50, 10], [50, 50]];
  assert.equal(canClosePolygonAtPoint(points.slice(0, 2), [11, 11], 14), false);
  assert.equal(canClosePolygonAtPoint(points, [20, 20], 14.2), true);
  assert.equal(canClosePolygonAtPoint(points, [21, 21], 14), false);
  assert.equal(canClosePolygonAtPoint(points, [10, 10], 0), false);
});

test('creates serializable rectangle, rotation, line and circle shapes', () => {
  assert.deepEqual(createDragShape('rect', [10, 20], [30, 40]), {
    label: 'object',
    shape_type: 'rectangle',
    points: [[10, 20], [30, 20], [30, 40], [10, 40]],
  });
  assert.deepEqual(createDragShape('rotation', [10, 20], [30, 40]), {
    label: 'object',
    shape_type: 'rotation',
    points: [[10, 20], [30, 20], [30, 40], [10, 40]],
    direction: 0,
  });
  assert.equal(createDragShape('line', [1, 2], [8, 9])?.points.length, 2);
  assert.equal(createDragShape('circle', [5, 5], [10, 5])?.shape_type, 'circle');
});

test('moves complete shapes without crossing image bounds', () => {
  const moved = translateShapeWithinImage({ label: 'x', shape_type: 'line', points: [[90, 90], [100, 100]] }, 20, 30, 100, 100);
  assert.deepEqual(moved.points, [[90, 90], [100, 100]]);
});

test('rectangle and rotation control points preserve their rectangle geometry', () => {
  const rectangle = moveShapeControlPoint({ label: 'x', shape_type: 'rectangle', points: [[0, 0], [10, 0], [10, 10], [0, 10]] }, 0, [2, 3], 100, 100);
  assert.deepEqual(rectangle.points, [[2, 3], [10, 3], [10, 10], [2, 10]]);
  const initial = rotateRotationShape({ label: 'x', shape_type: 'rotation', points: [[20, 20], [40, 20], [40, 30], [20, 30]], direction: 0 }, Math.PI / 4);
  for (let index = 0; index < 4; index += 1) {
    const opposite = initial.points[(index + 2) % 4];
    const source = initial.points[index];
    const rotation = moveShapeControlPoint(initial, index, [source[0] + 2.5, source[1] - 1.5], 100, 100);
    assert.deepEqual(rotation.points[(index + 2) % 4], opposite);
    const next = [rotation.points[(index + 1) % 4][0] - rotation.points[index][0], rotation.points[(index + 1) % 4][1] - rotation.points[index][1]];
    const previous = [rotation.points[(index + 3) % 4][0] - rotation.points[index][0], rotation.points[(index + 3) % 4][1] - rotation.points[index][1]];
    assert.ok(Math.abs(next[0] * previous[0] + next[1] * previous[1]) < 1e-9);
    assert.ok(Math.hypot(...next) >= 2);
    assert.ok(Math.hypot(...previous) >= 2);
  }
});

test('center rotation keeps the four-point OBB centered', () => {
  const shape = createDragShape('rotation', [40, 40], [80, 60])!;
  assert.deepEqual(rotationCenter(shape.points), [60, 50]);
  const rotated = rotateRotationShape(shape, Math.PI / 2);
  assert.deepEqual(rotationCenter(rotated.points), [60, 50]);
  assert.ok(Math.abs(rotated.direction! - Math.PI / 2) < 1e-9);
  const expected = [[70, 30], [70, 70], [50, 70], [50, 30]];
  rotated.points.forEach((point, index) => {
    assert.ok(Math.abs(point[0] - expected[index][0]) < 1e-9);
    assert.ok(Math.abs(point[1] - expected[index][1]) < 1e-9);
  });
});

test('freehand compaction drops near-identical samples', () => {
  assert.deepEqual(compactFreehandPoints([[0, 0], [0.1, 0.1], [2, 0]], 1), [[0, 0], [2, 0]]);
});

test('annotation import validates dimensions, types and bounds', () => {
  const valid = validateImportedAnnotationDocument({ shapes: [{ label: 'dot', shape_type: 'point', points: [[4, 5]] }] }, 20, 10);
  assert.equal(valid.shapes[0].label, 'dot');
  assert.throws(() => validateImportedAnnotationDocument({ imageWidth: 99, shapes: [] }, 20, 10), /imageWidth/);
  assert.throws(() => validateImportedAnnotationDocument({ shapes: [{ label: 'bad', shape_type: 'point', points: [[30, 5]] }] }, 20, 10), /越界/);
});

test('freehand drawing stays an open linestrip', () => {
  const shape = createFreehandLine([[10, 20], [40, 35], [90, 20]], 100, 100);
  assert.equal(shape?.shape_type, 'linestrip');
  assert.deepEqual(shape?.points, [[10, 20], [40, 35], [90, 20]]);
  assert.equal('brush_width' in (shape ?? {}), false);
  assert.notDeepEqual(shape?.points[0], shape?.points.at(-1));
});

test('freehand lines clamp sampled points inside image bounds', () => {
  const shape = createFreehandLine([[-2, 2], [50, 2], [70, 60]], 64, 64);
  assert.ok(shape);
  assert.equal(shape?.shape_type, 'linestrip');
  assert.equal(shape!.points.every(([x, y]) => x >= 0 && y >= 0 && x <= 64 && y <= 64), true);
});

test('freehand lines reject degenerate strokes and enforce the point budget', () => {
  assert.equal(createFreehandLine([[5, 5], [5.1, 5.1]], 100, 100), null);
  const many = Array.from({ length: 20_000 }, (_, index) => [index / 20, 50 + Math.sin(index / 20)] as [number, number]);
  const shape = createFreehandLine(many, 1000, 100, 200);
  assert.ok(shape);
  assert.equal(shape!.points.length, 200);
  assert.notDeepEqual(shape!.points[0], shape!.points.at(-1));
});
