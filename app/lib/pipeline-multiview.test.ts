import assert from 'node:assert/strict';
import test from 'node:test';

import { canHidePipelineLayer, containedPipelineImageRect, createPipelineSharedCursor, normalizedPipelinePan, pipelinePaneMetrics, pipelinePaneTransform, pipelinePaneVectorToReference, pipelineWheelInputToReference, pipelineSharedCursorPointForPane, resolvePipelineDisplayMode, snapPipelineGridCoordinate, stablePipelineDisplaySlots, updatePipelineLayerOpacity, updatePipelineLayerVisibility, type PipelineCoordinateMappingLike } from './pipeline-multiview.ts';

test('pixel grid coordinates snap to one-device-pixel centers at every DPR', () => {
  assert.equal(snapPipelineGridCoordinate(10.2, 1), 10.5);
  assert.equal(snapPipelineGridCoordinate(10.2, 2), 10.25);
  assert.equal(snapPipelineGridCoordinate(10.2, 3), 10.5 / 1); // 31.5 device pixels / 3
});

test('each pane derives label and true-pixel metrics from its own contained image', () => {
  const zoomed = pipelinePaneMetrics(400, 400, 400, 200, { scale: 4, x: 0, y: 0 }, 400, 400, true);
  assert.deepEqual(zoomed.contained, { left: 0, top: 100, width: 400, height: 200 });
  assert.equal(zoomed.imageUnitsPerScreenPixel, 0.25);
  assert.equal(zoomed.pixelWidthOnScreen, 4);
  assert.equal(zoomed.pixelHeightOnScreen, 4);
  assert.equal(zoomed.pixelGridVisible, false);
  const inspected = pipelinePaneMetrics(400, 400, 400, 200, { scale: 8, x: 0, y: 0 }, 400, 400, true);
  assert.equal(inspected.pixelWidthOnScreen, 8);
  assert.equal(inspected.pixelHeightOnScreen, 8);
  assert.equal(inspected.pixelGridVisible, true);
});

test('four overlay layers keep alpha and visibility independent at every endpoint', () => {
  const initial = Object.fromEntries(['a', 'b', 'c', 'd'].map((id) => [id, { visible: true, opacity: 50 }]));
  const opacities = [0, 23, 67, 100];
  const updated = ['a', 'b', 'c', 'd'].reduce((layers, id, index) => updatePipelineLayerOpacity(layers, id, opacities[index]), initial);
  assert.deepEqual(Object.fromEntries(Object.entries(updated).map(([id, layer]) => [id, layer.opacity])), { a: 0, b: 23, c: 67, d: 100 });
  const hidden = updatePipelineLayerVisibility(updated, 'd', false);
  const restored = updatePipelineLayerVisibility(hidden, 'd', true);
  assert.equal(hidden.d.visible, false);
  assert.deepEqual(restored.d, { visible: true, opacity: 100 });
  assert.deepEqual(restored.a, { visible: true, opacity: 0 });
});

test('shared pan is normalized against the reference pane', () => {
  assert.deepEqual(normalizedPipelinePan({ scale: 2, x: 50, y: -25 }, 200, 100), { x: 0.25, y: -0.25 });
  assert.equal(pipelinePaneTransform({ scale: 2, x: 50, y: -25 }, 200, 100), 'translate(25%, -25%) scale(2)');
});

test('split panes convert local zoom anchors and metrics through the full-canvas reference space', () => {
  assert.deepEqual(
    pipelinePaneVectorToReference({ x: 250, y: -200 }, 500, 400, 1000, 800),
    { x: 500, y: -400 },
  );
  const metrics = pipelinePaneMetrics(
    500,
    800,
    1000,
    800,
    { scale: 2, x: -250, y: 0 },
    1000,
    800,
  );
  assert.deepEqual(metrics.contained, { left: 0, top: 200, width: 500, height: 400 });
  assert.deepEqual(metrics.display, { left: -375, top: 0, width: 1000, height: 800 });
  assert.equal(pipelinePaneTransform({ scale: 2, x: -250, y: 0 }, 1000, 800), 'translate(-25%, 0%) scale(2)');
});

test('split panes scale pan deltas but keep trackpad zoom intensity independent of pane count', () => {
  const pan = { deltaX: 12, deltaY: -8, deltaMode: 0, ctrlKey: false, metaKey: false };
  assert.deepEqual(pipelineWheelInputToReference(pan, 500, 400, 1000, 800), { ...pan, deltaX: 24, deltaY: -16 });
  const zoom = { ...pan, ctrlKey: true };
  assert.equal(pipelineWheelInputToReference(zoom, 500, 400, 1000, 800), zoom);
});

test('overlay never allows hiding the final visible layer', () => {
  assert.equal(canHidePipelineLayer('a', { a: { visible: true }, b: { visible: true } }), true);
  assert.equal(canHidePipelineLayer('a', { a: { visible: true }, b: { visible: false } }), false);
  assert.equal(canHidePipelineLayer('b', { a: { visible: true }, b: { visible: false } }), true);
});

test('cursor maps through source coordinates across crop and resize panes', () => {
  const mapping = (
    id: string,
    width: number,
    height: number,
    sourceToOutput: [number, number, number, number, number, number],
    outputToSource: [number, number, number, number, number, number],
  ): PipelineCoordinateMappingLike => ({
    kind: id === 'source' ? 'identity' : 'affine',
    source_width: 1000,
    source_height: 800,
    output_width: width,
    output_height: height,
    source_to_output: sourceToOutput,
    output_to_source: outputToSource,
    coordinate_space_id: id,
    topology_safe: true,
  });
  const crop = mapping('crop', 400, 300, [1, 0, 0, 1, -100, -50], [1, 0, 0, 1, 100, 50]);
  const source = mapping('source', 1000, 800, [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]);
  const resize = mapping('resize', 2000, 1600, [2, 0, 0, 2, 0, 0], [.5, 0, 0, .5, 0, 0]);
  const otherCrop = mapping('other-crop', 200, 200, [1, 0, 0, 1, -200, -200], [1, 0, 0, 1, 200, 200]);
  const cursor = createPipelineSharedCursor('crop', 'Crop', 0, 0, crop);

  assert.deepEqual({ x: cursor.sourceX, y: cursor.sourceY }, { x: 100, y: 50 });
  assert.deepEqual(pipelineSharedCursorPointForPane(cursor, 'crop', 400, 300, crop), { x: 0, y: 0 });
  assert.deepEqual(pipelineSharedCursorPointForPane(cursor, 'source', 1000, 800, source), { x: 100, y: 50 });
  assert.deepEqual(pipelineSharedCursorPointForPane(cursor, 'resize', 2000, 1600, resize), { x: 200, y: 100 });
  assert.deepEqual(pipelineSharedCursorPointForPane(cursor, 'other-crop', 200, 200, otherCrop), { x: 0, y: 0 });
});

test('unavailable mappings keep a local cursor but never fake a cross-domain cursor', () => {
  const unavailable: PipelineCoordinateMappingLike = {
    kind: 'unavailable', source_width: 100, source_height: 100, output_width: 20, output_height: 10,
    coordinate_space_id: 'feature', topology_safe: false, reason: 'non-spatial',
  };
  const cursor = createPipelineSharedCursor('feature', 'Feature', 4, 5, unavailable);
  assert.equal(cursor.sourceX, null);
  assert.deepEqual(pipelineSharedCursorPointForPane(cursor, 'feature', 20, 10, unavailable), { x: 4, y: 5 });
  assert.equal(pipelineSharedCursorPointForPane(cursor, 'source', 100, 100, null), null);
});

test('contained image rect excludes horizontal and vertical letterbox areas', () => {
  assert.deepEqual(
    containedPipelineImageRect({ left: 10, top: 20, width: 400, height: 400 }, 400, 200),
    { left: 10, top: 120, width: 400, height: 200 },
  );
  assert.deepEqual(
    containedPipelineImageRect({ left: 10, top: 20, width: 400, height: 200 }, 100, 200),
    { left: 160, top: 20, width: 100, height: 200 },
  );
  assert.deepEqual(
    containedPipelineImageRect({ left: 5, top: 7, width: 0, height: 100 }, 100, 100),
    { left: 5, top: 7, width: 0, height: 0 },
  );
});

test('stable slots retain visualization definitions while current results are pending', () => {
  const definitions = [
    { id: 'display-a', parameters: { label: '原始阶段' } },
    { id: 'display-b', parameters: { label: '增强阶段' } },
  ];
  assert.deepEqual(stablePipelineDisplaySlots(definitions, []), [
    { visualization_id: 'display-a', label: '原始阶段', result: null },
    { visualization_id: 'display-b', label: '增强阶段', result: null },
  ]);
  const result = { visualization_id: 'display-b', label: '旧标签', artifact_id: 'artifact-b' };
  assert.deepEqual(stablePipelineDisplaySlots(definitions, [result]), [
    { visualization_id: 'display-a', label: '原始阶段', result: null },
    { visualization_id: 'display-b', label: '增强阶段', result },
  ]);
});

test('display mode preserves pending overlay but resets when multi-display no longer exists', () => {
  assert.equal(resolvePipelineDisplayMode('overlay', 2, false, false), 'overlay');
  assert.equal(resolvePipelineDisplayMode('overlay', 2, true, false), 'split');
  assert.equal(resolvePipelineDisplayMode('source', 1, false, false), 'source');
  assert.equal(resolvePipelineDisplayMode('overlay', 1, true, true), 'source');
});
