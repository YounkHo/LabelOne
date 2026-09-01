import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveCanvasCursorMode, resolveResizeCursor, type CanvasCursorTool } from './canvas-cursor.ts';

const enabled = { drawingEnabled: true, temporaryPan: false, panning: false };

test('canvas cursor distinguishes selection, pan idle, pan drag and drawing', () => {
  assert.equal(resolveCanvasCursorMode('select', enabled), 'select');
  assert.equal(resolveCanvasCursorMode('pan', enabled), 'pan');
  assert.equal(resolveCanvasCursorMode('pan', { ...enabled, panning: true }), 'panning');
  assert.equal(resolveCanvasCursorMode('select', { ...enabled, temporaryPan: true }), 'pan');
  assert.equal(resolveCanvasCursorMode('select', { ...enabled, temporaryPan: true, panning: true }), 'panning');
  assert.equal(resolveCanvasCursorMode('rect', { ...enabled, temporaryPan: true }), 'pan');
  assert.equal(resolveCanvasCursorMode('rect', { ...enabled, panning: true }), 'panning');
});

test('every drawing tool resolves to the crosshair cursor mode', () => {
  const drawingTools: CanvasCursorTool[] = ['rect', 'rotation', 'polygon', 'point', 'line', 'circle', 'brush'];
  for (const tool of drawingTools) {
    assert.equal(resolveCanvasCursorMode(tool, enabled), 'draw');
  }
  assert.equal(resolveCanvasCursorMode('rect', { ...enabled, drawingEnabled: false }), 'default');
});

test('resize cursor follows horizontal, vertical and diagonal control-point geometry', () => {
  assert.equal(resolveResizeCursor([[0, 0], [100, 0]], 0), 'ew-resize');
  assert.equal(resolveResizeCursor([[0, 0], [0, 100]], 0), 'ns-resize');
  assert.equal(resolveResizeCursor([[0, 0], [100, 0], [100, 100], [0, 100]], 0), 'nwse-resize');
  assert.equal(resolveResizeCursor([[50, 0], [100, 50], [50, 100], [0, 50]], 0), 'ns-resize');
  assert.equal(resolveResizeCursor([[100, 0], [150, 50], [50, 100], [0, 50]], 0), 'nesw-resize');
});
