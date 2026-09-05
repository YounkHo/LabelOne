import assert from 'node:assert/strict';
import test from 'node:test';

import {
  finalPipelineVisualizationId,
  insertPipelineNode,
  insertPipelineNodeAtGap,
  normalizeVisualizationTaps,
  pipelineInsertionGaps,
  pipelineLinearItems,
  pipelineSignature,
  removePipelineNode,
  serializePipelineNodes,
  visualizationOverlayCompatibility,
  type PipelineGraphNode,
  type PipelineVisualizationNode,
} from './pipeline-flow.ts';

const source: PipelineGraphNode = { id: 'source', kind: 'source', enabled: true, parameters: {}, operator_version: '1' };
const resize: PipelineGraphNode = { id: 'resize-1', kind: 'resize', enabled: true, parameters: { width: 100 }, operator_version: '1' };
const visualization = (id: string, tap: string): PipelineVisualizationNode => ({
  id,
  kind: 'visualize',
  enabled: true,
  parameters: { label: id },
  operator_version: '1',
  tap_after_node_id: tap,
});

test('main-chain insertion keeps source fixed and allows repeated operators at any connection', () => {
  const first = insertPipelineNode([source], resize, 1);
  const second = insertPipelineNode(first, { ...resize, id: 'resize-2' }, 1);
  assert.deepEqual(second.map((node) => node.id), ['source', 'resize-2', 'resize-1']);
  assert.deepEqual(removePipelineNode(second, 'source'), second);
  assert.deepEqual(removePipelineNode(second, 'resize-2').map((node) => node.id), ['source', 'resize-1']);
});

test('visualizations serialize as one tap per source stage instead of a visualization chain', () => {
  const nodes = [source, resize];
  const serialized = serializePipelineNodes(nodes, [visualization('display-a', 'resize-1'), visualization('display-b', 'resize-1')]);
  assert.deepEqual(serialized.map((node) => node.id), ['source', 'resize-1', 'display-a']);
});

test('linear flow fixes source at the top and a visualization at the bottom', () => {
  const items = pipelineLinearItems(
    [source, resize],
    [visualization('display-final', 'resize-1'), visualization('display-source', 'source')],
  );
  assert.deepEqual(items.map((item) => item.id), ['source', 'display-source', 'resize-1', 'display-final']);
  assert.equal(items[0].kind, 'source');
  assert.equal(items.at(-1)?.kind, 'visualize');
  assert.equal(items.some((item, index) => item.kind === 'visualize' && items[index + 1]?.kind === 'visualize'), false);
  assert.equal(finalPipelineVisualizationId(
    [source, resize],
    [visualization('display-final', 'resize-1'), visualization('display-source', 'source')],
  ), 'display-final');
});

test('final display selection follows normalized topology and falls back to source only without a display', () => {
  assert.equal(finalPipelineVisualizationId(
    [source, resize],
    [visualization('orphaned-display', 'missing'), visualization('source-display', 'source')],
  ), 'orphaned-display');
  assert.equal(finalPipelineVisualizationId([source], []), 'source');
});

test('every adjacent pair owns one gap and only main-to-main gaps may insert a visualization', () => {
  const finalDisplay = visualization('display-final', 'resize-1');
  const mainGap = pipelineInsertionGaps([source, resize], [finalDisplay])[0];
  assert.equal(mainGap.key, 'source->resize-1');
  assert.equal(mainGap.transformInsertionIndex, 1);
  assert.equal(mainGap.visualizationTapAfterNodeId, 'source');

  const sourceDisplay = visualization('display-source', 'source');
  const gaps = pipelineInsertionGaps([source, resize], [sourceDisplay, finalDisplay]);
  assert.deepEqual(gaps.map((gap) => gap.key), ['source->display-source', 'display-source->resize-1', 'resize-1->display-final']);
  assert.equal(gaps[0].retargetVisualizationId, 'display-source');
  assert.equal(gaps[0].transformInsertionIndex, 1);
  assert.equal(gaps[1].retargetVisualizationId, undefined);
  assert.equal(gaps[1].transformInsertionIndex, 1);
  assert.equal(gaps.every((gap) => gap.visualizationTapAfterNodeId === undefined), true);
});

test('inserting a transform immediately above a visualization reconnects that visualization to the new node', () => {
  const display = visualization('display-final', 'source');
  const gap = pipelineInsertionGaps([source], [display])[0];
  const result = insertPipelineNodeAtGap([source], [display], resize, gap);
  assert.deepEqual(result.nodes.map((node) => node.id), ['source', 'resize-1']);
  assert.equal(result.visualizations[0].tap_after_node_id, 'resize-1');
  assert.deepEqual(pipelineLinearItems(result.nodes, result.visualizations).map((item) => item.id), ['source', 'resize-1', 'display-final']);
});

test('visualization taps stay bounded and one visualization always follows the final stage', () => {
  const nodes = [source, resize];
  const normalized = normalizeVisualizationTaps(nodes, [
    visualization('display-a', 'source'),
    visualization('display-b', 'missing'),
    visualization('display-c', 'source'),
    visualization('display-d', 'source'),
    visualization('display-e', 'source'),
  ]);
  assert.equal(normalized.length, 2);
  assert.equal(normalized.some((item) => item.tap_after_node_id === 'resize-1'), true);
});

test('normalization preserves a valid final display before repairing invalid taps', () => {
  const normalized = normalizeVisualizationTaps([source, resize], [
    visualization('orphaned-display', 'missing'),
    visualization('final-display', 'resize-1'),
  ]);
  assert.deepEqual(normalized.map((item) => item.id), ['final-display']);
});

test('normalization deduplicates before applying the four-display limit and moves the latest remaining display to the final stage', () => {
  const a = { ...resize, id: 'a' };
  const b = { ...resize, id: 'b' };
  const c = { ...resize, id: 'c' };
  const d = { ...resize, id: 'd' };
  const normalized = normalizeVisualizationTaps([source, a, b, c, d], [
    visualization('source-a', 'source'),
    visualization('source-b', 'source'),
    visualization('source-c', 'source'),
    visualization('a-display', 'a'),
    visualization('b-display', 'b'),
    visualization('c-display', 'c'),
    visualization('d-display', 'd'),
  ]);
  assert.deepEqual(normalized.map((item) => item.tap_after_node_id), ['source', 'a', 'b', 'd']);

  const moved = normalizeVisualizationTaps([source, a, b], [
    visualization('source-display', 'source'),
    visualization('a-display', 'a'),
  ]);
  assert.deepEqual(moved.map((item) => [item.id, item.tap_after_node_id]), [
    ['source-display', 'source'],
    ['a-display', 'b'],
  ]);
});

test('pipeline signature is stable but changes for parameters, enabled state, order and visualization taps', () => {
  const nodes = [source, resize];
  const displays = [visualization('display-a', 'resize-1'), visualization('display-b', 'source')];
  const signature = pipelineSignature(nodes, displays);
  assert.equal(pipelineSignature(structuredClone(nodes), structuredClone(displays)), signature);
  assert.notEqual(pipelineSignature([{ ...source }, { ...resize, parameters: { width: 200 } }], displays), signature);
  assert.notEqual(pipelineSignature([{ ...source }, { ...resize, enabled: false }], displays), signature);
  assert.notEqual(pipelineSignature([source, { ...resize, id: 'resize-2' }, resize], displays), signature);
  assert.notEqual(pipelineSignature(nodes, [displays[0], { ...displays[1], tap_after_node_id: 'resize-1' }]), signature);
  assert.equal(pipelineSignature(nodes, [...displays].reverse()), signature);
});

test('overlay requires at least two results with identical valid dimensions', () => {
  assert.equal(visualizationOverlayCompatibility([{ width: 10, height: 10 }]).allowed, false);
  assert.equal(visualizationOverlayCompatibility([{ width: 10, height: 10 }, { width: 10, height: 10 }]).allowed, true);
  assert.match(visualizationOverlayCompatibility([{ width: 10, height: 10 }, { width: 12, height: 10 }]).reason, /尺寸不一致/);
  assert.match(visualizationOverlayCompatibility([{ width: 10, height: 10 }, { width: 10, height: 10, overlay_compatible: false }]).reason, /频域、向量、Token/);
  assert.equal(visualizationOverlayCompatibility([{ width: 10, height: 10, coordinate_space_id: 'same' }, { width: 10, height: 10, coordinate_space_id: 'same' }]).allowed, true);
  assert.match(visualizationOverlayCompatibility([{ width: 10, height: 10, coordinate_space_id: 'source' }, { width: 10, height: 10, coordinate_space_id: 'flipped' }]).reason, /坐标空间不同/);
});
