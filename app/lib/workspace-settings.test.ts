import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { globalWorkspaceSettings, snapshotPipelineSettings, usablePipelineSettings } from './workspace-settings.ts';

const pipeline = snapshotPipelineSettings({
  enabled: true,
  scope: 'current',
  nodes: [{ id: 'source', kind: 'source', name: '原图像', enabled: true, parameters: {} }],
  visualizations: [{ id: 'visualize-1', kind: 'visualize', name: '显示', enabled: true, parameters: { label: '显示' }, tap_after_node_id: 'source' }],
  displayMode: 'split',
  singleSource: 'source',
  layerState: { 'visualize-1': { visible: true, opacity: 88 } },
});

test('workspace snapshots preserve durable pipeline state without runtime artifacts', () => {
  assert.deepEqual(pipeline.nodes.map(({ id, kind }) => ({ id, kind })), [{ id: 'source', kind: 'source' }]);
  assert.equal(pipeline.visualizations[0].tap_after_node_id, 'source');
  assert.deepEqual(pipeline.layer_state, { 'visualize-1': { visible: true, opacity: 88 } });
  assert.doesNotMatch(JSON.stringify(pipeline), /artifact|timing|completedSignature|jobId/);
});

test('invalid pipeline snapshots fail closed and global defaults clone inference parameters', () => {
  assert.equal(usablePipelineSettings({ ...pipeline, visualizations: [] }), null);
  assert.equal(usablePipelineSettings({ ...pipeline, visualizations: [{ ...pipeline.visualizations[0], tap_after_node_id: 'missing' }] }), null);
  const parameters = { conf: 0.4 };
  const global = globalWorkspaceSettings(pipeline, { model_id: 'detector', provider: 'CPUExecutionProvider', parameters });
  parameters.conf = 0.9;
  assert.deepEqual(global.inference.parameters, { conf: 0.4 });
});

test('page persists dataset resume state and uses dataset over global pipeline defaults', () => {
  const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
  assert.match(page, /getDatasetSettings\(registered\.dataset_id\)/);
  assert.match(page, /getDatasetAsset\(registered\.dataset_id, workspace\.last_asset_id\)/);
  assert.match(page, /usablePipelineSettings\(workspace\.pipeline\)[\s\S]*?usablePipelineSettings\(defaultPipeline\)[\s\S]*?persistedPipelineSettings/);
  assert.match(page, /last_asset_id: currentFileId/);
  assert.match(page, /expected_revision: datasetWorkspaceRevisionRef\.current/);
  assert.match(page, /datasetWorkspaceReady/);
});
