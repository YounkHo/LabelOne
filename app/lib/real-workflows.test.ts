import assert from 'node:assert/strict';
import test from 'node:test';

import { buildPipelineOutputPolicy, isAbsoluteOutputPath, remoteInferenceConsentMatches, requiresRemoteInferenceConfirmation } from './real-workflows.ts';

test('pipeline output policy requires an absolute derived root and tile forces derived mode', () => {
  assert.equal(isAbsoluteOutputPath('/data/derived'), true);
  assert.equal(isAbsoluteOutputPath('C:\\data\\derived'), true);
  assert.throws(() => buildPipelineOutputPolicy({ mode: 'derived_dataset', output_root: 'relative', image_format: 'png', conflict: 'reuse' }, false), /绝对/);
  assert.deepEqual(buildPipelineOutputPolicy({ mode: 'preview', output_root: '/data/tiles', image_format: 'webp', conflict: 'error' }, true), { mode: 'derived_dataset', output_root: '/data/tiles', image_format: 'webp', conflict: 'error' });
  assert.equal('output_root' in buildPipelineOutputPolicy({ mode: 'preview', image_format: 'png', conflict: 'reuse' }, false), false);
});

test('only trusted remote adapters require one-shot confirmation and context must stay exact', () => {
  assert.equal(requiresRemoteInferenceConfirmation('trusted_remote_http'), true);
  assert.equal(requiresRemoteInferenceConfirmation('onnx_raw'), false);
  const pending = { action: 'current' as const, model_id: 'remote', dataset_id: 'dataset', asset_id: 'asset-a' };
  assert.equal(remoteInferenceConsentMatches(pending, { ...pending }), true);
  assert.equal(remoteInferenceConsentMatches(pending, { ...pending, asset_id: 'asset-b' }), false);
  assert.equal(remoteInferenceConsentMatches(pending, { ...pending, action: 'batch', asset_id: null }), false);
});
