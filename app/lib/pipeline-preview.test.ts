import assert from 'node:assert/strict';
import test from 'node:test';

import { formatPipelineTiming, neighboringAssetIds, PIPELINE_IMAGE_RETRY_LIMIT, pipelineArtifactDisplayUrl, pipelineImageRetryExhausted, pipelinePrecomputeKey, pipelinePreviewCacheKey, pipelinePreviewResultFromJobItem, pipelineRequestNodesEqual, storeCachedPipelinePreview, takeCachedPipelinePreview, pipelineValidationKey } from './pipeline-preview.ts';

test('pipeline artifact display retries use a fresh URL and stop after a bounded limit', () => {
  assert.equal(pipelineArtifactDisplayUrl('/api/pipeline-artifacts/abc', 4, 1), '/api/pipeline-artifacts/abc?display=4-1');
  assert.equal(pipelineArtifactDisplayUrl('/api/pipeline-artifacts/abc?x=1', 4, 2), '/api/pipeline-artifacts/abc?x=1&display=4-2');
  assert.equal(pipelineImageRetryExhausted(PIPELINE_IMAGE_RETRY_LIMIT), false);
  assert.equal(pipelineImageRetryExhausted(PIPELINE_IMAGE_RETRY_LIMIT + 1), true);
});

test('neighbor ordering alternates left and right by distance', () => {
  assert.deepEqual(neighboringAssetIds(['a', 'b', 'c', 'd', 'e', 'f'], 'd', 4), ['c', 'e', 'b', 'f']);
  assert.deepEqual(neighboringAssetIds(['a', 'b', 'c'], 'a', 4), ['b', 'c']);
  assert.deepEqual(neighboringAssetIds(['a'], 'missing', 4), []);
});

test('preview cache key excludes request priority and keeps identity fields explicit', () => {
  assert.equal(pipelinePreviewCacheKey('data/set', 'asset:1', 'pipeline:abc'), 'data%2Fset:asset%3A1:pipeline:abc');
  assert.equal(pipelineValidationKey('pipeline:abc', 640, 480), 'pipeline:abc:640x480');
  assert.equal(pipelinePrecomputeKey('data/set', 'pipeline:abc'), 'pipeline-precompute:data%2Fset:pipeline:abc');
});

test('preview cache restores synchronously, touches LRU order, and evicts the oldest entry', () => {
  const cache = new Map<string, { artifact: string }>();
  storeCachedPipelinePreview(cache, 'a', { artifact: 'a' }, 2);
  storeCachedPipelinePreview(cache, 'b', { artifact: 'b' }, 2);

  assert.deepEqual(takeCachedPipelinePreview(cache, 'a'), { artifact: 'a' });
  assert.deepEqual([...cache.keys()], ['b', 'a']);

  storeCachedPipelinePreview(cache, 'c', { artifact: 'c' }, 2);
  assert.deepEqual([...cache.keys()], ['a', 'c']);
  assert.equal(takeCachedPipelinePreview(cache, 'missing'), undefined);
});

test('only full preview-shaped job results enter the processed-image cache', () => {
  const preview = { dataset_id: 'd', asset_id: 'a', artifact_id: 'artifact', width: 10, height: 20, media_type: 'image/webp', annotation_document: { shapes: [] }, operator_timings_ms: {} };
  assert.equal(pipelinePreviewResultFromJobItem(preview), preview);
  assert.equal(pipelinePreviewResultFromJobItem({ output_root: '/derived' }), null);
  assert.equal(pipelinePreviewResultFromJobItem(undefined), null);
});

test('persisted precompute nodes compare canonically after reload', () => {
  const left = [{ id: 'a', kind: 'resize', enabled: true, parameters: { width: 2, height: 3 } }];
  const right = [{ kind: 'resize', parameters: { height: 3, width: 2 }, enabled: true, id: 'a' }];
  assert.equal(pipelineRequestNodesEqual(left, right), true);
  assert.equal(pipelineRequestNodesEqual(left, [{ ...right[0], id: 'b' }]), false);
});

test('timing formatting stays compact and includes real sample count', () => {
  assert.equal(formatPipelineTiming(0.321), '平均 0.32 ms');
  assert.equal(formatPipelineTiming(4.26, 8), '平均 4.3 ms · 8 次');
  assert.equal(formatPipelineTiming(18.8, 1), '平均 19 ms');
  assert.equal(formatPipelineTiming(undefined), null);
});
