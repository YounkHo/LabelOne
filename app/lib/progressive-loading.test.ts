import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');

test('dataset scanning opens a persisted first batch before the full scan succeeds', () => {
  assert.match(page, /session\.state === 'running'[\s\S]*?session\.persisted_items >= 32/);
  assert.match(page, /finishAutoScanRef\.current\([^)]*, true\)/);
  assert.match(page, /finalizeProgressiveScan/);
  assert.match(page, /其余数据继续后台加载/);
});

test('nearby images and annotations are prefetched through bounded caches', () => {
  assert.match(page, /neighboringAssetIds\(listedFileIds, currentFile\?\.id \?\? '', 4\)/);
  assert.match(page, /prefetchAnnotation\(dataset\.id!, assetId\)/);
  assert.match(page, /prioritizeJobItems\(currentPipelinePrecomputeJob\.jobId, preferredPipelineAssetIds\)/);
  assert.match(page, /neighborImageCacheRef\.current\.size > 12/);
  assert.match(backend, /ANNOTATION_CACHE_LIMIT = 24/);
  assert.match(backend, /annotationInflight/);
});
