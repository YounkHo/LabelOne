import assert from 'node:assert/strict';
import test from 'node:test';

import { directoryBasename, formatDownloadProgress, mergeAssetCursorPage, summarizeScanItems } from './dataset-stream.ts';
import type { AssetCursorPage, DatasetScanItem } from './contracts.ts';

const item = (asset_id: string, status: DatasetScanItem['status'] = 'valid'): DatasetScanItem => ({ asset_id, match_key: asset_id, display_path: asset_id, annotation_paths: [], annotation_file_exists: false, status, selectable: status === 'valid', issues: [] });

test('cursor pages append without duplicates and reset on revision change', () => {
  const current: AssetCursorPage = { items: [item('a'), item('b')], total: 4, next_cursor: 'one', index_revision: 2 };
  const appended = mergeAssetCursorPage(current, { items: [item('b'), item('c')], total: 4, next_cursor: 'two', index_revision: 2 }, true);
  assert.deepEqual(appended.items.map((entry) => entry.asset_id), ['a', 'b', 'c']);
  const reset = mergeAssetCursorPage(appended, { items: [item('z')], total: 1, next_cursor: null, index_revision: 3 }, true);
  assert.deepEqual(reset.items.map((entry) => entry.asset_id), ['z']);
});

test('incremental scan summary preserves disabled asset classes', () => {
  const summary = summarizeScanItems([item('ok'), item('dup', 'duplicate_match'), item('orphan', 'orphan_annotation'), item('bad', 'corrupt_image')]);
  assert.deepEqual(summary, { valid: 1, duplicate_match: 1, orphan_annotation: 1, corrupt_image: 1, corrupt_annotation: 0, hidden_image_only: 0 });
});

test('download progress is bounded and supports unknown totals', () => {
  assert.deepEqual(formatDownloadProgress(512, null), { percent: null, label: '512 B / 未知大小' });
  assert.deepEqual(formatDownloadProgress(1536, 1024), { percent: 100, label: '1.5 KiB / 1.0 KiB' });
});

test('directory helper derives registration name', () => {
  assert.equal(directoryBasename('/data/project/'), 'project');
  assert.equal(directoryBasename('C:\\data\\project'), 'project');
});
