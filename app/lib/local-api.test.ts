import assert from 'node:assert/strict';
import test from 'node:test';

import { localApiErrorPayload } from './local-api.ts';

test('unwraps FastAPI HTTPException details instead of hiding the real conflict', () => {
  assert.deepEqual(localApiErrorPayload({ detail: {
    code: 'dataset_has_active_jobs',
    message: '项目仍有关联的未完成后台任务，请先取消任务后再移除',
    details: { job_ids: ['job-1'] },
  } }, 409), {
    code: 'dataset_has_active_jobs',
    message: '项目仍有关联的未完成后台任务，请先取消任务后再移除',
    details: { job_ids: ['job-1'] },
  });
});

test('keeps a safe fallback for malformed non-JSON API failures', () => {
  assert.deepEqual(localApiErrorPayload(null, 503), {
    code: 'http_503',
    message: 'Local API request failed (503)',
    details: undefined,
  });
});
