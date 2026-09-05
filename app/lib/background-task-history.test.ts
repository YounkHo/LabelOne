import assert from 'node:assert/strict';
import test from 'node:test';

import { clearableCompletedTaskIds, coalesceBackgroundTaskHistory, filterBackgroundTaskHistory } from './background-task-history.ts';

const now = Date.parse('2026-08-31T12:00:00Z');
const task = (job_id: string, state: string, hoursAgo: number) => ({ job_id, state, updated_at: new Date(now - hoursAgo * 60 * 60 * 1000).toISOString() });

test('task history keeps active work, respects the selected time window and hides dismissed terminal rows', () => {
  const jobs = [task('active-old', 'running', 100), task('recent', 'succeeded', 2), task('old', 'succeeded', 30), task('ignored', 'failed', 1)];
  assert.deepEqual(filterBackgroundTaskHistory(jobs, { ignored: jobs[3].updated_at }, 24, now).map((job) => job.job_id), ['active-old', 'recent']);
  assert.deepEqual(filterBackgroundTaskHistory(jobs, {}, 168, now).map((job) => job.job_id), ['active-old', 'recent', 'old', 'ignored']);
  assert.deepEqual(filterBackgroundTaskHistory([{ ...jobs[3], updated_at: new Date(now).toISOString() }], { ignored: jobs[3].updated_at }, 24, now).map((job) => job.job_id), ['ignored']);
});

test('clear completed only dismisses successful or canceled terminal jobs', () => {
  const jobs = [task('done', 'succeeded', 1), task('canceled', 'canceled', 1), task('partial', 'succeeded_with_errors', 1), task('failed', 'failed', 1), task('running', 'running', 1)];
  assert.deepEqual(clearableCompletedTaskIds(jobs), ['done', 'canceled']);
});

test('duplicate snapshots and equivalent active downloads collapse to one visible task', () => {
  const burstTask = (job_id: string, offsetMs: number, weight_url_indices: number[]) => ({
    job_id,
    state: 'running',
    kind: 'model_download',
    created_at: new Date(now + offsetMs).toISOString(),
    updated_at: new Date(now + offsetMs).toISOString(),
    request: { model_id: 'sam', weight_url_indices },
    total: weight_url_indices.length,
    completed: 0,
    failed: 0,
    canceled: 0,
  });
  const downloads = [
    burstTask('download-old', 0, [0]),
    burstTask('other-weight', 10, [1]),
    burstTask('download-new', 17, [0]),
    { ...task('job', 'queued', 0), kind: 'pipeline', request: {} },
    { ...task('job', 'running', 0), kind: 'pipeline', request: {} },
  ];
  const coalesced = coalesceBackgroundTaskHistory(downloads);
  assert.deepEqual(coalesced.map((job) => job.job_id), ['download-new', 'job']);
  assert.deepEqual(coalesced[0].logical_job_ids, ['download-old', 'other-weight', 'download-new']);
  assert.deepEqual(coalesced[0].request.weight_url_indices, [0, 1]);
  assert.equal(coalesced[0].total, 3);
  assert.equal(coalesceBackgroundTaskHistory(downloads, 'download-old')[0].job_id, 'download-old');
});
