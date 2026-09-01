import assert from 'node:assert/strict';
import test from 'node:test';

import { clearableCompletedTaskIds, filterBackgroundTaskHistory } from './background-task-history.ts';

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
