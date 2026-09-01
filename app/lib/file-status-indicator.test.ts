import assert from 'node:assert/strict';
import test from 'node:test';

import type { JobItem, JobRecord } from './contracts.ts';
import { resolveFileStatusIndicator, selectFileProgressJob } from './file-status-indicator.ts';

const item = (state: JobItem['state'], progress?: JobItem['progress'], error?: string): JobItem => ({
  asset_id: 'asset',
  position: 0,
  state,
  attempts: 1,
  progress,
  error,
});

const job = (
  job_id: string,
  state: JobRecord['state'],
  dataset_id = 'dataset',
  kind: 'pipeline' | 'inference' = 'pipeline',
): JobRecord => ({
  job_id,
  kind,
  dataset_id,
  state,
  desired_state: 'run',
  generation: 0,
  request: { kind, dataset_id },
  total: 1,
  completed: state === 'succeeded' ? 1 : 0,
  failed: 0,
  canceled: 0,
  created_at: '2026-08-30T00:00:00Z',
  updated_at: '2026-08-30T00:00:00Z',
  items: [],
});

test('static file state uses a green check only when JSON exists', () => {
  assert.deepEqual(resolveFileStatusIndicator(true, 3), {
    kind: 'check',
    label: '已有 JSON 标注 · 3 个对象',
    progress: null,
  });
  assert.equal(resolveFileStatusIndicator(false, 0).kind, 'empty');
});

test('pipeline progress reports real completed steps and percentage', () => {
  assert.deepEqual(resolveFileStatusIndicator(true, 3, 'pipeline', item('running', {
    kind: 'pipeline',
    progress: 0.5,
    phase: 'operator',
    completed_steps: 2,
    total_steps: 4,
  })), {
    kind: 'progress',
    label: '处理流 · 正在执行算子 · 2/4 步 · 50%',
    progress: 0.5,
  });
});

test('running inference stays indeterminate and terminal states stay explicit', () => {
  assert.equal(resolveFileStatusIndicator(false, 0, 'inference', item('running')).kind, 'running');
  assert.equal(resolveFileStatusIndicator(false, 0, 'pipeline', item('queued')).progress, 0);
  assert.deepEqual(resolveFileStatusIndicator(false, 0, 'pipeline', item('succeeded')), {
    kind: 'check',
    label: '处理流 · 处理完成 · 100%',
    progress: 1,
  });
  assert.deepEqual(resolveFileStatusIndicator(false, 0, 'pipeline', item('failed', undefined, 'decode')), {
    kind: 'failed',
    label: '处理流 · 处理失败：decode',
    progress: null,
  });
  assert.equal(resolveFileStatusIndicator(false, 0, 'pipeline', item('canceled')).kind, 'empty');
  assert.deepEqual(resolveFileStatusIndicator(true, 3, 'pipeline', item('canceled')), {
    kind: 'check',
    label: '已有 JSON 标注 · 3 个对象',
    progress: null,
  });
});

test('file progress keeps the latest terminal job after active processing ends', () => {
  const completed = job('completed', 'succeeded');
  const older = job('older', 'failed');
  assert.equal(selectFileProgressJob([completed, older], 'dataset', 'completed')?.job_id, 'completed');
  const active = job('active', 'running');
  assert.equal(selectFileProgressJob([completed, active], 'dataset', null)?.job_id, 'active');
  assert.equal(selectFileProgressJob([job('paused', 'paused')], 'dataset', null), undefined);
  assert.equal(selectFileProgressJob([job('interrupted', 'interrupted')], 'dataset', null), undefined);
  assert.equal(selectFileProgressJob([job('other', 'running', 'other')], 'dataset', null), undefined);
});

test('a newest canceled job clears file-row task state instead of persisting or falling back to older history', () => {
  const olderCompleted = { ...job('older-completed', 'succeeded'), updated_at: '2026-08-30T01:00:00Z' };
  const newestCanceled = { ...job('newest-canceled', 'canceled'), updated_at: '2026-08-30T02:00:00Z' };
  assert.equal(selectFileProgressJob(
    [olderCompleted, newestCanceled],
    'dataset',
    'newest-canceled',
  ), undefined);

  const active = { ...job('active', 'running'), updated_at: '2026-08-30T00:30:00Z' };
  assert.equal(selectFileProgressJob(
    [olderCompleted, newestCanceled, active],
    'dataset',
    null,
  )?.job_id, 'active');
});

test('disabling the pipeline immediately drops pipeline snapshots but keeps inference status', () => {
  const failedPipeline = job('pipeline-failed', 'failed');
  assert.equal(selectFileProgressJob(
    [failedPipeline],
    'dataset',
    'pipeline-failed',
    { pipelineEnabled: false },
  ), undefined);

  const activeInference = job('inference-active', 'running', 'dataset', 'inference');
  assert.equal(selectFileProgressJob(
    [failedPipeline, activeInference],
    'dataset',
    'pipeline-failed',
    { pipelineEnabled: false },
  )?.job_id, 'inference-active');
});

test('editing the pipeline drops terminal snapshots from the previous definition', () => {
  const previousNodes = [{ id: 'source', kind: 'source', enabled: true, parameters: {} }, { id: 'resize-1', kind: 'resize', enabled: true, parameters: { width: 1_000_001 } }];
  const currentNodes = [{ id: 'source', kind: 'source', enabled: true, parameters: {} }, { id: 'resize-1', kind: 'resize', enabled: true, parameters: { width: 1024 } }];
  const failedPipeline = {
    ...job('pipeline-failed', 'failed'),
    request: { kind: 'pipeline' as const, dataset_id: 'dataset', pipeline_nodes: previousNodes },
  };

  assert.equal(selectFileProgressJob(
    [failedPipeline],
    'dataset',
    'pipeline-failed',
    { pipelineEnabled: true, pipelineNodes: currentNodes },
  ), undefined);
  assert.equal(selectFileProgressJob(
    [failedPipeline],
    'dataset',
    'pipeline-failed',
    { pipelineEnabled: true, pipelineNodes: previousNodes },
  )?.job_id, 'pipeline-failed');
});
