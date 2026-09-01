import type { JobItem, JobRecord } from './contracts.ts';

const activeJobStates = new Set(['queued', 'running', 'pausing', 'canceling']);
const terminalJobStates = new Set(['succeeded', 'succeeded_with_errors', 'failed', 'canceled']);

export type FileProgressJobOptions = {
  pipelineEnabled?: boolean;
  pipelineNodes?: NonNullable<JobRecord['request']['pipeline_nodes']>;
};

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, child]) => [key, canonical(child)]));
}

function pipelineNodesMatch(job: JobRecord, currentNodes: FileProgressJobOptions['pipelineNodes']): boolean {
  if (currentNodes === undefined) return true;
  const jobNodes = job.request.pipeline_nodes;
  return jobNodes !== undefined && JSON.stringify(canonical(jobNodes)) === JSON.stringify(canonical(currentNodes));
}

export type FileStatusIndicatorView = {
  kind: 'empty' | 'check' | 'queued' | 'running' | 'progress' | 'failed' | 'canceled';
  label: string;
  progress: number | null;
};

const jobLabels: Record<'pipeline' | 'inference', string> = {
  pipeline: '处理流',
  inference: '模型推理',
};

const phaseLabels: Record<string, string> = {
  starting: '准备中',
  loaded: '图像已载入',
  operator: '正在执行算子',
  writing: '正在写入结果',
  completed: '处理完成',
};

function staticFileIndicator(annotationFileExists: boolean, annotationCount: number): FileStatusIndicatorView {
  return annotationFileExists
    ? { kind: 'check', label: `已有 JSON 标注 · ${annotationCount} 个对象`, progress: null }
    : { kind: 'empty', label: '', progress: null };
}

export function selectFileProgressJob(
  jobs: JobRecord[],
  datasetId: string | undefined,
  preferredJobId: string | null,
  options: FileProgressJobOptions = {},
): JobRecord | undefined {
  if (!datasetId) return undefined;
  const relevant = jobs.filter((job) => (
    job.dataset_id === datasetId
    && (job.kind === 'inference' || (
      job.kind === 'pipeline'
      && options.pipelineEnabled !== false
      && pipelineNodesMatch(job, options.pipelineNodes)
    ))
  )).sort((left, right) => {
    const leftTime = Date.parse(left.updated_at);
    const rightTime = Date.parse(right.updated_at);
    return (Number.isFinite(rightTime) ? rightTime : 0) - (Number.isFinite(leftTime) ? leftTime : 0);
  });
  return relevant.find((job) => job.job_id === preferredJobId && activeJobStates.has(job.state))
    ?? relevant.find((job) => activeJobStates.has(job.state))
    ?? (relevant[0] && terminalJobStates.has(relevant[0].state) && relevant[0].state !== 'canceled'
      ? relevant[0]
      : undefined);
}

export function resolveFileStatusIndicator(
  annotationFileExists: boolean,
  annotationCount: number,
  jobKind?: JobRecord['kind'],
  item?: JobItem,
): FileStatusIndicatorView {
  if (!item || (jobKind !== 'pipeline' && jobKind !== 'inference')) {
    return staticFileIndicator(annotationFileExists, annotationCount);
  }
  const jobLabel = jobLabels[jobKind];
  if (item.state === 'queued') return { kind: 'queued', label: `${jobLabel} · 等待处理 · 0%`, progress: 0 };
  if (item.state === 'succeeded') return { kind: 'check', label: `${jobLabel} · 处理完成 · 100%`, progress: 1 };
  if (item.state === 'failed') return { kind: 'failed', label: `${jobLabel} · 处理失败${item.error ? `：${item.error}` : ''}`, progress: null };
  if (item.state === 'canceled') return staticFileIndicator(annotationFileExists, annotationCount);
  const rawProgress = item.progress?.progress;
  if (typeof rawProgress !== 'number' || !Number.isFinite(rawProgress)) {
    return { kind: 'running', label: `${jobLabel} · 处理中 · 当前任务未提供单图百分比`, progress: null };
  }
  const progress = Math.max(0, Math.min(1, rawProgress));
  const phase = typeof item.progress?.phase === 'string' ? phaseLabels[item.progress.phase] ?? item.progress.phase : '处理中';
  const steps = typeof item.progress?.completed_steps === 'number' && typeof item.progress?.total_steps === 'number'
    ? ` · ${item.progress.completed_steps}/${item.progress.total_steps} 步`
    : '';
  return {
    kind: 'progress',
    label: `${jobLabel} · ${phase}${steps} · ${Math.round(progress * 100)}%`,
    progress,
  };
}
