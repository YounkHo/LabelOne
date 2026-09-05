export const BACKGROUND_TASK_HISTORY_HOURS = [1, 24, 168] as const;
export type BackgroundTaskHistoryHours = typeof BACKGROUND_TASK_HISTORY_HOURS[number];

export const BACKGROUND_TASK_ACTIVE_STATES = new Set(['queued', 'running', 'pausing', 'canceling']);
export const BACKGROUND_TASK_ATTENTION_STATES = new Set(['failed', 'interrupted', 'succeeded_with_errors']);
export const BACKGROUND_TASK_CLEARABLE_STATES = new Set(['succeeded', 'canceled']);

type BackgroundTaskHistoryItem = {
  job_id: string;
  state: string;
  updated_at: string;
};

type BackgroundTaskLogicalItem = BackgroundTaskHistoryItem & {
  kind?: string;
  created_at?: string;
  request?: { model_id?: string; weight_url_indices?: number[] };
  total?: number;
  completed?: number;
  failed?: number;
  canceled?: number;
  logical_job_ids?: string[];
};

function modelDownloadBurstMatches(left: BackgroundTaskLogicalItem, right: BackgroundTaskLogicalItem): boolean {
  if (left.kind !== 'model_download' || right.kind !== 'model_download' || !left.request?.model_id || left.request.model_id !== right.request?.model_id) return false;
  const leftCreated = Date.parse(left.created_at ?? left.updated_at);
  const rightCreated = Date.parse(right.created_at ?? right.updated_at);
  return Number.isFinite(leftCreated) && Number.isFinite(rightCreated) && Math.abs(leftCreated - rightCreated) <= 2000;
}

function backgroundTaskPriority(job: BackgroundTaskHistoryItem): number {
  if (BACKGROUND_TASK_ACTIVE_STATES.has(job.state)) return 0;
  if (BACKGROUND_TASK_ATTENTION_STATES.has(job.state)) return 1;
  return 2;
}

export function coalesceBackgroundTaskHistory<T extends BackgroundTaskLogicalItem>(jobs: T[], preferredJobId?: string | null): Array<T & { logical_job_ids: string[] }> {
  const result: Array<T & { logical_job_ids: string[] }> = [];
  for (const job of jobs) {
    const existingIndex = result.findIndex((existing) => existing.job_id === job.job_id || modelDownloadBurstMatches(existing, job));
    if (existingIndex < 0) {
      result.push({ ...job, logical_job_ids: [job.job_id] });
      continue;
    }
    const existing = result[existingIndex];
    const sameJob = existing.logical_job_ids?.includes(job.job_id) ?? existing.job_id === job.job_id;
    const jobPriority = backgroundTaskPriority(job);
    const existingPriority = backgroundTaskPriority(existing);
    const replace = jobPriority < existingPriority
      || jobPriority === existingPriority && job.job_id === preferredJobId && existing.job_id !== preferredJobId
      || jobPriority === existingPriority && existing.job_id !== preferredJobId && Date.parse(job.updated_at) > Date.parse(existing.updated_at);
    if (sameJob) {
      if (replace) result[existingIndex] = { ...job, logical_job_ids: existing.logical_job_ids ?? [job.job_id] };
      continue;
    }
    const representative = replace ? job : existing;
    const weightIndices = [...new Set([...(existing.request?.weight_url_indices ?? []), ...(job.request?.weight_url_indices ?? [])])].sort((left, right) => left - right);
    result[existingIndex] = {
      ...representative,
      request: { ...representative.request, model_id: representative.request?.model_id, weight_url_indices: weightIndices },
      total: (existing.total ?? 0) + (job.total ?? 0),
      completed: (existing.completed ?? 0) + (job.completed ?? 0),
      failed: (existing.failed ?? 0) + (job.failed ?? 0),
      canceled: (existing.canceled ?? 0) + (job.canceled ?? 0),
      created_at: Date.parse(existing.created_at ?? existing.updated_at) <= Date.parse(job.created_at ?? job.updated_at) ? existing.created_at ?? existing.updated_at : job.created_at ?? job.updated_at,
      updated_at: Date.parse(existing.updated_at) >= Date.parse(job.updated_at) ? existing.updated_at : job.updated_at,
      logical_job_ids: [...new Set([...(existing.logical_job_ids ?? [existing.job_id]), job.job_id])],
    } as T & { logical_job_ids: string[] };
  }
  return result;
}

export function filterBackgroundTaskHistory<T extends BackgroundTaskHistoryItem>(jobs: T[], dismissedJobVersions: Readonly<Record<string, string>>, hours: BackgroundTaskHistoryHours, now = Date.now()): T[] {
  const cutoff = now - hours * 60 * 60 * 1000;
  return jobs.filter((job) => {
    if (BACKGROUND_TASK_ACTIVE_STATES.has(job.state)) return true;
    if (dismissedJobVersions[job.job_id] === job.updated_at) return false;
    const updatedAt = Date.parse(job.updated_at);
    return Number.isFinite(updatedAt) && updatedAt >= cutoff;
  });
}

export function clearableCompletedTaskIds<T extends BackgroundTaskHistoryItem>(jobs: T[]): string[] {
  return jobs.filter((job) => BACKGROUND_TASK_CLEARABLE_STATES.has(job.state)).flatMap((job) => 'logical_job_ids' in job && Array.isArray(job.logical_job_ids) ? job.logical_job_ids : [job.job_id]);
}
