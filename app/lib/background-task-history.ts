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
  return jobs.filter((job) => BACKGROUND_TASK_CLEARABLE_STATES.has(job.state)).map((job) => job.job_id);
}
