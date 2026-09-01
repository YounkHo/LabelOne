import type { BatchJobRequest } from './contracts.ts';

export type PipelineOutputConfig = NonNullable<BatchJobRequest['output_policy']>;

export type RemoteInferenceConsentContext = {
  action: 'current' | 'batch';
  model_id: string;
  dataset_id: string | null;
  asset_id: string | null;
};

export function requiresRemoteInferenceConfirmation(adapter: string | undefined): boolean {
  return adapter === 'trusted_remote_http';
}

export function remoteInferenceConsentMatches(pending: RemoteInferenceConsentContext, current: RemoteInferenceConsentContext): boolean {
  return pending.action === current.action && pending.model_id === current.model_id && pending.dataset_id === current.dataset_id && pending.asset_id === current.asset_id;
}

export function isAbsoluteOutputPath(value: string): boolean {
  return /^(?:\/|[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+)/.test(value.trim());
}

export function buildPipelineOutputPolicy(config: PipelineOutputConfig, hasTile: boolean): PipelineOutputConfig {
  const mode = hasTile ? 'derived_dataset' : config.mode;
  if (!['png', 'webp', 'jpeg'].includes(config.image_format)) throw new Error('输出格式必须是 png、webp 或 jpeg');
  if (!['reuse', 'error'].includes(config.conflict)) throw new Error('冲突策略必须是 reuse 或 error');
  if (mode === 'derived_dataset') {
    const outputRoot = config.output_root?.trim() ?? '';
    if (!isAbsoluteOutputPath(outputRoot)) throw new Error('派生数据集需要绝对 output_root');
    return { ...config, mode, output_root: outputRoot };
  }
  return { mode, image_format: config.image_format, conflict: config.conflict };
}
