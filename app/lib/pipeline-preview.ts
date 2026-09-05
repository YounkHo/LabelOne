import type { PipelinePreviewResult } from './contracts';

export const PIPELINE_PREVIEW_CACHE_LIMIT = 32;
export const PIPELINE_IMAGE_RETRY_LIMIT = 2;

export function pipelineArtifactDisplayUrl(url: string, epoch: number, attempt: number): string {
  if (attempt <= 0) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}display=${Math.max(0, Math.trunc(epoch))}-${Math.max(0, Math.trunc(attempt))}`;
}

export function pipelineImageRetryExhausted(attempt: number): boolean {
  return attempt > PIPELINE_IMAGE_RETRY_LIMIT;
}

export function takeCachedPipelinePreview<T>(cache: Map<string, T>, key: string): T | undefined {
  const cached = cache.get(key);
  if (cached === undefined) return undefined;
  cache.delete(key);
  cache.set(key, cached);
  return cached;
}

export function storeCachedPipelinePreview<T>(
  cache: Map<string, T>,
  key: string,
  result: T,
  maximumEntries = PIPELINE_PREVIEW_CACHE_LIMIT,
): void {
  cache.delete(key);
  cache.set(key, result);
  while (cache.size > maximumEntries) cache.delete(cache.keys().next().value!);
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, child]) => [key, canonical(child)]));
}

export function neighboringAssetIds(assetIds: string[], currentAssetId: string, maximum = 4): string[] {
  const currentIndex = assetIds.indexOf(currentAssetId);
  if (currentIndex < 0 || maximum <= 0) return [];
  const result: string[] = [];
  for (let distance = 1; result.length < maximum && (currentIndex - distance >= 0 || currentIndex + distance < assetIds.length); distance += 1) {
    const left = assetIds[currentIndex - distance];
    const right = assetIds[currentIndex + distance];
    if (left !== undefined) result.push(left);
    if (right !== undefined && result.length < maximum) result.push(right);
  }
  return result;
}

export function pipelinePreviewCacheKey(datasetId: string, assetId: string, signature: string): string {
  return `${encodeURIComponent(datasetId)}:${encodeURIComponent(assetId)}:${signature}`;
}

export function pipelineValidationKey(signature: string, width?: number, height?: number): string {
  return `${signature}:${width ?? '?'}x${height ?? '?'}`;
}

export function pipelinePrecomputeKey(datasetId: string, signature: string): string {
  return `pipeline-precompute:${encodeURIComponent(datasetId)}:${signature}`;
}

export function pipelineRequestNodesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

export function pipelineModelFeatureRuntimeKey(
  nodes: Array<{ kind: string; enabled?: boolean; parameters?: Record<string, unknown> }>,
  runtime: { model_id?: string; state?: string } | null | undefined,
  statusByModel: Record<string, { runtime_state?: string }> | null | undefined,
): string {
  const feature = nodes.find((node) => node.kind === 'model_feature' && node.enabled !== false);
  const modelId = typeof feature?.parameters?.model_id === 'string' ? feature.parameters.model_id.trim() : '';
  if (!feature || !modelId) return 'none';
  const state = runtime?.model_id === modelId
    ? runtime.state
    : statusByModel?.[modelId]?.runtime_state;
  return `${modelId}:${state || 'unloaded'}`;
}

export function hasDisplayablePipelinePane(items: Array<{ displayUrl?: string | null }>): boolean {
  return items.some((item) => Boolean(item.displayUrl));
}

export function pipelinePreviewResultFromJobItem(result: Record<string, unknown> | undefined): PipelinePreviewResult | null {
  if (!result
    || typeof result.dataset_id !== 'string'
    || typeof result.asset_id !== 'string'
    || typeof result.artifact_id !== 'string'
    || typeof result.width !== 'number'
    || typeof result.height !== 'number'
    || typeof result.media_type !== 'string'
    || !result.annotation_document
    || typeof result.annotation_document !== 'object'
    || !result.operator_timings_ms
    || typeof result.operator_timings_ms !== 'object') return null;
  return result as PipelinePreviewResult;
}

export function formatPipelineTiming(milliseconds: number | undefined, sampleCount?: number): string | null {
  if (milliseconds === undefined || !Number.isFinite(milliseconds) || milliseconds < 0) return null;
  const timing = milliseconds < 1 ? `${milliseconds.toFixed(2)} ms` : milliseconds < 10 ? `${milliseconds.toFixed(1)} ms` : `${Math.round(milliseconds)} ms`;
  return sampleCount && sampleCount > 1 ? `平均 ${timing} · ${sampleCount} 次` : `平均 ${timing}`;
}
