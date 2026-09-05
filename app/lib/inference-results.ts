import type { InferenceResult, JobRecord } from './contracts.ts';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function inferenceResultFromJobItem(value: unknown): InferenceResult | null {
  if (!isRecord(value) || typeof value.model_id !== 'string' || typeof value.image_path !== 'string') return null;
  if (!Array.isArray(value.annotations) || !Array.isArray(value.classifications) || !Array.isArray(value.artifacts) || !Array.isArray(value.rasters)) return null;
  if (!isRecord(value.timings_ms)) return null;
  return value as InferenceResult;
}

export function latestInferenceJobForModel(jobs: JobRecord[], datasetId: string | undefined, modelId: string): JobRecord | undefined {
  if (!datasetId || !modelId) return undefined;
  return jobs
    .filter((job) => job.kind === 'inference' && job.dataset_id === datasetId && job.request.model_id === modelId)
    .sort((left, right) => right.updated_at.localeCompare(left.updated_at))[0];
}

export function inferenceRasterMatchesSource(
  raster: { width: number; height: number },
  sourceWidth: number | undefined,
  sourceHeight: number | undefined,
): boolean {
  return Boolean(sourceWidth && sourceHeight && raster.width === sourceWidth && raster.height === sourceHeight);
}

export type InferenceRasterDisplayMode = 'overlay' | 'standalone' | 'unsupported';

export function inferenceRasterDisplayMode(
  raster: { role: string; width: number; height: number; metadata?: Record<string, unknown> },
  sourceWidth: number | undefined,
  sourceHeight: number | undefined,
): InferenceRasterDisplayMode {
  if (!sourceWidth || !sourceHeight || sourceWidth <= 0 || sourceHeight <= 0) return 'unsupported';
  const kind = String(raster.metadata?.kind ?? '').toLocaleLowerCase();
  const role = raster.role.toLocaleLowerCase();
  const superResolution = kind === 'super_resolution' || role.includes('super-resolution') || role.includes('super_resolution');
  const scaleX = raster.width / sourceWidth;
  const scaleY = raster.height / sourceHeight;
  if (superResolution) {
    return Number.isFinite(scaleX) && scaleX >= 1 && Math.abs(scaleX - scaleY) <= 1e-6 ? 'standalone' : 'unsupported';
  }
  return inferenceRasterMatchesSource(raster, sourceWidth, sourceHeight) ? 'overlay' : 'unsupported';
}

export function inferenceRasterCanvasScale(
  raster: { role: string; width: number; height: number; metadata?: Record<string, unknown> },
  sourceWidth: number | undefined,
  sourceHeight: number | undefined,
): number | null {
  if (inferenceRasterDisplayMode(raster, sourceWidth, sourceHeight) !== 'standalone' || !sourceWidth) return null;
  return raster.width / sourceWidth;
}

export function inferenceAnnotationsAreSegmentation(task: string, adapter?: string): boolean {
  const identity = `${task} ${adapter ?? ''}`.trim().toLocaleLowerCase();
  return identity.includes('segmentation') || identity.includes('segment_anything') || identity.includes('分割');
}

export type InferencePredictionCategory = { label: string; count: number; maxScore: number };

export function groupInferencePredictionsByCategory(
  predictions: Array<{ label: string; score: number }>,
): InferencePredictionCategory[] {
  const groups = new Map<string, InferencePredictionCategory>();
  for (const prediction of predictions) {
    const label = prediction.label.trim() || '未命名预测';
    const current = groups.get(label) ?? { label, count: 0, maxScore: 0 };
    current.count += 1;
    current.maxScore = Math.max(current.maxScore, Number.isFinite(prediction.score) ? prediction.score : 0);
    groups.set(label, current);
  }
  return [...groups.values()].sort((left, right) => right.count - left.count || right.maxScore - left.maxScore || left.label.localeCompare(right.label, 'zh-Hans-CN'));
}

export function inferencePredictionKey(modelId: string, imagePath: string, prediction: { label: string; score: number; shape_type: string; points: number[][] }): string {
  const text = JSON.stringify({ modelId, imagePath, label: prediction.label.trim(), score: prediction.score, shape_type: prediction.shape_type, points: prediction.points });
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) hash = Math.imul(hash ^ text.charCodeAt(index), 0x01000193);
  return `prediction:${(hash >>> 0).toString(16).padStart(8, '0')}`;
}
