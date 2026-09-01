import assert from 'node:assert/strict';
import test from 'node:test';

import type { JobRecord } from './contracts.ts';
import { groupInferencePredictionsByCategory, inferenceAnnotationsAreSegmentation, inferencePredictionKey, inferenceRasterMatchesSource, inferenceResultFromJobItem, latestInferenceJobForModel } from './inference-results.ts';

const result = {
  model_id: 'detector',
  image_path: '/data/current.png',
  annotations: [{ label: 'part', score: .9, shape_type: 'rectangle', points: [[1, 2], [3, 4]] }],
  classifications: [{ label: 'ok', score: .8, rank: 1 }],
  artifacts: [],
  rasters: [{ id: 'mask', role: 'mask', path: '/tmp/mask.png', media_type: 'image/png', width: 10, height: 10, size_bytes: 12, metadata: {} }],
  timings_ms: { total: 12 },
};

test('batch job item result restores the same inference result contract', () => {
  assert.deepEqual(inferenceResultFromJobItem(result), result);
  assert.equal(inferenceResultFromJobItem({ artifact_id: 'pipeline-output' }), null);
  assert.equal(inferenceResultFromJobItem({ ...result, rasters: undefined }), null);
});

test('latest matching inference job is scoped to dataset and model', () => {
  const job = (jobId: string, datasetId: string, modelId: string, updatedAt: string): JobRecord => ({
    job_id: jobId,
    kind: 'inference',
    dataset_id: datasetId,
    state: 'succeeded',
    desired_state: 'run',
    generation: 1,
    request: { kind: 'inference', dataset_id: datasetId, model_id: modelId },
    total: 1,
    completed: 1,
    failed: 0,
    canceled: 0,
    created_at: updatedAt,
    updated_at: updatedAt,
    items: [],
  });
  const jobs = [job('old', 'dataset', 'model', '2026-01-01'), job('new', 'dataset', 'model', '2026-01-02'), job('other', 'dataset', 'other', '2026-01-03')];
  assert.equal(latestInferenceJobForModel(jobs, 'dataset', 'model')?.job_id, 'new');
  assert.equal(latestInferenceJobForModel(jobs, 'missing', 'model'), undefined);
});

test('raster overlay requires exact source dimensions', () => {
  assert.equal(inferenceRasterMatchesSource({ width: 640, height: 480 }, 640, 480), true);
  assert.equal(inferenceRasterMatchesSource({ width: 320, height: 240 }, 640, 480), false);
  assert.equal(inferenceRasterMatchesSource({ width: 640, height: 480 }, undefined, 480), false);
});

test('segmentation annotation models are distinct from detection and OCR models', () => {
  assert.equal(inferenceAnnotationsAreSegmentation('segmentation', 'yolo_segmentation_onnx'), true);
  assert.equal(inferenceAnnotationsAreSegmentation('interactive_segmentation', 'segment_anything_onnx'), true);
  assert.equal(inferenceAnnotationsAreSegmentation('实例分割'), true);
  assert.equal(inferenceAnnotationsAreSegmentation('detection', 'yolo_detection_onnx'), false);
  assert.equal(inferenceAnnotationsAreSegmentation('ocr', 'paddle_ocr'), false);
});

test('detection predictions group by normalized label with bounded confidence summaries', () => {
  assert.deepEqual(groupInferencePredictionsByCategory([
    { label: ' defect ', score: .7 },
    { label: 'defect', score: .93 },
    { label: 'ok', score: .8 },
    { label: '', score: Number.NaN },
  ]), [
    { label: 'defect', count: 2, maxScore: .93 },
    { label: 'ok', count: 1, maxScore: .8 },
    { label: '未命名预测', count: 1, maxScore: 0 },
  ]);
});

test('prediction identity is stable for promotion and changes with geometry', () => {
  const prediction = { label: 'part', score: .9, shape_type: 'rectangle', points: [[1, 2], [3, 4]] };
  assert.equal(inferencePredictionKey('model', '/image.png', prediction), inferencePredictionKey('model', '/image.png', structuredClone(prediction)));
  assert.notEqual(inferencePredictionKey('model', '/image.png', prediction), inferencePredictionKey('model', '/image.png', { ...prediction, points: [[1, 2], [5, 6]] }));
});
