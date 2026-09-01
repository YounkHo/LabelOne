import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('inference result types expose independent controls only when data exists', () => {
  const panel = page.match(/rightTab === 'inference' && <section className="inference-panel">[\s\S]*?rightTab === 'agent'/)?.[0] ?? '';
  assert.doesNotMatch(panel, /Detection|显示检测和轮廓|currentDetectionPredictions\.length/);
  assert.match(page, /const showInferenceResult = [^;]*currentAnnotationsAreSegmentation \|\| currentClassifications\.length > 0 \|\| currentArtifacts\.length > 0 \|\| currentRasters\.length > 0/);
  assert.match(page, /\(currentSegmentationContours\.length > 0 \|\| currentRasters\.length > 0\) && <div className={`prediction-display-control \$\{showMasks/);
  assert.match(page, /currentClassifications\.length > 0 && <div className={`prediction-display-control \$\{showClassifications/);
  assert.match(page, /aria-label="显示分割和像素结果" aria-checked=\{showMasks && !pixelResultDisplayBlockedReason\}/);
  assert.match(page, /aria-label="显示分类 Top-K" aria-checked=\{showClassifications\}/);
  assert.doesNotMatch(page, /画布模型预测|showPred|setShowPred/);
});

test('detection, mask, and classification each use their native canvas presentation', () => {
  assert.match(page, /const visibleCurrentPredictionEntries = currentDetectionPredictions\.flatMap/);
  assert.match(page, /!showingPipelineImage && visiblePredictionCanvasEntries\.length > 0 && <g className=\{currentAnnotationsAreSegmentation \? 'inference-segmentation-contours' : 'inference-detections'\}/);
  assert.match(page, /!showingPipelineImage && canvasLabelOpacity\(view\.scale\) > 0 \? visiblePredictionCanvasEntries\.map/);
  assert.match(page, /const activeRaster = showMasks \? displayableRasters\.find/);
  assert.match(page, /currentRasters\.filter\(\(raster\) => inferenceRasterMatchesSource\(raster, currentFile\?\.width, currentFile\?\.height\)\)/);
  assert.match(page, /disabled=\{!compatible\} title=\{compatible \? `在画布显示 \$\{raster\.role\}` : mismatchReason\}/);
  assert.match(page, /style=\{\{ opacity: rasterOpacity \/ 100 \}\}/);
  assert.match(page, /showClassifications && currentClassifications\.length > 0 && <section className="canvas-classification-overlay"/);
  assert.match(page, /currentClassifications\.slice\(0, 5\)\.map/);
  assert.match(page, /showGT && displayedShapes\.map/);
  assert.match(css, /\.canvas-classification-overlay\{position:absolute;z-index:18;/);
  assert.match(css, /\.inference-segmentation-contours \.real-shape\.prediction\{/);
});

test('succeeded batch inference item restores the current asset result', () => {
  assert.match(page, /latestInferenceJobForModel\(backendJobs, dataset\.id, selectedModel\.id\)/);
  assert.match(page, /lookupJobItems\(currentBatchInferenceJob\.job_id, \[currentFile\.id\]\)/);
  assert.match(page, /currentBatchInferenceItem\?\.state === 'succeeded'/);
  assert.match(page, /inferenceResultFromJobItem\(currentBatchInferenceItem\.result\)/);
  assert.match(page, /const currentInferenceResult = directInferenceMatchesCurrent[\s\S]*batchInferenceMatchesCurrent \? currentBatchInferenceResult : null/);
  assert.match(page, /const currentPredictions = currentInferenceResult\?\.annotations \?\? \[\]/);
  assert.match(page, /currentInferenceResult\?\.timings_ms\.total/);
});

test('inference results use one compact dock and batch scope lives outside the inference panel', () => {
  const panel = page.match(/rightTab === 'inference' && <section className="inference-panel">[\s\S]*?rightTab === 'agent'/)?.[0] ?? '';
  assert.doesNotMatch(panel, /batch-inference-card|批量 AI 推理|创建批量任务/);
  assert.doesNotMatch(panel, /className="result-summary"|className="inference-artifacts"/);
  assert.match(panel, /className="inference-result-strip" role="status"/);
  assert.match(panel, /className="feature-result-compact"/);
  assert.match(panel, /<details key=\{artifact\.id\}>/);
  assert.match(css, /\.inference-results-region\{[^}]*overflow-x:hidden;overflow-y:auto/);
  assert.doesNotMatch(page, /接受模型|一键全部接受|accept-predictions|prepareAcceptedPredictions|hasAcceptedPredictionResult/);
});
