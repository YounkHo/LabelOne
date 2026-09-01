import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('ONNX layers come from the loaded runtime and require an explicit user selection', () => {
  assert.match(page, /const runtimeLayers = backend\.runtime\.data\?\.model_id === selectedModel\.id[\s\S]*?backend\.runtime\.data\.layers/);
  assert.match(page, /const availableLayers = runtimeLayers\.filter\(\(layer\) => layer\.captureable !== false\)/);
  assert.match(page, /const selectedLayer = availableLayers\.find\(\(layer\) => layer\.id === selectedLayerId\);/);
  assert.doesNotMatch(page, /runtime\.layers\.find\([\s\S]*?\?\? runtime\.layers\[0\]/);
  assert.match(page, /const captureFeatures = Boolean\(selectedLayer\)/);
  assert.match(page, /onChange=\{setSelectedLayerId\}/);
  assert.doesNotMatch(page, /setCaptureFeatures|type="checkbox"[^>]*captureFeatures/);
  assert.match(page, /setFeatureChannel\(0\);[\s\S]*?defaultFeatureProjection\(selectedLayer\)/);
});

test('feature capture uses graph metadata and a real server-rendered preview', () => {
  assert.match(page, /selectedLayer \? `已启用 · \$\{selectedLayer\.name\}`/);
  assert.match(page, /artifactPreviewUrl\(selectedFeatureArtifact\.id\)/);
  assert.match(page, /className=\{`feature-preview-image \$\{selectedFeatureKind\}`\}/);
  assert.match(hook, /`\$\{base\}\/artifacts\/\$\{encodeURIComponent\(artifactId\)\}\/preview`/);
  assert.doesNotMatch(page, /className="heatmap-preview"/);
  assert.doesNotMatch(css, /\.heatmap-preview\{|radial-gradient\(circle at 62% 42%/);
});

test('layer selection and ONNX feature controls live in one visualization card', () => {
  assert.equal(page.match(/Layer Visualization/g)?.length, 1);
  assert.match(page, /className="feature-capture-card"[\s\S]*?中间层可视化[\s\S]*?选择可视化层/);
  assert.doesNotMatch(page, /className=\{`inference-layer-selector/);
  assert.doesNotMatch(page, />ONNX 中间层特征</);
  assert.match(page, /featureTensorKindLabel\(selectedFeatureKind\)/);
  assert.match(page, /ariaLabel="特征值截断"/);
  assert.doesNotMatch(page, /setFeatureScale|setFeatureGain|setFeatureGamma|setInterpolation/);
  assert.doesNotMatch(page, />数值增益<|>Gamma<|>空间 Scale<|ariaLabel="空间插值"/);
});

test('model results and capture requests stay bound to the selected model and layer', () => {
  assert.match(page, /backend\.inference\.data\.model_id === selectedModel\.id/);
  assert.match(page, /const captureLayer = captureFeatures \? runtime\.layers\.find\(\(layer\) => layer\.id === selectedLayerId\) : undefined/);
  assert.match(page, /capture_layers: captureLayer \? \[captureLayer\.id\] : \[\]/);
  assert.match(page, /channel: featureChannel/);
  assert.doesNotMatch(page, /batch-inference-card|批量 AI 推理/);
});
