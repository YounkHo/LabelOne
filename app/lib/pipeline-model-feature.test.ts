import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const popover = readFileSync(new URL('../components/pipeline-insert-popover.tsx', import.meta.url), 'utf8');

test('model feature is a processing node linked to the shared inference configuration', () => {
  const featureConfig = page.slice(page.indexOf('function PipelineFeatureConfiguration'), page.indexOf('function PipelineParameterEditor'));
  assert.match(page, /node\.kind === 'model_feature'/);
  assert.match(page, /function PipelineFeatureConfiguration/);
  assert.match(page, /由推理选项卡统一配置/);
  assert.match(page, /前往推理配置/);
  assert.match(page, /const openInferenceFeatureConfiguration = \(node: FlowNode\)/);
  assert.match(page, /onOpenInference=\{selectedOperator\.kind === 'model_feature'/);
  assert.doesNotMatch(featureConfig, /<CustomSelect|<input|type="range"/);
  assert.doesNotMatch(page, /inferenceFeatureCardRef\.current\?\.scrollIntoView/);
  assert.match(page, /inferenceFeatureCardRef\.current\?\.focus\(\{ preventScroll: true \}\)/);
  assert.match(page, /ref=\{inferenceFeatureCardRef\} className="feature-capture-card" tabIndex=\{-1\}/);
  assert.match(page, /node\.kind !== 'model_feature'\) return node/);
  assert.match(page, /model_id: selectedModel\.id/);
  assert.match(page, /layer_id: selectedLayer\?\.id/);
  assert.match(page, /parameters\.projection = featureTransformParameters\.projection/);
  assert.match(page, /parameters\.clip = featureClip/);
  assert.doesNotMatch(page, /parameters\.(?:gain|gamma|spatial_scale|interpolation) =/);
  assert.doesNotMatch(page, /parameters\.(?:gain|gamma|spatial_scale|interpolation) =/);
  assert.match(css, /\.pipeline-feature-node-controls\.linked\{/);
});

test('a pipeline allows only one model feature and explains blocked insertions visibly', () => {
  assert.match(page, /const pipelineHasModelFeature = nodes\.some\(\(node\) => node\.kind === 'model_feature'\)/);
  assert.match(page, /disabledReason: '当前处理流已有模型中间层；请先删除或配置现有节点'/);
  assert.match(page, /kind === 'model_feature' && pipelineHasModelFeature[\s\S]*?每个处理流最多只能添加一个/);
  assert.match(popover, /item\.disabledReason \?\? \(item\.kind\.startsWith/);
  assert.match(popover, /visualizationDisabled \? visualizationTitle/);
  assert.match(page, /pipelineVisibleError[\s\S]*?pipelineValidationIndicatorState/);
  assert.match(page, /role=\{pipelineValidationIndicatorState === 'invalid' \? 'alert' : 'status'\}/);
  assert.match(css, /\.flow-insert-menu>button\.blocked-option\{/);
  assert.match(css, /\.pipeline-validation-indicator\.invalid\{/);
});

test('adding a model feature reuses the existing display and never creates an extra display', () => {
  const addNode = page.match(/const addNode = \(kind:[\s\S]*?\n  const deleteNode =/)?.[0];
  assert.ok(addNode);
  assert.match(addNode, /const nextVisualizations = kind === 'model_feature'[\s\S]*?label: '中间层特征'/);
  assert.doesNotMatch(addNode, /nextVisualizationId|label: '上游图像'|displayContract/);
  assert.match(addNode, /notify\(`已添加 \$\{operator\.name\}`\)/);
  assert.match(page, /models=\{displayedModelCatalog\}[\s\S]*?featureLayers=\{pipelineFeatureRuntimeLayers\}/);
});

test('model feature validation clears stale layers and model selection persists', () => {
  assert.match(page, /layer_id: selectedLayer\?\.id/);
  assert.doesNotMatch(page, /layer_id: selectedLayer\?\.id \?\? \(modelChanged/);
  assert.match(page, /LAST_SELECTED_MODEL_KEY = 'labelone-last-selected-model-v1'/);
  assert.match(page, /localStorage\.getItem\(LAST_SELECTED_MODEL_KEY\)/);
  assert.match(page, /localStorage\.setItem\(LAST_SELECTED_MODEL_KEY, id\)/);
});

test('split pane transforms normalize against the full canvas instead of the first half-width pane', () => {
  assert.match(page, /pipelinePaneTransform\(view, stageRef\.current\?\.clientWidth \?\? 840, stageRef\.current\?\.clientHeight \?\? 592\)/);
  assert.doesNotMatch(page, /pipelinePaneTransform\(view, imageRef\.current\?\.clientWidth/);
  assert.match(page, /activePreviewPane \? activePreviewPane\.getBoundingClientRect\(\) : stage\.getBoundingClientRect\(\)/);
  assert.match(page, /const pane = target\?\.closest<HTMLElement>\('\[data-pipeline-preview-pane\]'\)/);
});
