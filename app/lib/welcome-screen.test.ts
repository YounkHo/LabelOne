import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const welcome = readFileSync(new URL('../components/welcome-screen.tsx', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');
const contracts = readFileSync(new URL('./contracts.ts', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('an empty window shows start, recent projects and settings instead of the annotation workspace', () => {
  assert.match(page, /openedDataset && currentFile \? <section className="workspace">/);
  assert.match(page, /: <WelcomeScreen/);
  assert.match(page, /recentProjects=\{recentProjects\}/);
  assert.match(welcome, /'正在打开系统选择器…' : '打开项目'/);
  assert.match(welcome, /<strong>打开设置<\/strong>/);
  assert.match(welcome, /最近打开/);
  assert.match(welcome, /recentProjects\.map\(\(project\)/);
  assert.match(welcome, /onOpenRecent\(project\)/);
  assert.match(css, /\.welcome-screen\{/);
  assert.match(css, /\.recent-project-list>button:not\(:disabled\):hover/);
});

test('recent projects restore their saved asset and fall back to the first valid asset', () => {
  assert.match(page, /const openRecentProject = async \(registered: RegisteredDataset\)/);
  assert.match(page, /backend\.openRegisteredDataset\(registered\.dataset_id\)/);
  assert.match(page, /workspace\.last_asset_id/);
  assert.match(page, /getDatasetAsset\(registered\.dataset_id, workspace\.last_asset_id\)/);
  assert.match(page, /chosen \?\?= opened\.files\.find\(\(file\) => file\.status === 'valid' && file\.selectable === true\)/);
  assert.match(page, /setOpenedDataset\(opened\)/);
  assert.match(page, /markProjectRecent\(registered\.dataset_id\)/);
  assert.match(page, /labelone-recent-projects-v1/);
  assert.doesNotMatch(page, /datasetRestoreAttempted|preferredDatasetIdRef|preferredAssetIdRef/);
});

test('startup has no built-in datasets, fake annotations, predictions, inference or agent replies', () => {
  const sources = `${page}\n${backend}\n${contracts}`;
  assert.doesNotMatch(sources, /晶圆缺陷集|航拍建筑集|细胞分割集|wafer_zone_|demoAnnotationShapes|demoPredictionShape|modelCatalog|layerProfiles/);
  assert.doesNotMatch(sources, /演示回复|演示 Worker|推理当前图片（演示）|演示数据索引|演示模式/);
  assert.doesNotMatch(contracts, /'demo'/);
  assert.match(backend, /useState<BackendMode>\('probing'\)/);
});
