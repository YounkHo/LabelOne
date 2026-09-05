import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../components/global-settings-page.tsx', import.meta.url), 'utf8');
const dialog = readFileSync(new URL('../components/model-picker-dialog.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('the inference panel opens a modal model picker and auto-loads downloaded selections', () => {
  const panel = page.match(/rightTab === 'inference' && <section className="inference-panel">[\s\S]*?rightTab === 'agent'/)?.[0] ?? '';
  assert.match(panel, /className={`inference-model-browser/);
  assert.match(panel, /className="inference-model-trigger"/);
  assert.match(panel, /aria-haspopup="dialog"/);
  assert.match(panel, /aria-expanded=\{modelPickerOpen\}/);
  assert.match(panel, /modelPickerOpen && <ModelPickerDialog/);
  assert.match(page, /const toggleModelPicker = \(\) =>/);
  assert.match(page, /refreshBackendModels\(\)/);
  assert.match(page, /refreshModelWeights\(selectedModel\.id\)/);
  assert.match(page, /refreshBackendHealth\(\)/);
  assert.match(dialog, /role="dialog" aria-modal="true"/);
  assert.match(dialog, /role="tablist" aria-label="模型类别"/);
  assert.match(dialog, /role="listbox" aria-label="可用模型"/);
  assert.match(dialog, /document\.addEventListener\('keydown', handleKeyDown\)/);
  assert.match(dialog, /event\.key === 'Escape'/);
  assert.match(dialog, /event\.key !== 'Tab'/);
  assert.match(dialog, /useDeferredValue\(query\.trim\(\)\.toLocaleLowerCase\(\)\)/);
  assert.match(dialog, /htmlFor="model-picker-search-input"/);
  assert.match(dialog, /placeholder="搜索名称、任务、运行时或模型 ID"/);
  assert.match(dialog, /`\$\{model\.name\} \$\{model\.id\} \$\{model\.task\} \$\{model\.runtime\}`/);
  assert.match(dialog, /aria-label="清空模型搜索"/);
  assert.match(dialog, /searchRef\.current\?\.focus\(\)/);
  assert.match(dialog, /model\.runtimeState === 'loaded'/);
  assert.match(dialog, /model\.usageCount > 0/);
  assert.doesNotMatch(dialog, /loadedModelId/);
  assert.match(css, /\.model-picker-search:focus-within\{/);
  assert.doesNotMatch(panel, /ariaLabel="筛选模型类别"|ariaLabel="选择推理模型"/);
  assert.doesNotMatch(panel, /第一步|第二步|第三步/);
  assert.match(page, /model\.availability === 'available'[\s\S]*?void loadModelById\(model, operationId\)/);
  assert.match(page, /model\.availability === 'missing_weights'[\s\S]*?请先下载权重/);
  assert.match(panel, /modelActionKind === 'download' \? void downloadSelectedModel\(\) : void loadSelectedModel\(\)/);
  assert.match(page, /modelActionKind === 'download' \? '下载'/);
  assert.match(page, /modelActionKind === 'retry' \? '重试'/);
  assert.doesNotMatch(panel, /管理权重|切换模型|从系统导入|下载远程权重|inference-weight-manager/);
  assert.match(page, /setModelPickerOpen\(false\)/);
  assert.match(panel, /className={`model-action-button/);
  assert.match(panel, /className="model-picker-chevron"/);
  assert.doesNotMatch(panel, /\{modelPickerOpen \? '⌃' : '⌄'\}/);
});

test('model catalog status and explicit usage are service-persisted', () => {
  assert.match(backend, /status_by_model/);
  assert.match(backend, /const recordModelUsage = useCallback/);
  assert.match(backend, /\/models\/\$\{encodeURIComponent\(modelId\)\}\/usage/);
  assert.match(page, /runtimeState: backend\.models\.data\.status_by_model/);
  assert.match(page, /if \(model\.runtimeState === 'loaded'\)/);
  assert.match(page, /recordModelUsage\(model\.id\)/);
});

test('the selected model exposes real layer visualization directly below it', () => {
  const panel = page.match(/rightTab === 'inference' && <section className="inference-panel">[\s\S]*?rightTab === 'agent'/)?.[0] ?? '';
  assert.match(panel, /className="feature-capture-card"/);
  assert.match(panel, />中间层可视化</);
  assert.match(panel, /ariaLabel="选择可视化层"/);
  assert.match(panel, /availableLayers\.map\(\(layer\)/);
  assert.match(panel, /onChange=\{setSelectedLayerId\}/);
  assert.doesNotMatch(panel, /setCaptureFeatures|checked=\{captureFeatures/);
  assert.match(panel, /className="feature-layer-select"/);
});

test('the download action resolves all declared missing weights through persistent jobs', () => {
  assert.match(page, /const downloadSelectedModel = async/);
  assert.match(page, /const missing = weights\?\.filter\(\(weight\) => !weight\.downloaded\)/);
  assert.match(page, /backend\.downloadModelWeights\(selectedModel\.id, missing\.map\(\(weight\) => weight\.url_index\)\)/);
  assert.match(page, /backend\.watchJobEvents\(job\.job_id\)/);
  assert.match(page, /已创建 1 个权重下载任务 · \$\{missing\.length\} 个文件/);
  assert.match(page, /modelDownloadActionPendingRef\.current/);
  assert.match(backend, /const downloadModelWeights = useCallback/);
  assert.match(backend, /body: JSON\.stringify\(\{ url_indices: urlIndices \}\)/);
  assert.doesNotMatch(page, /for \(const weight of missing\)/);
  assert.match(page, /selectedModelDownloadProgressItems\.every/);
  assert.match(page, /selectedModelDownloadFileCount/);
  assert.match(page, /'--model-download-progress': `\$\{selectedModelDownloadProgress\}%`/);
  assert.match(page, /selectedModelDownloadProgress === null \? '下载中…' : `\$\{selectedModelDownloadProgress\}%`/);
  assert.match(css, /\.model-action-button\.downloading::after\{[^}]*width:var\(--model-download-progress\);height:3px/);
  assert.match(css, /\.model-action-button\.downloading\.indeterminate::after\{[^}]*animation:model-download-slide/);
  assert.match(dialog, /className={`model-picker-download/);
  assert.match(page, /autoLoadAfterDownloadRef\.current = selectedModel\.id/);
  assert.match(page, /selectedModel\.availability !== 'available'/);
});

test('download completion state is initialized before the auto-load effect reads it', () => {
  const declaration = page.indexOf('const selectedModelDownloadActive =');
  const effectGuard = page.indexOf("autoLoadAfterDownloadRef.current !== selectedModel.id || selectedModel.availability !== 'available' || selectedModelDownloadActive");
  assert.ok(declaration >= 0);
  assert.ok(effectGuard > declaration);
});

test('product settings and model UI do not expose compatibility-source branding', () => {
  const sources = `${page}\n${settings}\n${backend}`;
  assert.doesNotMatch(sources, /X-AnyLabeling|XAnyLabeling|x-anylabeling|modelSource/);
});
