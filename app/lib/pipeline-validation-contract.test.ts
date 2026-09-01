import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('definition validation runs online without requiring a dataset or selected asset', () => {
  const effect = page.match(/useEffect\(\(\) => \{\n\s*if \(!datasetWorkspaceReady \|\| backend\.mode !== 'online' \|\| !pipelineEnabled\) return;[\s\S]*?requestPipelineValidation\(\)[\s\S]*?\}, \[backend\.mode, datasetWorkspaceReady, pipelineEnabled, requestPipelineValidation\]\);/)?.[0] ?? '';
  assert.ok(effect);
  assert.doesNotMatch(effect, /dataset\.id|currentFile/);
  assert.match(page, /const datasetWorkspaceReady = !dataset\.id \|\| datasetWorkspaceHydratedId === dataset\.id/);
  assert.match(page, /nodes: pipelineRequestNodes/);
  assert.match(page, /pipelineValidationWidth !== undefined && pipelineValidationHeight !== undefined/);
});

test('automatic validation is represented only by a bottom-right status indicator', () => {
  assert.match(page, /className={`pipeline-validation-indicator \$\{pipelineValidationIndicatorState\}`}/);
  assert.match(page, /data-tooltip=\{pipelineValidationIndicatorText\}/);
  assert.match(page, /aria-label={`处理流校验：\$\{pipelineValidationIndicatorText\}`}/);
  assert.doesNotMatch(page, /重新校验处理流|pipeline-validation-retry|pipeline-canvas-toolbar|run-button compact/);
});

test('validation status explains pipeline reasons without depending on annotation saves', () => {
  for (const reason of ['服务离线', '流程关闭', '校验中', '定义已通过 · 未打开数据集', '定义已通过 · 未选择图像', '已通过 · 实时当前图', '已通过 · 当前图实时 + 全库后台预计算']) {
    assert.equal(page.includes(reason), true);
  }
  assert.equal(page.includes('定义已通过 · 等待保存当前标注后预计算全库'), false);
});
