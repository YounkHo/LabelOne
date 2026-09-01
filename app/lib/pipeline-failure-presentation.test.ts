import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');

test('pipeline execution failures are visible on both canvas and editor', () => {
  assert.match(page, /backend\.pipeline\.phase === 'error' \? backend\.pipeline\.error\?\.message \?\? '处理流执行失败'/);
  assert.match(page, /pipelineFailureNeedsModel[\s\S]*model\|layer\|runtime\|onnx\|weight\|load\|模型\|中间层\|加载/i);
  assert.match(page, /模型尚未加载，无法生成中间层结果。请前往推理页加载模型后重试。/);
  assert.match(page, /className="pipeline-canvas-failure" role="alert" onClick=\{\(\) => openRightTab\('pipeline'\)\}/);
  assert.match(page, /className="pipeline-editor-failure" role="alert"/);
  assert.match(page, /前往推理加载模型/);
  assert.match(css, /\.pipeline-canvas-failure\{position:absolute;z-index:18;left:12px;top:58px/);
  assert.match(css, /\.pipeline-editor-failure\{display:grid;/);
});

test('pipeline failure state clears on retry and successful preview', () => {
  assert.match(backend, /setPipeline\(\(old\) => \(\{ \.\.\.old, phase: 'loading', error: undefined \}\)\)/);
  assert.match(backend, /setPipeline\(\{ phase: 'ready', data: response \}\)/);
  assert.match(backend, /phase: 'error',[\s\S]*error: apiError\(error\)/);
  assert.match(page, /pipelineEnabled && backend\.pipeline\.phase === 'error'/);
});
