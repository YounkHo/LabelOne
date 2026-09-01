import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('current-image realtime is the baseline and all-image precompute is one additive switch', () => {
  assert.match(page, /className="pipeline-background-toggle"/);
  assert.match(page, /后台预计算全部图像/);
  assert.match(page, /关闭时仍实时处理当前图/);
  assert.match(page, /type="checkbox" aria-label="后台预计算全部图像" checked=\{pipelineScope === 'all'\}/);
  assert.match(page, /changePipelineScope\(event\.target\.checked \? 'all' : 'current'\)/);
  assert.doesNotMatch(page, /className="scope-segment"|className="scope-control"|>实时 \+ 全库<\/button>|>实时当前图<\/button>/);
  assert.doesNotMatch(css, /\.scope-control\{|\.scope-segment\{/);
  assert.match(page, /当前图实时 · 后台预计算\$\{pipelineScope === 'all' \? '已开' : '已关'\}/);
  assert.match(css, /\.pipeline-background-toggle\{/);
});
