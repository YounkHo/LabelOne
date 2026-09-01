import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('canvas display modes keep the original compact horizontal control', () => {
  const start = page.indexOf('{showPipelineViewControls && <div className="pipeline-view-controls"');
  const switcher = page.slice(start, page.indexOf('\n            {pipelinePreviewDirty', start));
  assert.doesNotMatch(page, /function PipelineViewIcon|pipeline-view-segments|pipeline-view-source/);
  assert.match(switcher, /<div role="radiogroup" aria-label="处理流显示布局"><div className=\{`pipeline-single-choice/);
  assert.match(switcher, /role="radio" aria-checked=\{effectiveVisualizationDisplayMode === 'source'\}/);
  assert.match(switcher, /role="radio" aria-checked=\{effectiveVisualizationDisplayMode === 'split'\}/);
  assert.match(switcher, /role="radio" aria-checked=\{effectiveVisualizationDisplayMode === 'overlay'\}/);
  assert.match(switcher, />分屏<\/button>/);
  assert.match(switcher, />叠加<\/button>/);
  assert.doesNotMatch(switcher, /<small>/);
  assert.match(page, /\{ value: 'source', label: '原图' \}/);
});

test('display switcher only shortens the original source selector', () => {
  assert.match(css, /body \.pipeline-view-controls \.pipeline-single-source-select\{width:112px;min-width:96px;max-width:120px\}/);
  assert.match(css, /body \.pipeline-single-choice\.active\{background:#1b3a32;color:#a4e5d2\}/);
  assert.doesNotMatch(css, /Canvas display switcher: one compact control|body \.pipeline-view-icon|body \.pipeline-view-source/);
});
