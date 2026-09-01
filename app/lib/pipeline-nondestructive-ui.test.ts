import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('pipeline preprocessing exposes no destructive output policy or duplicate static contract', () => {
  assert.doesNotMatch(page, /批处理输出|Derived dataset|output_root|文件冲突|pipeline-output-policy|className="flow-contract"/);
  assert.doesNotMatch(page, /pipelineOutputMode|pipelineOutputRoot|pipelineOutputFormat|pipelineOutputConflict|buildPipelineOutputPolicy/);
  assert.doesNotMatch(css, /\.pipeline-output-policy|\.flow-contract|\.tile-output-notice/);
  assert.match(page, /只写应用缓存，不修改原图或标注 JSON/);
  assert.doesNotMatch(page, /应用派生标注|保存派生标注/);
});

test('pipeline output stays cache-only while source annotation controls remain editable', () => {
  assert.doesNotMatch(page, /if \(showingPipelineImage\) return(?: false)?;/);
  assert.doesNotMatch(page, /allowWhileShowingPipelineImage/);
  assert.doesNotMatch(page, /disabled=\{showingPipelineImage \|\| annotationSaving\}/);
  assert.match(page, /className="shape-delete-button"[^>]+onClick=\{\(\) => deleteShapeAtIndex\(index\)\}/);
  assert.match(page, /className="annotation-category-delete"[^>]+onClick=\{\(\) => deleteAnnotationCategory\(category\)\}/);
  assert.doesNotMatch(page, /应用派生标注|保存派生标注/);
});
