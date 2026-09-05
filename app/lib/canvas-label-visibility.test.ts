import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('low zoom labels stop participating in layout and rendering', () => {
  assert.match(page, /canvasLabelOpacity\(view\.scale, selectedShapeIndex === index\) <= 0 \? \[\]/);
  assert.match(page, /!showingPipelineImage && canvasLabelOpacity\(view\.scale\) > 0 \? visiblePredictionCanvasEntries\.map/);
  assert.match(page, /const labelOpacity = canvasLabelOpacity\(view\.scale, selected\)/);
  assert.match(page, /labelLayout && labelOpacity > 0/);
  assert.match(page, /style=\{\{ \.\.\.categoryStyle, opacity: labelOpacity \}\}/);
});
