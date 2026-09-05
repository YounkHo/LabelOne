import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('canvas selection resolves geometry centrally instead of trusting SVG paint order', () => {
  assert.match(page, /annotationHitCandidates\(displayedShapes, \[point\.x, point\.y\], 7 \* imageUnitsPerScreenPixel\(\), hiddenShapeIndexes\)/);
  assert.match(page, /selectAnnotationHitIndex\(candidates, currentIndex, event\.detail > 1\)/);
  assert.match(page, /startShapeMove\(hitIndex, event\)/);
  assert.doesNotMatch(page, /selectedPreviewShapeIndex|setSelectedPreviewShapeIndex/);
  assert.doesNotMatch(page, /onPointerDown: \(event: React\.PointerEvent<SVGElement>\) => startShapeMove\(index, event\)/);
});

test('root SVG owns move completion and protects overlap cycling from canvas zoom', () => {
  assert.match(page, /if \(shapeDragRef\.current\) \{\s+moveShape\(event\);\s+return;/);
  assert.match(page, /if \(shapeDragRef\.current\) \{\s+endShapeMove\(event\);\s+return;/);
  assert.match(page, /canvasAnnotationHitCandidates\(event\)\.length > 0[\s\S]*event\.stopPropagation\(\)/);
  assert.match(page, /onDoubleClick=\{canvasAnnotationEditable \? handleCanvasDoubleClick : undefined\}/);
  assert.match(page, /连续点击可切换重叠或相互包含的框/);
});
