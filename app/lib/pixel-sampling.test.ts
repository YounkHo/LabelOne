import assert from 'node:assert/strict';
import test from 'node:test';

import { pixelSampleFromRgba, pixelValue, sourcePixelAtDisplayPoint } from './pixel-sampling.ts';

test('pixel V uses the HSV value channel', () => {
  assert.equal(pixelValue(255, 0, 0), 255);
  assert.equal(pixelValue(0, 182, 18), 182);
  assert.equal(pixelValue(10, 20, 30), 30);
});

test('RGBA samples are clamped and include V', () => {
  assert.deepEqual(pixelSampleFromRgba([300, 40.4, -8, 128.2]), { r: 255, g: 40, b: 0, a: 128, v: 255 });
  assert.deepEqual(pixelSampleFromRgba([10, 20, 30]), { r: 10, g: 20, b: 30, a: 255, v: 30 });
});

test('display coordinates map to bounded source pixels', () => {
  const rect = { left: 100, top: 50, width: 200, height: 100 };
  assert.deepEqual(sourcePixelAtDisplayPoint(100, 50, rect, 1000, 500), { x: 0, y: 0 });
  assert.deepEqual(sourcePixelAtDisplayPoint(200, 100, rect, 1000, 500), { x: 500, y: 250 });
  assert.deepEqual(sourcePixelAtDisplayPoint(300, 150, rect, 1000, 500), { x: 999, y: 499 });
  assert.equal(sourcePixelAtDisplayPoint(99, 100, rect, 1000, 500), null);
});
