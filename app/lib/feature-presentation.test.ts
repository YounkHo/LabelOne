import assert from 'node:assert/strict';
import test from 'node:test';

import type { FeatureLayer } from './contracts';
import { defaultFeatureProjection, featurePreviewDescription, featureProjectionOptions, featureTensorKind } from './feature-presentation.ts';

function layer(shape: FeatureLayer['shape'], axes: string[]): FeatureLayer {
  return { id: 'layer', group: 'test', name: 'layer', shape, axes, spatial: shape.length === 4, captureable: true };
}

test('feature tensor kinds distinguish maps, tokens, vectors and matrices', () => {
  assert.equal(featureTensorKind(layer([1, 32, 16, 16], ['N', 'C', 'H', 'W'])), 'spatial-map');
  assert.equal(featureTensorKind(layer([1, 197, 768], ['N', 'T', 'C'])), 'token-sequence');
  assert.equal(featureTensorKind(layer([1, 1_000], ['N', 'C'])), 'vector');
  assert.equal(featureTensorKind(layer([1_000], ['C'])), 'vector');
  assert.equal(featureTensorKind(layer([12, 64], ['T', 'C'])), 'matrix');
});

test('feature controls only expose transformations meaningful for the tensor kind', () => {
  const vector = layer([1, 1_000], ['N', 'C']);
  const spatial = layer([1, 32, 16, 16], ['N', 'C', 'H', 'W']);
  const tokens = layer([1, 197, 768], ['N', 'T', 'C']);

  assert.deepEqual(featureProjectionOptions(vector), ['None']);
  assert.equal(defaultFeatureProjection(vector), 'None');
  assert.equal(defaultFeatureProjection(tokens), 'Mean');
  assert.ok(!featureProjectionOptions(tokens).includes('Token Grid'));
  assert.ok(featureProjectionOptions(layer([1, 196, 768], ['N', 'T', 'C'])).includes('Token Grid'));
  assert.ok(featureProjectionOptions(spatial).includes('Single Channel'));
  assert.match(featurePreviewDescription('vector'), /折线预览，不使用空间插值/);
});
