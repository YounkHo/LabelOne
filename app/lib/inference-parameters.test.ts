import assert from 'node:assert/strict';
import test from 'node:test';

import { inferenceParameterDefaults, inferenceRequestSignature, normalizeInferenceParameterSchema } from './inference-parameters.ts';

test('model parameter schemas are normalized and defaults stay model-specific', () => {
  const schema = normalizeInferenceParameterSchema({ properties: {
    conf_threshold: { type: 'number', title: '置信度', minimum: 0, maximum: 1, default: 0.25 },
    top_k: { type: 'integer', minimum: 1, maximum: 100, default: 5 },
    output_cutout: { type: 'boolean', default: true },
    mode: { type: 'string', enum: ['a', 'b'], default: 'b' },
    invalid: { type: 'object' },
  } });

  assert.deepEqual(Object.keys(schema), ['conf_threshold', 'top_k', 'output_cutout', 'mode']);
  assert.deepEqual(inferenceParameterDefaults(schema), { conf_threshold: 0.25, top_k: 5, output_cutout: true, mode: 'b' });
});

test('request signatures are stable across object key order', () => {
  assert.equal(inferenceRequestSignature({ b: 2, a: { y: 1, x: 0 } }), inferenceRequestSignature({ a: { x: 0, y: 1 }, b: 2 }));
});
