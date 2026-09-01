import assert from 'node:assert/strict';
import test from 'node:test';

import { clampPipelineNumericValue, pipelineParameterControlKind, pipelineParameterSchemaForContext, pipelineRangeStep } from './pipeline-parameters.ts';

test('parameter control selection follows schema shape', () => {
  assert.equal(pipelineParameterControlKind({ type: 'number', minimum: 0, maximum: 1 }), 'range');
  assert.equal(pipelineParameterControlKind({ type: 'integer', minimum: 0, maximum: 1_000_000 }), 'number');
  assert.equal(pipelineParameterControlKind({ type: 'integer', minimum: 0, maximum: 100, 'x-ui': { control: 'number', role: 'region-x', unit: 'px' } }), 'number');
  assert.equal(pipelineParameterControlKind({ type: 'number', minimum: 0, maximum: 100, 'x-ui': { control: 'slider' } }), 'range');
  assert.equal(pipelineParameterControlKind({ type: 'integer', minimum: 1 }), 'number');
  assert.equal(pipelineParameterControlKind({ type: 'boolean' }), 'checkbox');
  assert.equal(pipelineParameterControlKind({ type: 'string', enum: ['a', 'b'] }), 'enum');
  assert.equal(pipelineParameterControlKind({ type: 'string' }), 'string');
});

test('numeric values clamp to schema bounds and integers round', () => {
  assert.equal(clampPipelineNumericValue(2, { type: 'number', minimum: 0, maximum: 1 }), 1);
  assert.equal(clampPipelineNumericValue(-2, { type: 'number', minimum: 0, maximum: 1 }), 0);
  assert.equal(clampPipelineNumericValue(2.6, { type: 'integer', minimum: 0, maximum: 10 }), 3);
  assert.equal(clampPipelineNumericValue(0, { type: 'integer', minimum: 0.2, maximum: 4.8 }), 1);
  assert.equal(clampPipelineNumericValue(9, { type: 'integer', minimum: 0.2, maximum: 4.8 }), 4);
  assert.equal(clampPipelineNumericValue('bad', { type: 'number', minimum: 2, maximum: 8, default: 4 }), 4);
});

test('range step is integral for integers and bounded for continuous values', () => {
  assert.equal(pipelineRangeStep({ type: 'integer', minimum: 0, maximum: 100 }), 1);
  assert.equal(pipelineRangeStep({ type: 'number', minimum: 0, maximum: 1 }), 0.005);
  assert.equal(pipelineRangeStep({ type: 'number', minimum: 0, maximum: 1, multipleOf: 0.01 }), 0.01);
});

test('crop and resize bounds adapt to the upstream image and paired values', () => {
  const cropSchemas = {
    x: { type: 'integer' as const, minimum: 0, maximum: 1_000_000, 'x-ui': { role: 'region-x' as const } },
    y: { type: 'integer' as const, minimum: 0, maximum: 1_000_000, 'x-ui': { role: 'region-y' as const } },
    width: { type: 'integer' as const, minimum: 1, maximum: 1_000_000, 'x-ui': { role: 'region-width' as const } },
    height: { type: 'integer' as const, minimum: 1, maximum: 1_000_000, 'x-ui': { role: 'region-height' as const } },
  };
  const cropContext = { inputWidth: 640, inputHeight: 480, parameters: { x: 100, y: 20 }, schemas: cropSchemas };
  assert.equal(pipelineParameterSchemaForContext(cropSchemas.x, cropContext).maximum, 639);
  assert.equal(pipelineParameterSchemaForContext(cropSchemas.y, cropContext).maximum, 479);
  assert.equal(pipelineParameterSchemaForContext(cropSchemas.width, cropContext).maximum, 540);
  assert.equal(pipelineParameterSchemaForContext(cropSchemas.height, cropContext).maximum, 460);

  const resizeSchemas = {
    width: { type: 'integer' as const, minimum: 1, maximum: 1_000_000, 'x-ui': { role: 'target-width' as const } },
    height: { type: 'integer' as const, minimum: 1, maximum: 1_000_000, 'x-ui': { role: 'target-height' as const } },
  };
  assert.equal(pipelineParameterSchemaForContext(resizeSchemas.width, { parameters: { height: 10_000 }, schemas: resizeSchemas }).maximum, 6400);
  assert.equal(pipelineParameterSchemaForContext(resizeSchemas.height, { parameters: { width: 8_000 }, schemas: resizeSchemas }).maximum, 8000);
});
