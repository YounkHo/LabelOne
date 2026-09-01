import assert from 'node:assert/strict';
import test from 'node:test';

import { flipCanvasDelta, flipCanvasPoint, flipCanvasShape, inverseTransformCanvasPoint, resolveCanvasPresentation, resolvePipelineCoordinateTransform, resolvePipelineFlipAxes, shouldSwitchToSelectAfterBlankClick, transformCanvasPoint, transformCanvasShape } from './canvas-interaction.ts';

test('pipeline image is paired with its transformed annotations and dimensions', () => {
  const shapes = [{ label: 'scratch', points: [[1, 2], [3, 4]] }];
  const flipped = [{ label: 'scratch', points: [[1, 298], [3, 296]] }];
  const pipelineInput = {
    sourceImageUrl: '/source.png',
    pipelineImageUrl: '/processed.png',
    pipelineEnabled: true,
    pipelineScope: 'current' as const,
    annotationShapes: shapes,
    pipelineAnnotationShapes: flipped,
    sourceWidth: 800,
    sourceHeight: 600,
    pipelineWidth: 400,
    pipelineHeight: 300,
  };
  const preview = resolveCanvasPresentation(pipelineInput);
  const failed = resolveCanvasPresentation({
    sourceImageUrl: '/source.png',
    pipelineImageUrl: null,
    pipelineEnabled: true,
    pipelineScope: 'current',
    annotationShapes: shapes,
    sourceWidth: 800,
    sourceHeight: 600,
  });
  const disabled = resolveCanvasPresentation({
    sourceImageUrl: '/source.png',
    pipelineImageUrl: '/processed.png',
    pipelineEnabled: false,
    pipelineScope: 'current',
    annotationShapes: shapes,
    sourceWidth: 800,
    sourceHeight: 600,
  });

  assert.equal(preview.imageUrl, '/processed.png');
  const allScope = resolveCanvasPresentation({ ...pipelineInput, pipelineImageUrl: '/batch.png', pipelineScope: 'all' });
  assert.equal(allScope.imageUrl, '/batch.png');
  assert.equal(allScope.shapes, flipped);
  assert.equal(failed.imageUrl, '/source.png');
  assert.equal(disabled.imageUrl, '/source.png');
  assert.equal(preview.shapes, flipped);
  assert.equal(failed.shapes, shapes);
  assert.equal(disabled.shapes, shapes);
  assert.deepEqual([preview.width, preview.height], [400, 300]);
  for (const result of [failed, disabled]) assert.deepEqual([result.width, result.height], [800, 600]);
  for (const incomplete of [
    { ...pipelineInput, pipelineAnnotationShapes: undefined },
    { ...pipelineInput, pipelineWidth: undefined },
    { ...pipelineInput, pipelineHeight: 0 },
    { ...pipelineInput, pipelineHeight: Number.NaN },
  ]) {
    const result = resolveCanvasPresentation(incomplete);
    assert.equal(result.imageUrl, '/source.png');
    assert.equal(result.shapes, shapes);
  }
});

test('flip-only pipelines keep preview and source editing coordinates reversible', () => {
  const vertical = resolvePipelineFlipAxes([
    { id: 'source', kind: 'source' },
    { id: 'flip', kind: 'flip', parameters: { axis: 'vertical' } },
    { id: 'display', kind: 'visualize' },
  ], 'display')!;
  assert.deepEqual(vertical, { horizontal: false, vertical: true });
  assert.deepEqual(flipCanvasPoint([20, 10], 100, 50, vertical), [20, 40]);
  assert.deepEqual(flipCanvasPoint([20, 40], 100, 50, vertical), [20, 10]);
  assert.deepEqual(flipCanvasDelta([4, 7], vertical), [4, -7]);
  assert.deepEqual(flipCanvasShape({ label: 'box', points: [[10, 5], [30, 20]] }, 100, 50, vertical).points, [[10, 45], [30, 30]]);
  assert.deepEqual(resolvePipelineFlipAxes([
    { kind: 'flip', parameters: { axis: 'horizontal' } },
    { kind: 'flip', parameters: { axis: 'horizontal' } },
  ]), { horizontal: false, vertical: false });
  assert.equal(resolvePipelineFlipAxes([{ kind: 'resize' }]), null);
});

test('crop resize flip and rotate compose into one reversible canvas transform', () => {
  const transform = resolvePipelineCoordinateTransform([
    { id: 'crop', kind: 'crop', parameters: { x: 10, y: 5, width: 80, height: 40 } },
    { id: 'resize', kind: 'resize', parameters: { width: 40, height: 20 } },
    { id: 'flip', kind: 'flip', parameters: { axis: 'horizontal' } },
    { id: 'rotate', kind: 'rotate', parameters: { degrees: 90 } },
    { id: 'display', kind: 'visualize' },
  ], 100, 50, 'display')!;
  assert.deepEqual([transform.width, transform.height], [20, 40]);
  assert.equal(transform.topologySafe, false);
  assert.deepEqual(transformCanvasPoint([30, 15], transform), [15, 30]);
  assert.deepEqual(inverseTransformCanvasPoint([15, 30], transform), [30, 15]);
  assert.deepEqual(transformCanvasShape({ points: [[30, 15], [50, 25]] }, transform).points, [[15, 30], [10, 20]]);

  const uniform = resolvePipelineCoordinateTransform([
    { kind: 'resize', parameters: { width: 200, height: 100 } },
    { kind: 'rotate', parameters: { degrees: 270 } },
  ], 100, 50)!;
  assert.equal(uniform.topologySafe, true);
  assert.deepEqual([uniform.width, uniform.height], [100, 200]);
  assert.equal(resolvePipelineCoordinateTransform([{ kind: 'custom.warp' }], 100, 50), null);
});

test('a completed drawing switches to select only on the next blank primary click', () => {
  const base = { armed: true, tool: 'rect', button: 0, spaceDown: false, blankTarget: true };
  assert.equal(shouldSwitchToSelectAfterBlankClick(base), true);
  assert.equal(shouldSwitchToSelectAfterBlankClick({ ...base, armed: false }), false);
  assert.equal(shouldSwitchToSelectAfterBlankClick({ ...base, blankTarget: false }), false);
  assert.equal(shouldSwitchToSelectAfterBlankClick({ ...base, button: 1 }), false);
  assert.equal(shouldSwitchToSelectAfterBlankClick({ ...base, spaceDown: true }), false);
  assert.equal(shouldSwitchToSelectAfterBlankClick({ ...base, tool: 'select' }), false);
});
