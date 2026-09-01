import assert from 'node:assert/strict';
import test from 'node:test';

import { hideAllShapes, remapHiddenShapesAfterDeletion, remapHiddenShapesAfterDeletions, remapSelectedShapeAfterDeletion, remapSelectedShapeAfterDeletions, setShapeVisibility } from './annotation-visibility.ts';

test('individual shape visibility is immutable and reversible', () => {
  const original = new Set([1]);
  const hidden = setShapeVisibility(original, 3, false);
  assert.deepEqual([...original], [1]);
  assert.deepEqual([...hidden], [1, 3]);
  assert.deepEqual([...setShapeVisibility(hidden, 1, true)], [3]);
});

test('hide all creates an index for every shape', () => {
  assert.deepEqual([...hideAllShapes(4)], [0, 1, 2, 3]);
  assert.deepEqual([...hideAllShapes(-1)], []);
});

test('hidden indexes stay attached to the same objects after deletion', () => {
  assert.deepEqual([...remapHiddenShapesAfterDeletion(new Set([0, 2, 4]), 2)], [0, 3]);
  assert.deepEqual([...remapHiddenShapesAfterDeletion(new Set([1, 3]), 0)], [0, 2]);
});

test('selected index follows the same object after another shape is deleted', () => {
  assert.equal(remapSelectedShapeAfterDeletion(null, 1), null);
  assert.equal(remapSelectedShapeAfterDeletion(1, 1), null);
  assert.equal(remapSelectedShapeAfterDeletion(3, 1), 2);
  assert.equal(remapSelectedShapeAfterDeletion(0, 1), 0);
});

test('visibility and selection stay attached after deleting a whole category', () => {
  assert.deepEqual([...remapHiddenShapesAfterDeletions(new Set([0, 2, 4, 5]), [1, 4])], [0, 1, 3]);
  assert.equal(remapSelectedShapeAfterDeletions(5, [1, 4]), 3);
  assert.equal(remapSelectedShapeAfterDeletions(4, [1, 4]), null);
  assert.equal(remapSelectedShapeAfterDeletions(null, [1, 4]), null);
});
