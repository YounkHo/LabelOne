import assert from 'node:assert/strict';
import test from 'node:test';

import { annotationIndexesForCategory, normalizeAnnotationCategory, renameAnnotationCategory, setAnnotationIndexesVisible } from './annotation-category-filter.ts';

test('category filtering preserves original shape indexes', () => {
  assert.deepEqual(annotationIndexesForCategory(['a', 'b', 'a', 'c'], 'a'), [0, 2]);
  assert.deepEqual(annotationIndexesForCategory(['a', 'b', 'a', 'c'], ''), [0, 1, 2, 3]);
  assert.deepEqual(annotationIndexesForCategory([' a ', 'b', ''], 'a'), [0]);
  assert.deepEqual(annotationIndexesForCategory([' a ', 'b', ''], '未命名'), [2]);
  assert.equal(normalizeAnnotationCategory('   '), '未命名');
});

test('bulk visibility only changes indexes in the filtered category', () => {
  const hidden = new Set([1, 2]);
  assert.deepEqual([...setAnnotationIndexesVisible(hidden, [0, 2], false)], [1, 2, 0]);
  assert.deepEqual([...setAnnotationIndexesVisible(hidden, [0, 2], true)], [1]);
});

test('category rename updates every matching shape once and preserves indexes', () => {
  const shapes = [{ label: ' scratch ', id: 1 }, { label: 'particle', id: 2 }, { label: 'scratch', id: 3 }];
  const renamed = renameAnnotationCategory(shapes, 'scratch', ' defect ');
  assert.deepEqual(renamed.indexes, [0, 2]);
  assert.deepEqual(renamed.shapes.map((shape) => shape.label), ['defect', 'particle', 'defect']);
  assert.equal(renamed.shapes[1], shapes[1]);
  assert.equal(shapes[0].label, ' scratch ');
});

test('category rename normalizes unicode and naturally merges into an existing label', () => {
  const decomposed = 'e\u0301';
  const renamed = renameAnnotationCategory([{ label: decomposed }, { label: '目标' }], 'é', '目标');
  assert.deepEqual(renamed.indexes, [0]);
  assert.deepEqual(renamed.shapes.map((shape) => shape.label), ['目标', '目标']);
});
