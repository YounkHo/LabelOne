import assert from 'node:assert/strict';
import test from 'node:test';

import { buildAnnotationLabelChoices, normalizeAnnotationLabel, positionFloatingLabelMenu } from './annotation-labels.ts';

test('normalizes valid labels and rejects blank or oversized labels', () => {
  assert.equal(normalizeAnnotationLabel('  scratch  '), 'scratch');
  assert.equal(normalizeAnnotationLabel('   '), null);
  assert.equal(normalizeAnnotationLabel('x'.repeat(129)), null);
});

test('known label choices are deduplicated, frequency ordered and preferred first', () => {
  assert.deepEqual(buildAnnotationLabelChoices([' scratch ', 'particle', '', 'scratch', 'chip-edge'], 'particle'), [
    'particle',
    'scratch',
    'chip-edge',
  ]);
});

test('known label choices stay bounded and do not invent placeholder categories', () => {
  assert.deepEqual(buildAnnotationLabelChoices([], ''), []);
  assert.deepEqual(buildAnnotationLabelChoices(['a', 'b', 'c'], '', 2), ['a', 'b']);
});

test('floating label menu follows the pointer and flips at viewport edges', () => {
  assert.deepEqual(positionFloatingLabelMenu({ x: 100, y: 100 }, { width: 1200, height: 800 }), {
    x: 110,
    y: 110,
    transformOrigin: 'left top',
  });
  assert.deepEqual(positionFloatingLabelMenu({ x: 1180, y: 780 }, { width: 1200, height: 800 }), {
    x: 850,
    y: 410,
    transformOrigin: 'right bottom',
  });
});

test('floating label menu stays inside very small viewports', () => {
  assert.deepEqual(positionFloatingLabelMenu({ x: 250, y: 160 }, { width: 280, height: 220 }), {
    x: 8,
    y: 8,
    transformOrigin: 'left top',
  });
  assert.deepEqual(positionFloatingLabelMenu({ x: Number.NaN, y: Number.POSITIVE_INFINITY }, { width: 800, height: 600 }), {
    x: 18,
    y: 18,
    transformOrigin: 'left top',
  });
});
