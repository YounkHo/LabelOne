import assert from 'node:assert/strict';
import test from 'node:test';

import { annotationCategoryColors, DEFAULT_ANNOTATION_CATEGORY_STROKES, normalizeAnnotationCategoryColor } from './annotation-category-colors.ts';

function relativeLuminance(color: string): number {
  const channels = [1, 3, 5].map((offset) => Number.parseInt(color.slice(offset, offset + 2), 16) / 255);
  const [red, green, blue] = channels.map((channel) => channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(first: string, second: string): number {
  const lighter = Math.max(relativeLuminance(first), relativeLuminance(second));
  const darker = Math.min(relativeLuminance(first), relativeLuminance(second));
  return (lighter + 0.05) / (darker + 0.05);
}

test('default palette is a calm, unique twelve-color set with strong dark-theme contrast', () => {
  assert.deepEqual(DEFAULT_ANNOTATION_CATEGORY_STROKES, [
    '#6fc2e8', '#e6b566', '#a995e8', '#75c98e', '#df8fb3', '#7aa9ed',
    '#e28d61', '#5dbcb1', '#e47d77', '#a8c85f', '#879bd8', '#cf91d1',
  ]);
  assert.equal(new Set(DEFAULT_ANNOTATION_CATEGORY_STROKES).size, 12);
  for (const stroke of DEFAULT_ANNOTATION_CATEGORY_STROKES) {
    assert.match(stroke, /^#[0-9a-f]{6}$/);
    assert.ok(contrastRatio(stroke, '#0d151f') >= 4.5);
  }
});

test('the same annotation category always receives the same color', () => {
  assert.deepEqual(annotationCategoryColors('scratch'), annotationCategoryColors('scratch'));
  assert.deepEqual(annotationCategoryColors('  scratch  '), annotationCategoryColors('scratch'));
  assert.deepEqual(annotationCategoryColors(''), annotationCategoryColors('   '));
});

test('common distinct categories receive different categorical colors', () => {
  assert.notEqual(annotationCategoryColors('scratch').stroke, annotationCategoryColors('particle').stroke);
  assert.notEqual(annotationCategoryColors('cat-a').stroke, annotationCategoryColors('cat-b').stroke);
});

test('category color does not depend on shape geometry', () => {
  const rectangleCategory = annotationCategoryColors('defect');
  const rotationCategory = annotationCategoryColors('defect');
  assert.deepEqual(rectangleCategory, rotationCategory);
});

test('a valid custom category color drives stroke, fill, and label background', () => {
  const colors = annotationCategoryColors('defect', '#12ABef');
  assert.equal(colors.stroke, '#12abef');
  assert.equal(colors.fill, '#12abef30');
  assert.match(colors.labelBackground, /^#[0-9a-f]{6}$/);
  assert.notEqual(colors.labelBackground, colors.stroke);
});

test('a valid user color overrides the default palette independently of the label hash', () => {
  const first = annotationCategoryColors('scratch', '#12abef');
  const second = annotationCategoryColors('particle', '#12abef');
  assert.deepEqual(first, second);
});

test('default labels use translucent fills and readable label backgrounds', () => {
  for (let index = 0; index < 60; index += 1) {
    const colors = annotationCategoryColors(`category-${index}`);
    assert.equal(colors.fill, `${colors.stroke}30`);
    assert.ok(contrastRatio('#f4f8fb', colors.labelBackground) >= 4.5);
  }
});

test('invalid custom colors are rejected without changing the stable fallback', () => {
  assert.equal(normalizeAnnotationCategoryColor('#123456'), '#123456');
  assert.equal(normalizeAnnotationCategoryColor('#ABCDEF'), '#abcdef');
  assert.equal(normalizeAnnotationCategoryColor('red'), null);
  assert.equal(normalizeAnnotationCategoryColor('#123'), null);
  assert.deepEqual(annotationCategoryColors('defect', 'red'), annotationCategoryColors('defect'));
});
