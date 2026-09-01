import assert from 'node:assert/strict';
import test from 'node:test';

import { fileAnnotationFilterLabels, fileAnnotationFilters, matchesFileAnnotationFilter } from './file-annotation-filter.ts';

test('file annotation filter exposes direct JSON choices in menu order', () => {
  assert.deepEqual(fileAnnotationFilters, ['all', 'with_json', 'without_json']);
});

test('file annotation filter uses JSON file existence instead of shape count', () => {
  assert.equal(matchesFileAnnotationFilter('all', false), true);
  assert.equal(matchesFileAnnotationFilter('with_json', true), true);
  assert.equal(matchesFileAnnotationFilter('with_json', false), false);
  assert.equal(matchesFileAnnotationFilter('without_json', false), true);
  assert.equal(matchesFileAnnotationFilter('without_json', true), false);
});

test('file annotation filter has concise labels for the compact control', () => {
  assert.deepEqual(fileAnnotationFilterLabels, {
    all: '全部文件',
    with_json: '有 JSON',
    without_json: '无 JSON',
  });
});
