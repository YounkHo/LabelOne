import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_ANNOTATION_AUTO_SAVE, shouldWriteAnnotationFile } from './annotation-save-policy.ts';

test('annotation JSON auto-save is off by default', () => {
  assert.equal(DEFAULT_ANNOTATION_AUTO_SAVE, false);
});

test('manual navigation save always writes the annotation file', () => {
  assert.equal(shouldWriteAnnotationFile(true, false), true);
});

test('background persistence writes JSON only when auto-save is enabled', () => {
  assert.equal(shouldWriteAnnotationFile(false, false), false);
  assert.equal(shouldWriteAnnotationFile(false, true), true);
});
