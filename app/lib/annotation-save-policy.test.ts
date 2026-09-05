import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_ANNOTATION_AUTO_SAVE,
  normalizeAnnotationAutoSavePreference,
  shouldWriteAnnotationFile,
} from './annotation-save-policy.ts';

test('annotation JSON auto-save is off by default', () => {
  assert.equal(DEFAULT_ANNOTATION_AUTO_SAVE, false);
});

test('global auto-save preference restores only an explicit boolean', () => {
  assert.equal(normalizeAnnotationAutoSavePreference(true), true);
  assert.equal(normalizeAnnotationAutoSavePreference(false), false);
  assert.equal(normalizeAnnotationAutoSavePreference('true'), DEFAULT_ANNOTATION_AUTO_SAVE);
  assert.equal(normalizeAnnotationAutoSavePreference(undefined), DEFAULT_ANNOTATION_AUTO_SAVE);
});

test('manual navigation save always writes the annotation file', () => {
  assert.equal(shouldWriteAnnotationFile(true, false), true);
});

test('background persistence writes JSON only when auto-save is enabled', () => {
  assert.equal(shouldWriteAnnotationFile(false, false), false);
  assert.equal(shouldWriteAnnotationFile(false, true), true);
});
