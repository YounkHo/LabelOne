import assert from 'node:assert/strict';
import test from 'node:test';

import { annotationDraftKey, annotationFingerprint, commitAnnotationHistory, createAnnotationHistory, redoAnnotationHistory, undoAnnotationHistory } from './annotation-history.ts';
import { validatePersistedAnnotationDraft } from './annotation-drafts.ts';
import { renameAnnotationCategory } from './annotation-category-filter.ts';

test('history keeps at least 100 committed undo steps', () => {
  let history = createAnnotationHistory({ value: 0 });
  for (let value = 1; value <= 130; value += 1) history = commitAnnotationHistory(history, { value });
  assert.equal(history.past.length, 100);
  for (let index = 0; index < 100; index += 1) history = undoAnnotationHistory(history);
  assert.equal(history.present.value, 30);
});

test('undo, redo and branching commits form coherent transactions', () => {
  let history = createAnnotationHistory({ shapes: [] as number[] });
  history = commitAnnotationHistory(history, { shapes: [1] });
  history = commitAnnotationHistory(history, { shapes: [1, 2] });
  history = undoAnnotationHistory(history);
  assert.deepEqual(history.present.shapes, [1]);
  history = redoAnnotationHistory(history);
  assert.deepEqual(history.present.shapes, [1, 2]);
  history = undoAnnotationHistory(history);
  history = commitAnnotationHistory(history, { shapes: [1, 3] });
  assert.equal(history.future.length, 0);
  assert.deepEqual(redoAnnotationHistory(history), history);
});

test('equivalent documents are not added to history', () => {
  const history = createAnnotationHistory({ z: 1, nested: { b: 2, a: 3 } });
  const next = commitAnnotationHistory(history, { nested: { a: 3, b: 2 }, z: 1 });
  assert.equal(next, history);
  assert.equal(annotationFingerprint(history.present), annotationFingerprint(next.present));
});

test('renaming a whole category is one undoable history transaction', () => {
  const original = { shapes: [{ label: 'a' }, { label: 'b' }, { label: 'a' }] };
  let history = createAnnotationHistory(original);
  const renamed = renameAnnotationCategory(history.present.shapes, 'a', 'b');
  history = commitAnnotationHistory(history, { ...history.present, shapes: renamed.shapes });
  assert.equal(history.past.length, 1);
  assert.deepEqual(history.present.shapes.map((shape) => shape.label), ['b', 'b', 'b']);
  history = undoAnnotationHistory(history);
  assert.deepEqual(history.present, original);
  history = redoAnnotationHistory(history);
  assert.deepEqual(history.present.shapes.map((shape) => shape.label), ['b', 'b', 'b']);
});

test('persisted draft identity and shape structure are validated', () => {
  const record = {
    key: annotationDraftKey('dataset', 'asset'),
    dataset_id: 'dataset',
    asset_id: 'asset',
    base_revision: 'revision-1',
    document: { shapes: [] },
    updated_at: 42,
  };
  assert.deepEqual(validatePersistedAnnotationDraft(record, 'dataset', 'asset'), record);
  assert.equal(validatePersistedAnnotationDraft({ ...record, key: 'wrong' }), null);
  assert.equal(validatePersistedAnnotationDraft({ ...record, document: {} }), null);
  assert.equal(validatePersistedAnnotationDraft({ ...record, document: { shapes: [{ label: 'bad', shape_type: 'point', points: [[Number.NaN, 0]] }] } }), null);
  assert.equal(validatePersistedAnnotationDraft(record, 'other', 'asset'), null);
});
