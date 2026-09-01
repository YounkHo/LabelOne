import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('rotation angle controls provide one-degree and tenth-degree adjustments', () => {
  const editor = page.match(/<section className="real-rotation-editor">[\s\S]*?<\/section>/)?.[0] ?? '';
  assert.match(editor, /rotateSelectedShape\(-1\)\}>−1°/);
  assert.match(editor, /rotateSelectedShape\(-0\.1\)\}>−0\.1°/);
  assert.match(editor, /rotateSelectedShape\(0\.1\)\}>＋0\.1°/);
  assert.match(editor, /rotateSelectedShape\(1\)\}>＋1°/);
  assert.doesNotMatch(editor, /rotateSelectedShape\((?:-?15|-?2)\)/);
});
