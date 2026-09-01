import assert from 'node:assert/strict';
import test from 'node:test';

import {
  defaultShortcutMap,
  displayShortcut,
  findShortcutConflict,
  isForbiddenShortcut,
  resolveShortcutAction,
  resolvedShortcutMap,
  sanitizeShortcutOverrides,
  shortcutFromKeyboardEvent,
} from './keyboard-shortcuts.ts';

const keyboardEvent = (key: string, modifiers: Partial<KeyboardEvent> = {}) => ({
  key,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  isComposing: false,
  ...modifiers,
}) as KeyboardEvent;

test('default shortcuts cover editing, navigation, canvas and every annotation tool', () => {
  const shortcuts = defaultShortcutMap();
  assert.equal(shortcuts['edit.save'], 'Mod+S');
  assert.equal(shortcuts['edit.changeCategory'], 'F2');
  assert.equal(shortcuts['navigation.previous'], 'A');
  assert.equal(shortcuts['canvas.fit'], '0');
  assert.equal(shortcuts['tool.rotation'], 'O');
  assert.equal(Object.keys(shortcuts).length, 19);
});

test('keyboard events require exact modifiers and keep shifted punctuation usable', () => {
  assert.equal(shortcutFromKeyboardEvent(keyboardEvent('p')), 'P');
  assert.equal(shortcutFromKeyboardEvent(keyboardEvent('p', { metaKey: true })), 'Mod+P');
  assert.equal(shortcutFromKeyboardEvent(keyboardEvent('Z', { metaKey: true, shiftKey: true })), 'Mod+Shift+Z');
  assert.equal(shortcutFromKeyboardEvent(keyboardEvent('+', { shiftKey: true })), '+');
  assert.equal(shortcutFromKeyboardEvent(keyboardEvent('Meta', { metaKey: true })), null);
});

test('resolution respects scope and fixed arrow alternatives', () => {
  const shortcuts = defaultShortcutMap();
  assert.equal(resolveShortcutAction(keyboardEvent('s', { metaKey: true }), shortcuts, 'app'), 'edit.save');
  assert.equal(resolveShortcutAction(keyboardEvent('F2'), shortcuts, 'app'), 'edit.changeCategory');
  assert.equal(resolveShortcutAction(keyboardEvent('a'), shortcuts, 'app'), null);
  assert.equal(resolveShortcutAction(keyboardEvent('ArrowLeft'), shortcuts, 'canvas'), 'navigation.previous');
  assert.equal(resolveShortcutAction(keyboardEvent('r'), shortcuts, 'canvas'), 'tool.rect');
  assert.equal(resolveShortcutAction(keyboardEvent('r', { metaKey: true }), shortcuts, 'canvas'), null);
});

test('overrides are sanitized, resolved and checked for conflicts', () => {
  const sanitized = sanitizeShortcutOverrides({ 'tool.rect': 'Q', unknown: 'X', 'tool.line': 'Escape' });
  assert.deepEqual(sanitized, { 'tool.rect': 'Q' });
  const shortcuts = resolvedShortcutMap(sanitized);
  assert.equal(shortcuts['tool.rect'], 'Q');
  assert.equal(shortcuts['tool.line'], 'L');
  assert.equal(findShortcutConflict(shortcuts, 'tool.circle', 'Q')?.id, 'tool.rect');
  assert.equal(isForbiddenShortcut('Mod+R'), true);
});

test('shortcut display uses platform-friendly glyphs', () => {
  assert.equal(displayShortcut('Mod+Shift+Z', true), '⌘+⇧+Z');
  assert.equal(displayShortcut('ArrowLeft', true), '←');
  assert.equal(displayShortcut('Mod+S', false), 'Ctrl+S');
});
