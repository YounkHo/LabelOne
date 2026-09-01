import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const component = readFileSync(new URL('../components/custom-select.tsx', import.meta.url), 'utf8');

test('application pages no longer render browser-native select controls', () => {
  assert.doesNotMatch(page, /<select|<option/);
  assert.ok((page.match(/<CustomSelect/g) ?? []).length >= 8);
});

test('custom select owns its portal, listbox semantics, and keyboard behavior', () => {
  assert.match(component, /createPortal/);
  assert.match(component, /role="combobox"/);
  assert.match(component, /role="listbox"/);
  assert.match(component, /role="option"/);
  for (const key of ['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter', 'Escape', 'Tab']) assert.match(component, new RegExp(`event\\.key === '${key}'`));
  assert.match(component, /document\.addEventListener\('pointerdown', closeOutside\)/);
  assert.match(component, /window\.addEventListener\('scroll', updatePosition, true\)/);
});
