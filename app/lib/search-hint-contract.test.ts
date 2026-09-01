import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('search hint is explicitly toggled instead of opening on input focus', () => {
  assert.doesNotMatch(page, /onFocus=\{\(\) => setSearchHelp\(true\)\}/);
  assert.match(page, /aria-expanded=\{searchHelp\} aria-controls="search-hint-panel"/);
  assert.match(page, /id="search-hint-panel"[^>]+role="region"/);
  assert.match(page, /event\.key === 'Escape'\) setSearchHelp\(false\)/);
  assert.ok((page.match(/setSearchHelp\(false\)/g) ?? []).length >= 5);
});

test('search mode control stays inside its grid column and leaves input hint padding', () => {
  assert.match(css, /\.search-mode\{width:calc\(100% - 5px\)/);
  assert.match(css, /\.smart-search input\{min-width:0;padding:0 6px/);
});

test('regex shortcut is a reversible pressed-state toggle', () => {
  assert.match(page, /aria-pressed=\{searchMode === 'regex'\}/);
  assert.match(page, /setSearchMode\(\(mode\) => mode === 'regex' \? 'smart' : 'regex'\)/);
  assert.match(page, /title=\{searchMode === 'regex' \? '退出正则搜索' : '启用正则搜索'\}/);
});
