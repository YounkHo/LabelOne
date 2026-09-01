import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { fullscreenPortalTarget } from './portal-target.ts';

const customSelect = readFileSync(new URL('../components/custom-select.tsx', import.meta.url), 'utf8');
const insertPopover = readFileSync(new URL('../components/pipeline-insert-popover.tsx', import.meta.url), 'utf8');
const tooltip = readFileSync(new URL('../components/global-tooltip.tsx', import.meta.url), 'utf8');

test('portals mount inside the fullscreen element when one exists', () => {
  const body = { id: 'body' };
  const fullscreen = { id: 'fullscreen' };
  assert.equal(fullscreenPortalTarget({ body, fullscreenElement: null }), body);
  assert.equal(fullscreenPortalTarget({ body, fullscreenElement: fullscreen }), fullscreen);
});

test('all floating UI uses the fullscreen-aware portal target', () => {
  for (const source of [customSelect, insertPopover, tooltip]) {
    assert.match(source, /fullscreenPortalTarget\(document\)/);
    assert.doesNotMatch(source, /createPortal\([\s\S]*?, document\.body\)/);
  }
});
