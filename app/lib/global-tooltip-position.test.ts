import assert from 'node:assert/strict';
import test from 'node:test';

import { positionGlobalTooltip } from './global-tooltip-position.ts';

test('global tooltip prefers below and stays inside horizontal viewport edges', () => {
  assert.deepEqual(positionGlobalTooltip(
    { left: 4, top: 20, right: 24, bottom: 40, width: 20, height: 20 },
    { width: 180, height: 44 },
    { width: 320, height: 240 },
  ), { left: 8, top: 48, placement: 'bottom' });
});

test('global tooltip flips above near the bottom edge', () => {
  assert.deepEqual(positionGlobalTooltip(
    { left: 180, top: 210, right: 220, bottom: 230, width: 40, height: 20 },
    { width: 120, height: 50 },
    { width: 400, height: 250 },
  ), { left: 140, top: 152, placement: 'top' });
});

test('pointer-sized anchors position the tooltip at the cursor', () => {
  assert.deepEqual(positionGlobalTooltip(
    { left: 100, top: 60, right: 100, bottom: 60, width: 0, height: 0 },
    { width: 80, height: 30 },
    { width: 320, height: 240 },
  ), { left: 60, top: 68, placement: 'bottom' });
});
