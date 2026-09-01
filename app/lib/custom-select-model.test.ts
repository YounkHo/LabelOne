import assert from 'node:assert/strict';
import test from 'node:test';

import { nextEnabledOption, selectMenuPlacement } from './custom-select-model.ts';

test('custom select keyboard navigation skips disabled options and wraps', () => {
  const options = [{}, { disabled: true }, {}];
  assert.equal(nextEnabledOption(options, 0, 1), 2);
  assert.equal(nextEnabledOption(options, 2, 1), 0);
  assert.equal(nextEnabledOption(options, 0, -1), 2);
  assert.equal(nextEnabledOption([{ disabled: true }], 0, 1), -1);
});

test('custom select opens toward the side with useful viewport space', () => {
  assert.deepEqual(selectMenuPlacement({ top: 20, bottom: 50 }, 800, 240), { placement: 'bottom', top: 55, maxHeight: 240 });
  assert.deepEqual(selectMenuPlacement({ top: 700, bottom: 730 }, 760, 240), { placement: 'top', top: 455, maxHeight: 240 });
});
