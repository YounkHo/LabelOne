import assert from 'node:assert/strict';
import test from 'node:test';

import { pipelineInsertPopoverPosition } from './pipeline-insert-popover.ts';

test('operator popover opens below when there is room and stays horizontally inside the viewport', () => {
  assert.deepEqual(
    pipelineInsertPopoverPosition({ left: 1100, right: 1120, top: 100, bottom: 120, width: 20 }, 1200, 800, 288, 244),
    { left: 904, top: 126, width: 288, maxHeight: 244, placement: 'bottom' },
  );
});

test('operator popover flips above a low anchor and limits height to available space', () => {
  assert.deepEqual(
    pipelineInsertPopoverPosition({ left: 900, right: 920, top: 700, bottom: 720, width: 20 }, 1200, 760, 288, 244),
    { left: 766, top: 450, width: 288, maxHeight: 244, placement: 'top' },
  );
  assert.deepEqual(
    pipelineInsertPopoverPosition({ left: 100, right: 120, top: 90, bottom: 110, width: 20 }, 500, 180, 288, 244),
    { left: 8, top: 8, width: 288, maxHeight: 76, placement: 'top' },
  );
});
