import assert from 'node:assert/strict';
import test from 'node:test';

import { scrollTopForActiveListItem } from './virtual-list-scroll.ts';

test('active virtual-list rows scroll into the nearest visible position', () => {
  assert.equal(scrollTopForActiveListItem({ activeIndex: 2, currentScrollTop: 0, itemCount: 100, rowHeight: 55, viewportHeight: 220 }), 0);
  assert.equal(scrollTopForActiveListItem({ activeIndex: 4, currentScrollTop: 0, itemCount: 100, rowHeight: 55, viewportHeight: 220 }), 55);
  assert.equal(scrollTopForActiveListItem({ activeIndex: 3, currentScrollTop: 220, itemCount: 100, rowHeight: 55, viewportHeight: 220 }), 165);
});

test('active virtual-list scrolling handles appended pages and invalid targets', () => {
  assert.equal(scrollTopForActiveListItem({ activeIndex: 100, currentScrollTop: 5280, itemCount: 101, rowHeight: 55, viewportHeight: 220 }), 5335);
  assert.equal(scrollTopForActiveListItem({ activeIndex: -1, currentScrollTop: 330, itemCount: 101, rowHeight: 55, viewportHeight: 220 }), 330);
});
