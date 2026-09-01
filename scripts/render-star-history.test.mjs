import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeStarHistory, renderStarHistory } from './render-star-history.mjs';

const fixture = {
  repository: 'YounkHo/LabelOne',
  created_at: '2026-09-01T00:00:00Z',
  starred_at: ['2026-09-03T12:00:00Z', '2026-09-01T08:00:00Z'],
  updated_at: '2026-09-08T00:00:00Z',
};

test('normalizes star timestamps into chronological order', () => {
  const normalized = normalizeStarHistory(fixture);
  assert.deepEqual(normalized.stars, [Date.parse(fixture.starred_at[1]), Date.parse(fixture.starred_at[0])]);
});

test('renders accessible light and dark step charts without external resources', () => {
  for (const theme of ['light', 'dark']) {
    const svg = renderStarHistory(fixture, theme);
    assert.match(svg, /<title id="title">YounkHo\/LabelOne star history<\/title>/);
    assert.match(svg, />2 stars</);
    assert.match(svg, /M [\d.]+ [\d.]+ H [\d.]+ V [\d.]+/);
    assert.doesNotMatch(svg, /<(?:script|image)|(?:href|src)=/);
  }
});
