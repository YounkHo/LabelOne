import assert from 'node:assert/strict';
import test from 'node:test';

import { navigatorPointToView, navigatorViewport, type CanvasNavigatorMetrics } from './canvas-navigator.ts';

const metrics: CanvasNavigatorMetrics = {
  navigatorWidth: 140,
  navigatorHeight: 70,
  imageWidth: 1000,
  imageHeight: 500,
  viewportWidth: 500,
  viewportHeight: 250,
};

test('navigator viewport reflects canvas zoom and translation', () => {
  assert.deepEqual(navigatorViewport({ scale: 2, x: 0, y: 0 }, metrics), {
    left: 52.5,
    top: 26.25,
    width: 35,
    height: 17.5,
  });
  const moved = navigatorViewport({ scale: 2, x: -500, y: -125 }, metrics);
  assert.ok(moved.left > 52.5);
  assert.ok(moved.top > 26.25);
});

test('navigator drag maps back to bounded canvas translation', () => {
  const rightEdge = navigatorPointToView({ x: 140, y: 35 }, { scale: 2, x: 0, y: 0 }, metrics);
  assert.deepEqual(rightEdge, { scale: 2, x: -750, y: 0 });
  const bottomEdge = navigatorPointToView({ x: 70, y: 70 }, { scale: 2, x: 0, y: 0 }, metrics);
  assert.deepEqual(bottomEdge, { scale: 2, x: 0, y: -375 });
});

test('navigator mapping round-trips the visible center and handles contain letterboxing', () => {
  const source = { scale: 2, x: -300, y: 120 };
  const box = navigatorViewport(source, metrics);
  const restored = navigatorPointToView({ x: box.left + box.width / 2, y: box.top + box.height / 2 }, source, metrics);
  assert.ok(Math.abs(restored.x - source.x) < 1e-9);
  assert.ok(Math.abs(restored.y - source.y) < 1e-9);

  const portrait = { ...metrics, imageWidth: 500, imageHeight: 1000 };
  const portraitBox = navigatorViewport({ scale: 1, x: 0, y: 0 }, portrait);
  assert.ok(portraitBox.left >= 52.5);
  assert.ok(portraitBox.left + portraitBox.width <= 87.5);
});

test('navigator disables panning when the scaled image fits the viewport', () => {
  assert.deepEqual(navigatorPointToView({ x: 140, y: 70 }, { scale: 0.5, x: 10, y: 10 }, metrics), { scale: 0.5, x: 0, y: 0 });
});
