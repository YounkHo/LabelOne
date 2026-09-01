import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  applyCanvasWheel,
  CANVAS_ZOOM_STEP,
  canvasScaleForSourcePixelSize,
  canvasScaleFromPercent,
  isSourcePixelGridVisible,
  isBrowserZoomKeyboardShortcut,
  maximumCanvasScaleForPixelInspection,
  normalizeWheelDelta,
  panCanvasView,
  resolveCanvasKeyboardCommand,
  sourcePixelScreenSize,
  zoomCanvasView,
  type CanvasView,
} from './canvas-viewport.ts';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('pointer-anchored zoom keeps the same image point under the pointer', () => {
  const view: CanvasView = { scale: 2, x: 40, y: -20 };
  const anchor = { x: 180, y: 90 };
  const before = {
    x: (anchor.x - view.x) / view.scale,
    y: (anchor.y - view.y) / view.scale,
  };
  const next = zoomCanvasView(view, CANVAS_ZOOM_STEP, 8, anchor);
  assert.ok(Math.abs((anchor.x - next.x) / next.scale - before.x) < 1e-9);
  assert.ok(Math.abs((anchor.y - next.y) / next.scale - before.y) < 1e-9);
});

test('zoom clamps at both canvas scale boundaries without moving at a reached boundary', () => {
  assert.deepEqual(zoomCanvasView({ scale: 8, x: 5, y: 7 }, 2, 8, { x: 10, y: 10 }), { scale: 8, x: 5, y: 7 });
  assert.equal(zoomCanvasView({ scale: 1, x: 0, y: 0 }, 0.01, 8).scale, 0.25);
});

test('custom zoom percentage uses dynamic bounds and a safe fallback', () => {
  assert.equal(canvasScaleFromPercent(10, 8), 0.25);
  assert.equal(canvasScaleFromPercent(175, 8), 1.75);
  assert.equal(canvasScaleFromPercent(900, 8), 8);
  assert.equal(canvasScaleFromPercent(Number.NaN, 8, 1.4), 1.4);
  assert.equal(canvasScaleFromPercent(12_000, 256), 120);
});

test('source pixel metrics map one image pixel to its real screen size', () => {
  assert.equal(sourcePixelScreenSize(2000, 500, 4), 1);
  assert.equal(sourcePixelScreenSize(2000, 500, 16), 4);
  assert.equal(canvasScaleForSourcePixelSize(2000, 500, 4), 16);
  assert.equal(canvasScaleForSourcePixelSize(2000, 500, 8), 32);
  assert.equal(sourcePixelScreenSize(0, 500, 4), 0);
  assert.equal(maximumCanvasScaleForPixelInspection(2000, 500, 32), 128);
  assert.equal(maximumCanvasScaleForPixelInspection(100, 800, 32), 4);
  assert.equal(maximumCanvasScaleForPixelInspection(10_000, 500, 32), 256);
  assert.equal(isSourcePixelGridVisible(true, 7.99, 8), false);
  assert.equal(isSourcePixelGridVisible(true, 8, 8), true);
  assert.equal(isSourcePixelGridVisible(false, 32, 32), false);
});

test('plain wheel pans while trackpad pinch or modified wheel zooms', () => {
  const view: CanvasView = { scale: 1, x: 10, y: 20 };
  assert.deepEqual(panCanvasView(view, 4, -6), { scale: 1, x: 6, y: 26 });
  const panned = applyCanvasWheel(view, { deltaX: 4, deltaY: -6, deltaMode: 0, ctrlKey: false, metaKey: false }, 8, { x: 0, y: 0 });
  assert.equal(panned.mode, 'pan');
  assert.deepEqual(panned.view, { scale: 1, x: 6, y: 26 });
  const pinched = applyCanvasWheel(view, { deltaX: 0, deltaY: -20, deltaMode: 0, ctrlKey: true, metaKey: false }, 8, { x: 30, y: 40 });
  assert.equal(pinched.mode, 'zoom');
  assert.ok(pinched.view.scale > view.scale);
});

test('wheel line and page deltas are normalized before zooming or panning', () => {
  assert.equal(normalizeWheelDelta(3, 0), 3);
  assert.equal(normalizeWheelDelta(3, 1), 48);
  assert.equal(normalizeWheelDelta(2, 2, 600), 1200);
});

test('keyboard mapping covers main keyboard and numpad zoom commands', () => {
  assert.equal(resolveCanvasKeyboardCommand('+'), 'zoom-in');
  assert.equal(resolveCanvasKeyboardCommand('='), 'zoom-in');
  assert.equal(resolveCanvasKeyboardCommand('', 'NumpadSubtract'), 'zoom-out');
  assert.equal(resolveCanvasKeyboardCommand('0'), 'fit');
  assert.equal(resolveCanvasKeyboardCommand('', 'Numpad1'), 'actual-size');
  assert.equal(resolveCanvasKeyboardCommand('a'), null);
});

test('browser zoom shortcuts are blocked outside the canvas without swallowing ordinary keys', () => {
  assert.equal(isBrowserZoomKeyboardShortcut({ key: '=', ctrlKey: false, metaKey: true }), true);
  assert.equal(isBrowserZoomKeyboardShortcut({ key: '-', ctrlKey: true, metaKey: false }), true);
  assert.equal(isBrowserZoomKeyboardShortcut({ key: '0', ctrlKey: true, metaKey: false }), true);
  assert.equal(isBrowserZoomKeyboardShortcut({ key: '1', ctrlKey: false, metaKey: true }), false);
  assert.equal(isBrowserZoomKeyboardShortcut({ key: '+', ctrlKey: false, metaKey: false }), false);
});

test('application boundary blocks browser zoom before events reach non-canvas UI', () => {
  assert.match(page, /window\.addEventListener\('keydown', preventBrowserZoomKey, \{ capture: true \}\)/);
  assert.match(page, /if \(event\.ctrlKey \|\| event\.metaKey\) event\.preventDefault\(\)/);
  assert.match(page, /window\.addEventListener\('wheel', preventBrowserWheelZoom, \{ capture: true, passive: false \}\)/);
  assert.match(page, /window\.addEventListener\('gesturestart', onGestureStart, \{ capture: true, passive: false \}\)/);
  assert.match(page, /window\.addEventListener\('gesturechange', onGestureChange, \{ capture: true, passive: false \}\)/);
});
