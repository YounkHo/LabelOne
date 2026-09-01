import assert from 'node:assert/strict';
import test from 'node:test';

import { CANVAS_CONTROL_POINT_RADIUS_PX, CANVAS_VERTEX_CONTROL_DIAMETER_PX, canvasAnnotationOpticalScale, canvasLabelLayout, canvasLabelLayouts, canvasLabelOpacity, canvasLabelTopVertexAnchor, canvasVisibleImageBounds, fitCanvasLabelText, truncateCanvasLabel, type CanvasLabelLayout } from './canvas-label.ts';

function labelsOverlap(left: CanvasLabelLayout, right: CanvasLabelLayout): boolean {
  return left.x < right.x + right.width && left.x + left.width > right.x && left.y < right.y + right.height && left.y + left.height > right.y;
}

test('label keeps the same screen dimensions across canvas zoom levels', () => {
  const units = [2, 1, 0.25]; // 50%, 100%, 400%
  const layouts = units.map((unit) => canvasLabelLayout([[100, 100], [300, 300]], 'scratch · 96%', 1000, 800, unit)!);
  layouts.forEach((layout, index) => {
    const unit = units[index];
    assert.equal(layout.fontSize / unit, 11);
    assert.equal(layout.height / unit, 19);
    assert.equal(layout.paddingX / unit, 5);
    assert.equal((layout.y + layout.height + 3 * unit) / unit, 100 / unit);
  });
  assert.equal(layouts[0].width / units[0], layouts[1].width / units[1]);
  assert.equal(layouts[1].width / units[1], layouts[2].width / units[2]);
});

test('label flips inside near the top and stays within horizontal image bounds', () => {
  const topLeft = canvasLabelLayout([[0, 1], [40, 30]], 'very-long-category-name', 200, 120, 1)!;
  assert.equal(topLeft.placement, 'inside');
  assert.equal(topLeft.y, 4);
  assert.ok(topLeft.x >= 4);
  assert.ok(topLeft.x + topLeft.width <= 196);

  const right = canvasLabelLayout([[198, 80], [200, 100]], 'edge', 200, 120, 1)!;
  assert.equal(right.y + right.height + 3, 80);
  assert.equal(right.x + right.width, 196);
});

test('rotated rectangle label stays centered on its real top vertex', () => {
  const points = [[200, 100], [300, 200], [200, 300], [100, 200]];
  const orders = [points, [...points].reverse(), [points[2], points[3], points[0], points[1]]];
  for (const ordered of orders) {
    const anchor = canvasLabelTopVertexAnchor(ordered);
    assert.deepEqual(anchor, { x: 200, y: 100, align: 'center' });
    for (const unit of [2, 1, 0.25, 0.05]) {
      const [layout] = canvasLabelLayouts([{ points: ordered, text: 'rotation', anchor }], 400, 400, unit);
      assert.ok(layout);
      assert.equal(layout.x + layout.width / 2, 200);
      assert.ok(Math.abs((anchor.y - layout.y - layout.height) / unit - 3) < 1e-9);
      assert.equal(layout.height / unit, 19);
      assert.equal(layout.placement, 'above');
    }
  }
});

test('label text truncation counts user-visible glyphs', () => {
  assert.equal(truncateCanvasLabel('scratch', 8), 'scratch');
  assert.equal(truncateCanvasLabel('超长类别标签abcdef', 8), '超长类别标签a…');
  const fitted = fitCanvasLabelText('超长类别标签-with-a-very-long-suffix');
  assert.ok(fitted.endsWith('…'));
  assert.ok(fitted.length < '超长类别标签-with-a-very-long-suffix'.length);
});

test('visible image bounds follow canvas zoom and pan in image coordinates', () => {
  assert.deepEqual(canvasVisibleImageBounds(1000, 800, 500, 400, 500, 400, { scale: 4, x: 0, y: 0 }), {
    left: 375,
    top: 300,
    right: 625,
    bottom: 500,
  });
  assert.deepEqual(canvasVisibleImageBounds(1000, 800, 500, 400, 500, 400, { scale: 4, x: 100, y: -40 }), {
    left: 325,
    top: 320,
    right: 575,
    bottom: 520,
  });
});

test('partially visible shapes keep a sticky label while offscreen shapes do not', () => {
  const visible = { left: 375, top: 300, right: 625, bottom: 500 };
  const sticky = canvasLabelLayout([[100, 100], [500, 450]], 'large-region', 1000, 800, 0.5, visible)!;
  const nextSticky = canvasLabelLayout([[100, 100], [500, 450]], 'second-region', 1000, 800, 0.5, visible, 1)!;
  assert.equal(sticky.placement, 'sticky');
  assert.ok(sticky.x >= visible.left + 2);
  assert.ok(sticky.y >= visible.top + 2);
  assert.ok(sticky.x + sticky.width <= visible.right - 2);
  assert.equal(nextSticky.y - sticky.y, 11); // (19px label + 3px gap) × 0.5 image units/px
  assert.equal(canvasLabelLayout([[0, 0], [100, 100]], 'offscreen', 1000, 800, 0.5, visible), null);
});

test('maximum zoom keeps the same screen-sized label metrics', () => {
  const layout = canvasLabelLayout([[490, 390], [510, 410]], 'max-zoom', 1000, 800, 0.05, { left: 450, top: 350, right: 550, bottom: 450 })!;
  assert.equal(layout.fontSize / 0.05, 11);
  assert.equal(layout.height / 0.05, 19);
  assert.equal(layout.paddingX / 0.05, 5);
  assert.equal(layout.radius / 0.05, 4);
  assert.ok(layout.width / 0.05 >= 36 && layout.width / 0.05 <= 220);
});

test('maximum zoom quiets annotation strokes without shrinking direct-manipulation handles', () => {
  assert.equal(canvasAnnotationOpticalScale(1), 1);
  assert.equal(canvasAnnotationOpticalScale(8), 0.72);
  assert.equal(CANVAS_CONTROL_POINT_RADIUS_PX, 7);
  assert.equal(CANVAS_VERTEX_CONTROL_DIAMETER_PX, 14);
});

test('canvas labels fade like map labels and disappear at low zoom', () => {
  assert.equal(canvasLabelOpacity(1), 1);
  assert.equal(canvasLabelOpacity(0.72), 1);
  assert.ok(canvasLabelOpacity(0.6) > 0 && canvasLabelOpacity(0.6) < 1);
  assert.equal(canvasLabelOpacity(0.48), 0);
  assert.equal(canvasLabelOpacity(0.25), 0);
  assert.equal(canvasLabelOpacity(0.46, true), 1);
  assert.ok(canvasLabelOpacity(0.36, true) > 0 && canvasLabelOpacity(0.36, true) < 1);
  assert.equal(canvasLabelOpacity(0.28, true), 0);
  assert.equal(canvasLabelOpacity(Number.NaN), 0);
});

test('batch layout separates labels for overlapping visible shapes', () => {
  const requests = ['scratch', 'particle', 'chip-edge'].map((text) => ({ points: [[100, 100], [260, 240]], text }));
  const layouts = canvasLabelLayouts(requests, 800, 600, 1).map((layout) => layout!);
  assert.equal(layouts.length, 3);
  for (let left = 0; left < layouts.length; left += 1) {
    for (let right = left + 1; right < layouts.length; right += 1) assert.equal(labelsOverlap(layouts[left], layouts[right]), false);
  }
  assert.deepEqual(canvasLabelLayouts(requests, 800, 600, 1), canvasLabelLayouts(requests, 800, 600, 1));
});

test('selected label priority keeps the natural position and moves lower-priority labels', () => {
  const points = [[100, 100], [260, 240]];
  const natural = canvasLabelLayout(points, 'selected', 800, 600, 1)!;
  const layouts = canvasLabelLayouts([
    { points, text: 'prediction', priority: 0 },
    { points, text: 'selected', priority: 2 },
  ], 800, 600, 1).map((layout) => layout!);
  assert.equal(layouts[1].x, natural.x);
  assert.equal(layouts[1].y, natural.y);
  assert.equal(labelsOverlap(layouts[0], layouts[1]), false);
});

test('dense labels stay near their frame and hide instead of escaping into distant rows', () => {
  const points = [[100, 100], [260, 240]];
  const requests = [
    ...Array.from({ length: 20 }, (_, index) => ({ points, text: `ordinary-${index}`, priority: 1 })),
    { points, text: 'selected', priority: 2 },
  ];
  const layouts = canvasLabelLayouts(requests, 800, 600, 1);
  const ordinaryVisible = layouts.slice(0, -1).filter((layout): layout is CanvasLabelLayout => layout !== null);
  assert.ok(ordinaryVisible.length > 0 && ordinaryVisible.length < 20);
  ordinaryVisible.forEach((layout) => {
    const horizontalGap = layout.x + layout.width < 100 ? 100 - (layout.x + layout.width) : layout.x > 260 ? layout.x - 260 : 0;
    const verticalGap = layout.y + layout.height < 100 ? 100 - (layout.y + layout.height) : layout.y > 240 ? layout.y - 240 : 0;
    assert.ok(Math.hypot(horizontalGap, verticalGap) <= 25);
  });
  const selected = layouts.at(-1)!;
  const natural = canvasLabelLayout(points, 'selected', 800, 600, 1)!;
  assert.ok(selected);
  assert.deepEqual([selected.x, selected.y], [natural.x, natural.y]);
});
