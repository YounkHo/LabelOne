import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const shortcuts = readFileSync(new URL('./keyboard-shortcuts.ts', import.meta.url), 'utf8');

test('freehand drawing explains that it creates an open continuous line', () => {
  assert.match(page, /\['brush', '自由线'\]/);
  assert.match(page, /<ShapeTypeIcon shapeType=\{id === 'brush' \? 'linestrip' : id\}/);
  assert.match(page, /data-tooltip=\{guidance\}/);
  assert.match(page, /自由线 · 按住左键拖动绘制 · 松开完成 · 不会闭合/);
  assert.match(page, /正在绘制开放连续线 · 松开完成并选择标签/);
  assert.match(page, /拖动距离太短，未创建连续线；请按住并拖动一段距离/);
  assert.match(shortcuts, /id: 'tool\.brush', label: '自由线', description: '按住拖动绘制开放连续线'/);
  assert.doesNotMatch(page, /区域画笔|画笔笔宽|brush-inline-width|brushWidth/);
});

test('freehand preview and pending shape stay open polylines with category color', () => {
  assert.match(page, /const freehandGuideStyle = annotationCategoryStyle\(freehandGuideLabel, annotationCategoryColorOverrides\)/);
  assert.match(page, /<polyline className="draw-preview freeform shape-linestrip freehand-line-preview"[\s\S]*?fill="none" style=\{freehandGuideStyle\}/);
  assert.match(page, /createFreehandLine\(points,/);
  assert.match(page, /renderPendingManualShape\(pendingManualShape\.shape, pendingManualShapeStyle\)/);
  assert.match(page, /shape\.shape_type === 'line' \|\| shape\.shape_type === 'linestrip'\) return <polyline[^>]+fill="none"/);
  assert.match(css, /\.freehand-line-preview\{fill:none;stroke-linecap:round;stroke-linejoin:round/);
  assert.match(css, /\.draw-preview\.pending-label\{fill:var\(--shape-fill,#ffcf8a24\);stroke:var\(--shape-color,#ffd496\)/);
  assert.doesNotMatch(page, /brush-cursor-preview|brush-footprint-preview|brush-centerline-preview|createBrushPolygon/);
  assert.doesNotMatch(css, /brush-cursor-preview|brush-footprint-preview|brush-centerline-preview/);
});

test('freehand mode keeps the standard crosshair and has no area footprint', () => {
  assert.match(page, /className="canvas-crosshair" aria-hidden="true"/);
  assert.doesNotMatch(page, /BrushAreaIcon|brush-area-icon|brush-mode|<circle className="brush-cursor-preview"/);
  assert.doesNotMatch(css, /brush-area-icon|brush-mode|brush-inline-width/);
});
