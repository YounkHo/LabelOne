import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('annotation shape icons share one SVG geometry and stroke system', () => {
  assert.match(page, /function ShapeTypeIcon\(\{ shapeType \}: \{ shapeType: string \}\)/);
  assert.match(page, /shapeType === 'rect' \? 'rectangle'/);
  for (const shapeType of ['rectangle', 'rotation', 'polygon', 'point', 'line', 'linestrip', 'circle']) {
    assert.match(page, new RegExp(`normalizedShapeType === '${shapeType}'`));
  }
  assert.doesNotMatch(page, /shapeTypeGlyphs|RotationRectangleIcon/);
  assert.match(page, /<ShapeTypeIcon shapeType=\{id === 'brush' \? 'linestrip' : id\} \/>/);
  assert.match(page, /className=\{`annotation-object-shape[\s\S]*?<ShapeTypeIcon shapeType=\{shape\.shape_type\} \/>/);
  assert.match(page, /manual-shape-kind[\s\S]*?<ShapeTypeIcon shapeType=\{pendingManualShape\.shape\.shape_type\} \/>/);
  assert.match(css, /\.annotation-shape-type-icon\{width:15px;height:15px;display:block;overflow:visible;fill:none;stroke:currentColor;stroke-width:1\.45/);
  assert.match(css, /\.annotation-object-shape \.annotation-shape-type-icon\{width:13px;height:13px\}/);
  assert.match(css, /\.annotation-object-icon \.annotation-object-visibility\{right:-3px;bottom:-3px;width:12px;height:12px/);
});

test('inference configuration stays fixed and keeps compact results in an internal region', () => {
  const inference = page.match(/rightTab === 'inference' && <section className="inference-panel">[\s\S]*?rightTab === 'agent'/)?.[0] ?? '';
  assert.match(css, /\.inference-panel\{height:100%;min-height:0;display:flex;flex-direction:column;overflow:hidden/);
  assert.match(inference, /className="feature-capture-card"/);
  assert.match(css, /\.feature-controls\{grid-template-columns:repeat\(3,minmax\(0,1fr\)\)\}/);
  assert.match(css, /\.feature-layer-meta\{display:grid;grid-template-columns:minmax\(0,1fr\) minmax\(0,auto\)/);
  assert.match(css, /\.tensor-shape\{grid-template-columns:64px minmax\(0,1fr\)\}/);
  assert.match(css, /\.model-run\{box-shadow:0 7px 22px #182f5d52!important\}/);
  assert.match(css, /\.inference-result-strip\{/);
  assert.match(css, /\.feature-result-compact summary\{/);
});
