import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const component = readFileSync(new URL('../components/pipeline-parameter-control.tsx', import.meta.url), 'utf8');
const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('exact numeric parameters use editable number drafts instead of million-step sliders', () => {
  assert.match(component, /pipelineParameterSchemaForContext\(schema, context\)/);
  assert.match(component, /className="pipeline-parameter-number-field"/);
  assert.match(component, /onChange=\{\(event\) => setNumberDraft\(event\.target\.value\)\}/);
  assert.match(component, /onBlur=\{commitNumber\}/);
  assert.match(component, /effectiveSchema\.multipleOf \?\? \(effectiveSchema\.type === 'integer' \? 1 : 'any'\)/);
  assert.match(css, /\.pipeline-parameter-number-field\{min-width:0;display:grid/);
});

test('node parameters receive dimensions resolved immediately before the selected operator', () => {
  assert.match(page, /const selectedOperatorInputTransform = selectedOperator \? resolvePipelineCoordinateTransform\(/);
  assert.match(page, /selectedOperator\.id,/);
  assert.match(page, /inputWidth=\{selectedOperatorInputTransform\?\.width\} inputHeight=\{selectedOperatorInputTransform\?\.height\}/);
});

test('crop switches between one ratio slider and an exact four-number region', () => {
  assert.match(page, /className="crop-parameter-mode" role="group" aria-label="裁剪配置方式"/);
  assert.match(page, /边缘比例/);
  assert.match(page, /精确区域/);
  assert.match(page, /cropUsesExactRegion \? cropRegionParameterNames\.includes\(name\) : name === 'margin_ratio'/);
  assert.match(page, /onChange\('width', Math\.max\(1, Math\.floor\(inputWidth \?\? 1\)\)\)/);
  assert.match(page, /for \(const name of cropRegionParameterNames\) onChange\(name, undefined\)/);
  assert.match(css, /\.crop-parameter-mode\{display:grid;grid-template-columns:1fr 1fr/);
});
