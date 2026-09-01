import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('every right sidebar tab stays fixed and rejects vertical panel scrolling', () => {
  assert.match(css, /\.workspace \{[^}]*height:calc\(100vh - 54px\);[^}]*min-height:0;[^}]*overflow:hidden/);
  assert.match(css, /\.sidebar \{[^}]*height:100%;min-height:0/);
  assert.match(css, /\.right-sidebar \{[^}]*max-height:100%;[^}]*grid-template-rows:45px minmax\(0,1fr\);[^}]*overflow:hidden/);
  assert.match(css, /\.right-panel-scroll,\.right-sidebar\.layers-tab \.right-panel-scroll,\.right-sidebar\.pipeline-tab \.right-panel-scroll\{[^}]*height:100%;[^}]*min-height:0;overflow:hidden\}/);
  assert.match(css, /\.right-sidebar\.pipeline-tab \.right-panel-scroll\{height:100%;overflow:hidden\}/);
  assert.doesNotMatch(css, /\.right-panel-scroll[^}]*overflow-y:auto/);
  assert.match(page, /className={`right-panel-scroll tab-\$\{rightTab\}`}/);
});

test('fixed panels retain only their purpose-built internal content scrollers', () => {
  assert.match(css, /\.pipeline-panel>\.flow-canvas\{min-height:120px;flex:1;overflow:auto\}/);
  assert.match(css, /\.layers-panel \.annotation-object-list\{min-height:0;max-height:none;flex:1\}/);
  assert.match(css, /\.agent-messages \{ min-height:0; overflow:auto;/);
  assert.match(css, /\.right-sidebar\.inference-tab \.right-panel-scroll\{height:100%;overflow:hidden\}/);
  assert.match(css, /\.inference-panel\{height:100%;min-height:0;display:flex;flex-direction:column;overflow:hidden/);
  assert.match(css, /\.inference-results-region\{[^}]*overflow-x:hidden;overflow-y:auto/);
  for (const tab of ['layers', 'pipeline', 'inference', 'agent']) {
    assert.match(page, new RegExp(`rightTab === '${tab}'`));
  }
});
