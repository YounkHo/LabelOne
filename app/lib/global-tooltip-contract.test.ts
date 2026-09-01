import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const component = readFileSync(new URL('../components/global-tooltip.tsx', import.meta.url), 'utf8');
const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../components/global-settings-page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('global tooltip portals all title and data-tooltip hints outside clipped surfaces', () => {
  assert.match(component, /closest<HTMLElement>\('\[data-tooltip\]'\)[\s\S]*closest<HTMLElement>\('\[title\]'\)/);
  assert.match(component, /target\.removeAttribute\('title'\)/);
  assert.match(component, /createPortal\([\s\S]*fullscreenPortalTarget\(document\)\)/);
  assert.match(component, /document\.addEventListener\('focusin'/);
  assert.match(component, /event\.key === 'Escape'/);
  assert.match(component, /event\.pointerType === 'touch'/);
  assert.match(component, /target\.dataset\.tooltipAnchor === 'pointer'/);
  assert.match(component, /document\.addEventListener\('pointermove', onPointerMove\)/);
  assert.match(component, /width: 0, height: 0/);
  assert.match(page, /<GlobalTooltip \/>/);
  assert.match(css, /\.global-tooltip\{position:fixed/);
});

test('shortcut controls expose a description and keycap to the global tooltip', () => {
  assert.match(settings, /data-tooltip-title=\{definition\.label\}/);
  assert.match(settings, /data-tooltip=\{definition\.description\}/);
  assert.match(settings, /data-shortcut=\{shortcut\}/);
  assert.match(page, /data-tooltip-title=\{definition\.label\}/);
  assert.match(page, /label="放大" description="以画布中心为锚点放大/);
  assert.match(page, /data-tooltip-title="上一张图片"/);
  assert.match(page, /data-tooltip-title="下一张图片"/);
  assert.match(page, /data-shortcut=\{`\$\{displayShortcut\(shortcuts\['navigation\.previous'\], useMacShortcutSymbols\)\} \/ ←`\}/);
  assert.doesNotMatch(page, /canvas-page-button[^>]*>[\s\S]{0,200}<kbd/);
  assert.match(css, /\.global-tooltip>kbd\{/);
});
