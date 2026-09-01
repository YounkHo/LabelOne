import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { normalizeUiLanguage, translateUiText, UI_LANGUAGE_STORAGE_KEY } from './i18n.ts';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../components/global-settings-page.tsx', import.meta.url), 'utf8');
const welcome = readFileSync(new URL('../components/welcome-screen.tsx', import.meta.url), 'utf8');
const modelPicker = readFileSync(new URL('../components/model-picker-dialog.tsx', import.meta.url), 'utf8');
const bridge = readFileSync(new URL('../components/ui-language-bridge.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('language preference accepts only Chinese and English', () => {
  assert.equal(UI_LANGUAGE_STORAGE_KEY, 'labelone-ui-language-v1');
  assert.equal(normalizeUiLanguage('en'), 'en');
  assert.equal(normalizeUiLanguage('zh-CN'), 'zh-CN');
  assert.equal(normalizeUiLanguage('fr'), 'zh-CN');
  assert.equal(normalizeUiLanguage(null), 'zh-CN');
});

test('English catalog covers primary workspace, pipeline, inference and accessibility labels', () => {
  assert.equal(translateUiText('处理流', 'en'), 'Pipeline');
  assert.equal(translateUiText('推理参数', 'en'), 'Inference parameters');
  assert.equal(translateUiText('显示真实像素网格', 'en'), 'Show true pixel grid');
  assert.equal(translateUiText('参数已更新 · 保留上一版分屏，正在重新计算', 'en'), 'Parameters updated · Keeping the previous split while recomputing');
  assert.equal(translateUiText('人工标注', 'en'), 'Manual annotations');
  assert.equal(translateUiText('前往推理配置', 'en'), 'Open inference settings');
  assert.equal(translateUiText('处理流', 'zh-CN'), '处理流');
});

test('Chinese catalog removes hard-coded English chrome without translating user data', () => {
  assert.equal(translateUiText('LOCAL IMAGE WORKSPACE', 'zh-CN'), '本地图像工作台');
  assert.equal(translateUiText('Layer Visualization', 'zh-CN'), '中间层可视化');
  assert.equal(translateUiText('Classification', 'zh-CN'), '分类');
  assert.equal(translateUiText('Remote Inference Privacy', 'zh-CN'), '远程推理隐私');
  assert.equal(translateUiText('Unsaved Annotation', 'zh-CN'), '未保存标注');
  assert.equal(translateUiText('my-custom-label', 'zh-CN'), 'my-custom-label');
});

test('all static eyebrow headings have a complete translation in both directions', () => {
  const headings = [page, settings, welcome, modelPicker].flatMap((source) => [...source.matchAll(/className="eyebrow">([^<{]+)</g)].map((match) => match[1].trim()));
  for (const heading of headings) {
    if (/^[A-Za-z&/\s]+$/.test(heading)) assert.notEqual(translateUiText(heading, 'zh-CN'), heading, `missing Chinese heading: ${heading}`);
    if (/\p{Script=Han}/u.test(heading)) assert.doesNotMatch(translateUiText(heading, 'en'), /\p{Script=Han}/u, `missing English heading: ${heading}`);
  }
});

test('top-right and settings language switches persist and update document language', () => {
  assert.match(page, /const \[uiLanguage, setUiLanguage\] = useState<UiLanguage>\('zh-CN'\)/);
  assert.match(page, /window\.localStorage\.setItem\(UI_LANGUAGE_STORAGE_KEY, next\)/);
  assert.match(page, /className="top-language-button"/);
  assert.match(page, /<UiLanguageBridge language=\{uiLanguage\} \/>/);
  assert.match(settings, /className="settings-language-button"/);
  assert.match(bridge, /document\.documentElement\.lang = language/);
  assert.match(bridge, /new MutationObserver/);
  assert.doesNotMatch(bridge, /language === 'zh-CN'\) return/);
  assert.match(bridge, /'aria-label'.*'title'.*'placeholder'.*'data-tooltip'/s);
  assert.match(css, /\.top-language-button,\.settings-language-button\{/);
  assert.match(css, /\.language-announcement\{position:fixed;width:1px;height:1px/);
});
