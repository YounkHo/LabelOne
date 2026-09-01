import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('the top path exposes separate directory and file-name text targets', () => {
  assert.match(page, /resolveCurrentFilePath\(\{ datasetPath: dataset\.path, fileName: currentFile\?\.name, displayPath: currentFile\?\.displayPath, imagePath: currentFile\?\.imagePath \}\)/);
  assert.match(page, /const canCopyAbsoluteImagePath = Boolean\(currentFile\?\.imagePath && currentFilePath\.isAbsolute\)/);
  assert.match(page, /className={`current-file-location \$\{canCopyAbsoluteImagePath \? 'has-full-path' : 'static-path'\}`}/);
  assert.match(page, /canCopyAbsoluteImagePath \? <button type="button" className="path-copy-segment path-copy-full" aria-label="复制当前图片完整绝对路径"/);
  assert.match(page, /<span className="path-copy-directory-text">\{currentFilePath\.directoryLabel\}<\/span><span className="path-copy-filename-reserve" aria-hidden="true">\{currentFilePath\.fileName\}<\/span><\/button> : <span className="path-copy-static path-copy-directory"/);
  assert.match(page, /<span>本地路径不可用 · \{currentFilePath\.directoryLabel\}<\/span>/);
  assert.match(page, /className="path-copy-segment path-copy-filename" aria-label="复制当前图片文件名" disabled=\{!currentFilePath\.fileName\}/);
  assert.match(page, /copyTextToClipboard\(currentFilePath\.fullPath, '已复制完整路径'\)/);
  assert.match(page, /copyTextToClipboard\(currentFilePath\.fileName, `已复制文件名：\$\{currentFilePath\.fileName\}`\)/);
  assert.doesNotMatch(page, /path-copy-button|path-copy-actions|>⧉</);
});

test('clipboard feedback has a browser API and a controlled fallback', () => {
  assert.match(page, /navigator\.clipboard\?\.writeText/);
  assert.match(page, /document\.execCommand\('copy'\)/);
  assert.match(page, /notify\('复制失败，请检查浏览器剪贴板权限'\)/);
});

test('each text target owns its hover state while long paths truncate first', () => {
  assert.match(css, /\.current-file-location\{[^}]*min-width:0;flex:1;display:flex;[^}]*overflow:hidden/);
  assert.match(css, /\.path-copy-segment>span,\.path-copy-static>span\{[^}]*overflow:hidden;text-overflow:ellipsis/);
  assert.match(css, /\.path-copy-full\{width:100%;flex:1/);
  assert.match(css, /\.path-copy-filename-reserve\{min-width:24px;max-width:45%;flex:0 1 auto;[^}]*visibility:hidden/);
  assert.match(css, /\.path-copy-filename\{min-width:24px;max-width:45%;flex:0 1 auto/);
  assert.match(css, /\.has-full-path>\.path-copy-filename\{position:absolute;z-index:2;top:0;right:0\}/);
  assert.match(css, /\.path-copy-static\{cursor:default/);
  assert.match(css, /\.path-copy-segment::after\{content:attr\(data-copy-hint\);[^}]*opacity:0/);
  assert.match(css, /\.path-copy-segment:focus-visible\{/);
  assert.match(css, /\.path-copy-segment:focus-visible::after\{opacity:1\}/);
  assert.match(css, /@media\(hover:hover\) and \(pointer:fine\)\{\.path-copy-segment:hover:not\(:disabled\)::after\{opacity:1\}/);
  assert.match(css, /\.path-copy-full:hover:not\(:disabled\)\{background:#14352d;[^}]*#3c7a69/);
  assert.match(css, /\.path-copy-full:hover:not\(:disabled\)\+\.path-copy-filename\{color:#c2eee2\}/);
  assert.match(css, /\.path-copy-filename:hover:not\(:disabled\)\{background:#172d47;[^}]*#416f9d/);
  assert.match(css, /\.has-full-path:has\(>\.path-copy-filename:hover\)>\.path-copy-full\{background:transparent;color:#64758a;box-shadow:none\}/);
  assert.doesNotMatch(css, /\.path-copy-button|\.path-copy-actions|\.current-file-path/);
});
