import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const settingsPage = readFileSync(new URL('../components/global-settings-page.tsx', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('global settings uses an opaque page-level modal instead of the annotation popover', () => {
  assert.match(page, /settingsOpen && <GlobalSettingsPage/);
  assert.match(settingsPage, /id="global-settings-page" className="global-settings-page" role="dialog" aria-modal="true"/);
  assert.match(settingsPage, /模型与存储/);
  assert.match(settingsPage, /AI 服务/);
  assert.match(settingsPage, /系统与网络/);
  assert.match(settingsPage, /算子库/);
  assert.match(settingsPage, /快捷键/);
  assert.match(css, /\.global-settings-page\{position:fixed;z-index:100;inset:0;[^}]*background:#080d14/);
  assert.doesNotMatch(settingsPage, /annotation-box|real-annotation-layer|当前图对象|标注框/);
  assert.doesNotMatch(page, /top-settings-popover|top-setting-switch|top-setting-range/);
  assert.doesNotMatch(css, /\.top-settings-popover|\.top-setting-switch|\.top-setting-range/);
});

test('cloud AI settings keep credentials in the local-service environment and preserve tool safety', () => {
  assert.match(settingsPage, /OpenAI-compatible 服务/);
  assert.match(settingsPage, /API Key 环境变量/);
  assert.match(settingsPage, /不会进入浏览器、设置 JSON、日志或 Agent 历史/);
  assert.match(settingsPage, /只接收你输入的 Agent 文字和可用工具清单/);
  assert.match(page, /backend\.updateApplicationSettings\(\{ cloud_ai: cloudAiDraft \}\)/);
  assert.doesNotMatch(settingsPage, /type="password"|name="api_key"/);
});

test('operators are imported one package at a time from system settings and merged into the registry', () => {
  assert.match(settingsPage, /section === 'operators'/);
  assert.match(settingsPage, /从系统导入算子/);
  assert.match(settingsPage, /安装并合并到算子库/);
  assert.match(settingsPage, /立即刷新并出现在处理流插入菜单/);
  assert.match(page, /aria-label="从系统导入算子包"/);
  assert.match(backend, /\/pipelines\/operators\/inspect\?filename=/);
  assert.match(backend, /await refreshPipelineRegistry\(\)/);
  assert.match(page, /inspectPipelineOperator\(file\)/);
  assert.match(page, /confirmOperatorPackageImport/);
  assert.match(settingsPage, /<strong>\{operator\.title\}<\/strong><small>\{operator\.description\}<\/small><code>ID · \{operator\.kind\}<\/code>/);
  assert.match(settingsPage, /operatorInspection\.operator\.description/);
  assert.doesNotMatch(page, /创建安全声明式复合算子|compositeDefinitionText|saveCompositeDefinition|expandAndAddComposite/);
  assert.doesNotMatch(backend, /createPipelineComposite|expandPipelineComposite/);
  assert.match(settingsPage, /operator\.annotation_policy\?\.mode/);
});

test('model directory and preferred download source are backed by the local service', () => {
  assert.match(backend, /localRequest<ApplicationSettings>\(base, '\/settings'/);
  assert.match(backend, /method: 'PATCH'/);
  assert.match(page, /backend\.updateApplicationSettings\(\{ model_weights_dir: value \}\)/);
  assert.match(settingsPage, /重启本地服务后生效；已有权重不会自动迁移/);
  assert.match(settingsPage, /LABELONE_MODEL_WEIGHTS_DIR/);
  assert.match(settingsPage, /ariaLabel="模型下载源"/);
  assert.match(settingsPage, /GitHub|ModelScope|Hugging Face/);
  assert.match(page, /updateApplicationSettings\(\{ model_download_source: modelDownloadSource \}\)/);
  assert.doesNotMatch(settingsPage, /X-AnyLabeling|X label/);
  assert.match(settingsPage, /模型选择、加载及权重导入统一在右侧推理面板完成/);
});

test('system settings configure credential-free proxy routing for local-service requests', () => {
  assert.match(settingsPage, /section === 'system'/);
  assert.match(settingsPage, /网络代理模式/);
  assert.match(settingsPage, /跟随系统 \/ 环境代理/);
  assert.match(settingsPage, /不使用代理 · 直连/);
  assert.match(settingsPage, /手动配置代理/);
  assert.match(settingsPage, /http:\/\/127\.0\.0\.1:7890/);
  assert.match(settingsPage, /代理地址不得包含用户名或密码/);
  assert.match(page, /updateApplicationSettings\(\{ network_proxy: networkProxyDraft \}\)/);
  assert.match(css, /\.proxy-settings-form\{/);
  assert.doesNotMatch(settingsPage, /type="password"|proxy_password/);
});

test('shortcut settings record keys, detect conflicts and pause workspace shortcuts', () => {
  assert.match(page, /const \[recordingShortcut, setRecordingShortcut\]/);
  assert.match(page, /findShortcutConflict\(shortcuts, action, binding\)/);
  assert.match(page, /if \(settingsOpen\) return;/);
  assert.match(settingsPage, /onKeyDown=\{\(event\) => onRecordShortcut\(definition\.id, event\)\}/);
  assert.match(settingsPage, /全部恢复默认/);
  assert.match(page, /labelone-global-settings-v1/);
  assert.match(page, /resolveShortcutAction\(event, shortcuts/);
});
