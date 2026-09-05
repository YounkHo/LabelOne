import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const agentService = readFileSync(new URL('../../server/src/labelone/agent/service.py', import.meta.url), 'utf8');

test('background jobs no longer take a right-sidebar tab', () => {
  assert.match(page, /type RightTab = 'layers' \| 'pipeline' \| 'inference' \| 'agent'/);
  assert.match(page, /\[\['layers', '对象'\], \['pipeline', '处理流'\], \['inference', '推理'\], \['agent', 'Agent'\]\]/);
  assert.doesNotMatch(page, /rightTab === 'tasks' && <|setRightTab\('tasks'\)|\['tasks'/);
  assert.doesNotMatch(page, /任务中心/);
  assert.match(page, /state\.rightTab === 'tasks'\) setRightTab\('layers'\)/);
  assert.match(css, /\.right-tabs\{height:45px;display:grid;grid-template-columns:repeat\(4,1fr\)/);
  assert.doesNotMatch(css, /\.task-center|\.task-intro|\.task-list|\.task-card|\.empty-tasks/);
});

test('background activity lives in the top-right actions as a browser-style progress control', () => {
  assert.match(page, /const backgroundTaskControl = taskIconVisible \? <section ref=\{taskActivityRef\} className={`background-task-control/);
  const taskControl = page.indexOf('{backgroundTaskControl}');
  const settings = page.indexOf('ref={settingsButtonRef}', taskControl);
  const fullscreen = page.indexOf('aria-label={isFullscreen', settings);
  assert.ok(taskControl >= 0 && taskControl < settings && settings < fullscreen);
  assert.match(page, /className={`top-action-button background-task-button/);
  assert.match(page, /<BackgroundTasksIcon attention=\{attentionTaskCount > 0\} complete=\{taskIconComplete\}/);
  assert.match(page, /className={`background-task-progress \$\{latestTaskDownloadView\?\.percent === null \? 'indeterminate' : ''\}`/);
  assert.match(page, /aria-haspopup="dialog" aria-expanded=\{taskStreamVisible\} aria-pressed=\{taskStreamOpen\}/);
  assert.match(page, /className="background-activity" role="dialog" aria-label="后台任务"/);
  assert.match(page, /className="activity-stream" role="region" aria-label="后台活动流"/);
  assert.match(page, /className="activity-row-main" aria-pressed=\{selected\}/);
  assert.match(page, /controlTaskGroup\(logicalJobIds, 'pause'\)/);
  assert.match(page, /controlTaskGroup\(logicalJobIds, 'resume'\)/);
  assert.match(page, /controlTaskGroup\(logicalJobIds, 'cancel'\)/);
  assert.match(page, /backend\.jobItems\.phase === 'ready'/);
  assert.doesNotMatch(page, /right-sidebar[^\n]+has-activity|activity-open|activity-collapsed/);
  assert.doesNotMatch(css, /\.right-sidebar\.has-activity|\.activity-dock-toggle/);
  assert.match(css, /\.background-task-control\{position:relative;flex:none\}/);
  assert.match(css, /\.background-activity\{position:absolute;z-index:60;right:0;top:38px/);
  assert.match(css, /\.activity-stream\{[^}]*overflow:auto/);
});

test('hover and focus preview tasks while click pins the popover without stealing the current tab', () => {
  assert.doesNotMatch(page, /previousActiveTaskCountRef/);
  assert.doesNotMatch(page, /if \(activeTaskCount > .*setTaskStreamOpen\(true\)/);
  assert.doesNotMatch(page, /setTaskStreamOpen\(true\)/);
  assert.match(page, /onClick=\{\(\) => setTaskStreamOpen\(\(open\) => !open\)\}/);
  assert.match(page, /const taskStreamVisible = taskStreamOpen \|\| taskStreamHovered \|\| taskStreamFocused/);
  assert.match(page, /onMouseEnter=\{\(\) => setTaskStreamHovered\(true\)\}/);
  assert.match(page, /onMouseLeave=\{\(\) => setTaskStreamHovered\(false\)\}/);
  assert.match(page, /onFocusCapture=\{\(\) => setTaskStreamFocused\(true\)\}/);
  assert.match(page, /void backend\.refreshJobs\(\)\.catch\(\(\) => undefined\)/);
  assert.match(page, /可在右上角查看进度/);
  assert.match(agentService, /后续进度可在右上角后台任务中查看/);
});

test('recent task windows, local clearing and actionable attention states stay reachable', () => {
  assert.match(page, /const taskIconVisible = taskStreamJobs\.length > 0 \|\| taskStreamVisible/);
  assert.match(page, /const taskIconComplete = taskStreamJobs\.length > 0 && activeTaskCount === 0 && attentionTaskCount === 0/);
  assert.match(page, /filterBackgroundTaskHistory\(backendJobs, dismissedTaskVersions, taskHistoryHours\)/);
  assert.match(page, /BACKGROUND_TASK_HISTORY_HOURS\.map/);
  assert.match(page, />清除\{clearableTaskIds\.length/);
  assert.match(page, />恢复<\/button>/);
  assert.match(page, /重试失败项/);
  assert.match(page, /忽略提醒/);
  assert.match(page, /服务端任务记录仍保留/);
  assert.match(page, /className={`background-task-badge \$\{attentionTaskCount > 0 \? 'attention' : ''\}`/);
  assert.doesNotMatch(page, /<circle className="task-progress/);
  assert.match(css, /\.background-task-progress\{position:absolute/);
  assert.match(css, /\.activity-history-toolbar\{/);
  assert.match(css, /\.activity-attention-actions\{/);
  assert.doesNotMatch(page, /activity-summary-progress/);
  assert.match(page, /const taskPopoverSummary =/);
  assert.match(css, /Background tasks: one compact hierarchy/);
});

test('the activity popover closes on outside click or Escape', () => {
  assert.match(page, /const taskActivityRef = useRef<HTMLElement>\(null\)/);
  assert.match(page, /const taskActivityButtonRef = useRef<HTMLButtonElement>\(null\)/);
  assert.match(page, /!taskActivityRef\.current\?\.contains\(target\)/);
  assert.match(page, /event\.key !== 'Escape'/);
  assert.match(page, /taskActivityButtonRef\.current\?\.focus\(\)/);
  assert.match(page, /document\.addEventListener\('pointerdown', closeOutside\)/);
  assert.match(page, /document\.addEventListener\('keydown', closeWithEscape\)/);
  assert.match(page, /setTaskStreamOpen\(false\);\s+setTaskStreamHovered\(false\);\s+setTaskStreamFocused\(false\);\s+setSettingsSection/);
});
