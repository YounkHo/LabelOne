import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const contracts = readFileSync(new URL('./contracts.ts', import.meta.url), 'utf8');

test('confirmed Agent UI actions reuse existing application functions', () => {
  assert.match(page, /action === 'ui\.open_dataset'[\s\S]*?pickImageDataset\(\)/);
  assert.match(page, /action === 'ui\.import_operator'[\s\S]*?setSettingsSection\('operators'\)[\s\S]*?operatorPackageInputRef\.current\?\.click\(\)/);
  assert.match(page, /action === 'ui\.open_models'[\s\S]*?setRightTab\('inference'\)[\s\S]*?setModelPickerOpen\(true\)/);
  assert.match(page, /受控数据工作流助手/);
  assert.match(page, /界面动作和任务必须确认/);
});

test('pipeline drafts only load registered transform nodes into editor state', () => {
  assert.match(page, /action === 'pipeline\.draft'/);
  assert.match(page, /\['source', 'output', 'visualize'\]\.includes\(kind\)/);
  assert.match(page, /pipelineContracts\.find\(\(candidate\) => candidate\.kind === kind\)/);
  assert.match(page, /setNodes\(nextNodes\)/);
  assert.match(page, /const nextVisualizations = normalizeVisualizationTaps\(nextNodes, visualizations\)/);
  assert.match(page, /setSinglePipelineSource\(finalPipelineVisualizationId\(nextNodes, nextVisualizations\)\)/);
  assert.match(page, /setRightTab\('pipeline'\)/);
});

test('proposal results expose only structured app actions', () => {
  assert.match(contracts, /result\?: Record<string, unknown> \| null/);
  assert.doesNotMatch(page, /shell\.exec|terminal\.exec|source\.write|code\.execute/);
});

test('Agent is fully gated by backend readiness and exposes a configuration path', () => {
  assert.match(page, /const agentBackendReady = backend\.mode === 'online' && backend\.agentStatus\.data\?\.state === 'ready'/);
  assert.match(page, /Agent 后端未配置/);
  assert.match(page, /配置 Agent 后端/);
  assert.match(page, /setSettingsSection\('ai'\); setSettingsOpen\(true\)/);
  assert.match(page, /未配置时不会发送请求，也不会执行任何 Agent 工具/);
  assert.match(contracts, /state: 'ready' \| 'unconfigured'/);
});

test('quick checks send explicit allowlisted tool calls instead of ambiguous phrases', () => {
  assert.match(page, /tool: 'dataset\.stats', arguments: \{\}/);
  assert.match(page, /tool: 'dataset\.search', arguments: \{ annotated: false, limit: 20 \}/);
  assert.match(page, /tool: 'dataset\.distribution', arguments: \{ top_n: 20 \}/);
  assert.match(page, /tool: 'annotation\.qa', arguments: \{\}/);
  assert.match(contracts, /tool_call\?: \{/);
  assert.match(contracts, /tool_results: AgentToolResult\[\]/);
});

test('Agent workspace explains capability and confirmation boundaries', () => {
  assert.match(page, /数据检查/);
  assert.match(page, /界面与草案/);
  assert.match(page, /后台执行/);
  assert.match(page, /只读工具直接执行；界面动作和任务必须确认/);
  assert.match(page, /proposal\.executed \? <div className="proposal-executed"/);
  assert.match(page, /该提案属于之前的项目或图片/);
  assert.doesNotMatch(page, /对话助手|云端自然语言对话|本地确定性工具/);
});
