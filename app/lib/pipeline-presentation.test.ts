import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const insertPopover = readFileSync(new URL('../components/pipeline-insert-popover.tsx', import.meta.url), 'utf8');
const parameterControl = readFileSync(new URL('../components/pipeline-parameter-control.tsx', import.meta.url), 'utf8');

test('pipeline graph keeps source, transforms and displays in one vertical execution sequence', () => {
  const graph = page.match(/<div className="flow-canvas"[\s\S]*?<PipelineParameterEditor/)?.[0] ?? '';
  assert.match(graph, /visualization-node output-node/);
  assert.doesNotMatch(graph, /node-port|>IMG</);
  assert.doesNotMatch(graph, /add-visualization-tap|＋ 可视化/);
  assert.match(graph, /pipelineLinearNodes\.map\(\(node, sequenceIndex\)/);
  assert.match(graph, /node\.kind === 'source' \? 'source-node' : isVisualization \? 'visualization-node output-node' : 'operator-node'/);
  assert.match(graph, /isVisualization \? '显示上游节点的输出'/);
  assert.doesNotMatch(graph, /flow-visualization-branch|flow-output-stage/);
  assert.doesNotMatch(css, /\.flow-visualization-branch|\.flow-output-stage/);
  assert.doesNotMatch(graph, /<input value=\{String\(item\.parameters\.label/);
  assert.doesNotMatch(graph, /CustomSelect ariaLabel=\{`\$\{String\(item\.parameters\.label/);
  assert.doesNotMatch(page, /来源阶段|可视化来源阶段|retargetVisualization/);
  assert.match(page, /<PipelineParameterEditor node=\{selectedOperator\}/);
  assert.match(page, /className="operator-description"/);
});

test('every adjacent node pair owns one insertion gap and consecutive displays are disabled', () => {
  const graph = page.match(/<div className="flow-canvas"[\s\S]*?<PipelineParameterEditor/)?.[0] ?? '';
  const mainChain = graph.indexOf('{pipelineLinearNodes.map((node, sequenceIndex) => {');
  const insertion = graph.indexOf('flow-insert-connector');
  assert.ok(mainChain >= 0 && mainChain < insertion);
  assert.match(graph, /const gap = pipelineGaps\[sequenceIndex\]/);
  assert.match(graph, /\{gap && <div className=\{`flow-insert-connector/);
  assert.match(graph, /pipelineInsertGapKey === gap\.key/);
  assert.match(graph, /visualizationDisabledByNeighbor = Boolean\(gap && !gap\.visualizationTapAfterNodeId\)/);
  assert.match(graph, /相邻节点已有显示，不能连续添加显示/);
  assert.match(graph, /onAddOperator=\{\(kind\) => addNode\(kind, gap\)\}/);
  assert.match(graph, /if \(gap\.visualizationTapAfterNodeId\) addVisualization\(gap\.visualizationTapAfterNodeId\)/);
  assert.doesNotMatch(css, /\.flow-visualization-branch/);
});

test('operator search exists only in connection insertion menus', () => {
  assert.doesNotMatch(page, /className="operator-palette"/);
  assert.doesNotMatch(css, /\.operator-palette(?:\s|\{|\.|:)/);
  assert.doesNotMatch(page, /className="operator-browser"/);
  assert.match(page, /<PipelineInsertPopover[^>]*anchor=\{pipelineInsertAnchorRef\.current\}/);
  assert.match(insertPopover, /className="flow-insert-menu flow-insert-popover"/);
  assert.match(insertPopover, /<input autoFocus type="search"/);
  assert.match(insertPopover, /className=\{`visualization-option \$\{visualizationDisabled \? 'blocked-option' : ''\}`\}/);
  assert.match(page, /onAddVisualization=\{\(\) => \{ if \(gap\.visualizationTapAfterNodeId\)/);
});

test('operator insertion popover portals above clipped panels and follows its anchor', () => {
  assert.match(insertPopover, /createPortal\([\s\S]*fullscreenPortalTarget\(document\)\)/);
  assert.match(insertPopover, /anchor\.getBoundingClientRect\(\)/);
  assert.match(insertPopover, /window\.addEventListener\('scroll', schedulePosition, true\)/);
  assert.match(insertPopover, /window\.addEventListener\('resize', schedulePosition\)/);
  assert.match(insertPopover, /window\.visualViewport\?\.addEventListener\('resize', schedulePosition\)/);
  assert.match(insertPopover, /event\.key === 'Escape'/);
  assert.match(insertPopover, /anchor\?\.focus\(\{ preventScroll: true \}\)/);
  assert.match(page, /aria-haspopup="dialog"/);
  assert.match(css, /\.flow-insert-menu\{position:fixed;z-index:140/);
  assert.doesNotMatch(css, /\.flow-insert-menu\{[^}]*(?:top:24px|left:6px|right:6px)/);
});

test('legacy visualization tap card styles and inline editor are removed', () => {
  assert.doesNotMatch(page, /className={`visualization-tap/);
  assert.doesNotMatch(css, /\.visualization-tap(?:\{|\.|:|>)/);
  assert.doesNotMatch(page, /className="visualization-taps"/);
  assert.doesNotMatch(css, /\.visualization-taps(?:\{|\.|:|>)/);
  assert.doesNotMatch(css, /\.add-visualization-tap(?:\{|\.|:|>)/);
  assert.doesNotMatch(css, /\.node-port(?:\{|\.|:|>)/);
  assert.match(css, /\.visualization-node\{/);
  assert.doesNotMatch(css, /\.visualization-node\{[^}]*(?:width|margin|padding|position|display|grid-template)/);
});

test('registry titles are the only displayed operator names', () => {
  assert.match(page, /kind: 'source', name: '原图像'/);
  assert.match(page, /name: contract\.title/);
  assert.match(page, /name: contract\?\.title \?\? operator\.name/);
  assert.match(page, /name: contract\.title, color:/);
  assert.match(page, /const visualizationName = visualizationContract\?\.title \?\? '显示'/);
  assert.doesNotMatch(insertPopover, /<strong>可视化<\/strong>/);
});

test('operator details disclose verified image and annotation spatial behavior', () => {
  assert.match(page, /contract\?\.description \?\? '该算子暂未提供说明。'/);
  assert.match(page, /空间：\{String\(contract\?\.annotation_policy\?\.spatial_behavior \?\? 'unknown'\)\}/);
  assert.match(page, /contract\?\.annotation_policy\?\.synchronized === true \? '标签同步' : '未验证'/);
  assert.match(page, /标注：\{String\(contract\?\.annotation_policy\?\.mode \?\? 'unknown'\)\}/);
  assert.match(parameterControl, /className="pipeline-parameter-heading"/);
  assert.match(parameterControl, /effectiveSchema\.description \?\? '该参数暂未提供说明。'/);
  assert.match(parameterControl, /<strong>\{label\}<\/strong><small>/);
  assert.match(css, /\.pipeline-parameter-heading small\{/);
});

test('validated current-image preview runs automatically before background neighbor prefetch', () => {
  assert.match(page, /const requestPipelineValidation = useCallback/);
  assert.match(page, /validatePipeline\(\{[\s\S]*?nodes: pipelineRequestNodes,[\s\S]*?pipelineValidationWidth !== undefined && pipelineValidationHeight !== undefined/);
  assert.match(page, /validatedPipelineKey !== currentPipelineValidationKey \|\| !pipelineValidationState\.data\?\.valid/);
  assert.match(page, /previewPipeline\(\{ dataset_id: dataset\.id, asset_id: currentFile\.id, priority: 'interactive'/);
  assert.match(page, /prefetchPipelinePreview\(\{ dataset_id: dataset\.id!, asset_id: assetId, priority: 'background'/);
  assert.match(page, /preferred_asset_ids: preferredPipelineAssetIds/);
  assert.match(page, /takeCachedPipelinePreview\(pipelinePreviewCacheRef\.current, key\)/);
  assert.match(page, /storeCachedPipelinePreview\(pipelinePreviewCacheRef\.current, key, result\)/);
  assert.doesNotMatch(page, /\}, 380\)/);
});

test('flow canvas keeps only a bottom-right validation light without manual run controls', () => {
  assert.match(page, /formatPipelineTiming\(nodeTiming\?\.milliseconds, nodeTiming\?\.samples\)/);
  assert.match(page, /nodeTimingText \?\? '计算中'/);
  const flow = page.match(/<div className="flow-canvas"[\s\S]*?<PipelineParameterEditor/)?.[0] ?? '';
  assert.match(flow, /className={`pipeline-validation-indicator/);
  assert.match(flow, /right:7px;bottom:7px|data-tooltip-title="处理流校验"/);
  assert.doesNotMatch(flow, /pipeline-canvas-toolbar|pipeline-validation-compact|pipeline-validation-retry|run-button compact|重新校验处理流|▶ 运行/);
  assert.doesNotMatch(flow, /pipeline-registry-tray|pipeline-registry-trigger|pipeline-registry-popover|>REG</);
  assert.match(flow, /className="flow-sequence"/);
  assert.doesNotMatch(page, /pipeline-run-dock|registry-status|立即刷新当前图|const runPipeline/);
  assert.doesNotMatch(css, /pipeline-run-dock|pipeline-validation-state|registry-status|pipeline-canvas-toolbar|pipeline-validation-compact|pipeline-validation-retry/);
  assert.doesNotMatch(page, /pipelineRegistryTrayOpen|pipelineRegistryTrayRef|pipelineRegistryTriggerRef|dismissRegistryTray|closeRegistryTray/);
  assert.match(css, /\.pipeline-panel\{height:100%;min-height:0;display:flex;flex-direction:column;overflow:hidden\}/);
  assert.match(css, /\.pipeline-panel>\.flow-canvas\{position:relative;min-height:120px;flex:1;display:flex;flex-direction:column;overflow:hidden/);
  assert.match(css, /\.flow-sequence\{min-height:0;flex:1;overflow:auto;padding:8px 9px 32px/);
  assert.match(css, /\.pipeline-validation-indicator\{position:absolute;z-index:12;right:7px;bottom:7px;width:18px;height:18px/);
  assert.doesNotMatch(css, /pipeline-registry-tray|pipeline-registry-trigger|pipeline-registry-popover/);
});
