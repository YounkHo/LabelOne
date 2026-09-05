import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const backend = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');

test('two to four pipeline displays fill the canvas and share normalized view without transforming the grid', () => {
  assert.match(page, /showingMultiplePipelineViews = showPipelineViewControls && pipelineHasDisplayablePane && effectiveVisualizationDisplayMode !== 'source'/);
  assert.match(page, /className={`pipeline-preview-surface mode-\$\{effectiveVisualizationDisplayMode\} count-\$\{pipelineCanvasItems\.length\}`}/);
  assert.doesNotMatch(page, /className={`pipeline-preview-surface[^\n]*style=\{\{ transform:/);
  assert.match(page, /className="pipeline-preview-pane-content" style=\{\{ transform: pipelineSharedPaneTransform \}\}/);
  assert.match(css, /\.pipeline-preview-surface\{position:absolute;z-index:1;inset:0;width:100%;height:100%;margin:0;overflow:hidden;border:0;border-radius:0/);
  assert.match(css, /\.pipeline-preview-grid\{display:grid;width:100%;height:100%;min-width:0;min-height:0;gap:1px;padding:0/);
  assert.match(css, /\.pipeline-preview-surface\.count-2 \.pipeline-preview-grid\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\);grid-template-rows:minmax\(0,1fr\)\}/);
  assert.match(css, /\.pipeline-preview-surface\.mode-split\.count-3>\.pipeline-preview-grid\{grid-template-columns:repeat\(3,minmax\(0,1fr\)\);grid-template-rows:minmax\(0,1fr\)\}/);
  assert.match(css, /\.pipeline-preview-surface\.mode-split\.count-4>\.pipeline-preview-grid\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\);grid-template-rows:repeat\(2,minmax\(0,1fr\)\)\}/);
  assert.match(css, /\.pipeline-preview-surface\.mode-split \.pipeline-preview-pane\{box-sizing:border-box;width:100%;height:100%/);
  assert.match(css, /\.pipeline-preview-label\{position:absolute;z-index:9;/);
  assert.doesNotMatch(page, /data-pipeline-label=\{item\.label\} style=\{\{ aspectRatio:/);
  assert.doesNotMatch(css, /\.pipeline-preview-surface\{[^}]*pointer-events:none/);
  assert.match(page, /pipelineDisplayItems\.map\(\(item, index\)/);
  assert.match(page, /onError=\{\(\) => \{ if \(item\.url\) handlePipelineImageError\(item\.url\); \}\}/);
  assert.match(page, /crossOrigin="anonymous"/);
});

test('pointer coordinates and pixel sampling use the contained image rect in every pane', () => {
  assert.match(page, /target\?\.closest<HTMLElement>\('\[data-pipeline-preview-pane\]'\)/);
  assert.match(page, /data-pipeline-visualization-id=\{item\.visualization_id\} data-pipeline-width=\{item\.width\} data-pipeline-height=\{item\.height\} data-pipeline-label=\{item\.label\}/);
  assert.match(page, /activePreviewPane \? pipelinePreviewContainedRect\(activePreviewPane\)/);
  assert.match(page, /readDisplayedPixel\(latest\.clientX, latest\.clientY, latest\.pane\)/);
  assert.match(page, /const surface = activePane\?\.isConnected \? activePane : stageRef\.current \?\? imageRef\.current/);
  assert.match(page, /containedPipelineImageRect\(imageBounds, image\.naturalWidth, image\.naturalHeight\)/);
  assert.match(page, /createPipelineSharedCursor\(/);
  assert.match(page, /pipelineCoordinateMappingForItem\(activePipelineItem\)/);
  assert.match(page, /pipelineSharedCursorPointForPane\(pipelineSharedCursor, item\.visualization_id/);
  assert.match(page, /pixelWidth,\s*pixelHeight,\s*\)/);
  assert.match(page, /renderPipelineSharedCrosshair = \(item: Pick<PipelineVisualizationResult, 'visualization_id' \| 'width' \| 'height' \| 'coordinate_mapping'>, screenAligned = false\)/);
  assert.doesNotMatch(page, /normalizedPipelineCursor/);
  assert.match(page, /preserveAspectRatio="xMidYMid meet"/);
  assert.match(page, /stageRef\.current \?\? imageRef\.current/);
  assert.match(page, /pipelineSharedCursor\.label/);
  assert.match(page, /!showingPipelinePaneViews && <div ref=\{canvasCrosshairRef\}/);
});

test('overlay requires compatible dimensions and keeps one layer visible', () => {
  assert.match(page, /pipelineDisplaySlotsReady\s+\? visualizationOverlayCompatibility\(pipelineDisplayItems\)/);
  assert.match(page, /role="radio" aria-checked=\{effectiveVisualizationDisplayMode === 'split'\}/);
  assert.match(page, /role="radio" aria-checked=\{effectiveVisualizationDisplayMode === 'overlay'\}/);
  assert.match(page, /disabled=\{!pipelineOverlayCompatibility\.allowed && effectiveVisualizationDisplayMode !== 'overlay'\}/);
  assert.match(page, /canHidePipelineLayer\(item\.visualization_id, visualizationLayerState\)/);
  assert.match(page, /至少保留一个可见层/);
  assert.match(page, /coordinate_space_id: item\?\.coordinate_mapping\?\.coordinate_space_id/);
  assert.match(page, /className="visualization-alpha-mixer" role="group"/);
  assert.match(page, /pipelineDisplayItems\.slice\(\)\.reverse\(\)\.map/);
  assert.match(page, /aria-labelledby=\{labelId\} aria-valuetext=\{`\$\{layer\.opacity\}%`\}/);
  assert.doesNotMatch(page, /<label key=\{item\.visualization_id\}><button role="switch"/);
  assert.match(page, /setVisualizationLayerOpacity\(item\.visualization_id, Number\(event\.currentTarget\.value\)\)/);
  assert.match(css, /\.pipeline-preview-surface\.mode-overlay\{width:100%;height:100%\}/);
  assert.match(css, /\.pipeline-overlay-stack\{position:absolute;inset:0;width:100%;height:100%;overflow:hidden;border:0;border-radius:0/);
  assert.match(css, /\.pipeline-panel>\.visualization-layer-controls\{flex:0 0 auto;max-height:none;overflow:visible\}/);
  assert.match(css, /\.visualization-alpha-row\{box-sizing:border-box;display:grid;grid-template-columns:22px 62px minmax\(92px,1fr\) 28px/);
  assert.match(css, /\.visualization-alpha-row>input\[type=range\]\{width:calc\(100% - 12px\);min-width:0;height:20px;margin:0 6px/);
});

test('parameter changes keep the last successful panes while the next preview is validated and rendered', () => {
  assert.match(page, /const pipelineBelongsToCurrentAsset = Boolean\(/);
  assert.match(page, /const pipelineMatchesCurrent = pipelineBelongsToCurrentAsset\s+&& completedPipelineSignature === currentPipelineExecutionSignature/);
  assert.match(page, /const pipelinePreviewItems = useMemo\(\(\) => pipelineBelongsToCurrentAsset/);
  assert.match(page, /const pipelinePreviewDirty = pipelineBelongsToCurrentAsset && !pipelineMatchesCurrent/);
  assert.match(page, /参数已更新 · 保留上一版分屏，正在重新计算/);
  assert.match(backend, /setPipeline\(\(old\) => \(\{ phase: 'error', data: old\.data, stale: Boolean\(old\.data\), error: apiError\(error\) \}\)\)/);
});

test('single-source, split, and overlay modes preserve stable slots while switching assets', () => {
  assert.match(page, /useState<PipelineDisplayMode>\('source'\)/);
  assert.match(page, /useState\(\(\) => finalPipelineVisualizationId\(initialNodes, initialVisualizations\)\)/);
  assert.match(page, /if \(enabled\) \{\s+setSinglePipelineSource\(finalPipelineSource\);\s+setVisualizationDisplayMode\('source'\)/);
  assert.match(page, /setSinglePipelineSource\(finalPipelineSource\)/);
  assert.match(page, /className="pipeline-single-source-select" ariaLabel="单画面来源"/);
  assert.match(page, /\{ value: 'source', label: '原图' \}/);
  assert.match(page, /value: item\.visualization_id, label: `D\$\{index \+ 1\}/);
  assert.match(page, /aria-pressed=\{effectiveVisualizationDisplayMode === 'split'\}/);
  assert.match(page, /aria-pressed=\{effectiveVisualizationDisplayMode === 'overlay'\}/);
  assert.match(page, /setSinglePipelineSource\(value\); setVisualizationDisplayMode\('source'\)/);
  assert.match(page, /const canvasPipelineItem = effectiveVisualizationDisplayMode === 'source'[\s\S]*?\? selectedSinglePipelineItem[\s\S]*?: pipelineDisplayItems\[0\]/);
  assert.match(page, /pipelineImageUrl: activeStandaloneRasterUrl \?\? effectivePipelineImageUrl/);
  assert.match(page, /const showingPipelinePaneViews = showingMultiplePipelineViews/);
  assert.match(page, /singlePipelineSource === 'source' \? '单画面 · 原图'/);
  assert.match(page, /stablePipelineDisplaySlots\(visualizations, pipelinePreviewItems, MAX_PIPELINE_VISUALIZATIONS\)/);
  assert.match(page, /const pipelinePreviewItems = useMemo\(\(\) => pipelineBelongsToCurrentAsset/);
  assert.match(page, /pipelineDisplaySlots\.every\(\(slot\) => slot\.result !== null\)/);
  assert.match(page, /resolvePipelineDisplayMode\(\s*visualizationDisplayMode,\s*pipelineDisplaySlots\.length,\s*pipelineDisplaySlotsReady,\s*pipelineOverlayCompatibility\.allowed/);
  assert.match(page, /effectiveVisualizationDisplayMode !== visualizationDisplayMode\) setVisualizationDisplayMode\(effectiveVisualizationDisplayMode\)/);
  assert.match(page, /item\.result \? '该处理流图像暂时不可读取' : '正在计算此显示…'/);
  assert.match(page, /!pipelineDisplaySlotsReady && <div className="image-load-empty pipeline-overlay-loading" role="status">正在加载当前图的 \{pipelineDisplaySlots\.length\} 个显示结果<\/div>/);
  assert.match(page, /叠加 · \{pipelineDisplaySlots\.length\} 层\{pipelineDisplaySlotsReady \? '' : ' · 加载中'\}/);
});

test('every split and overlay result renders its own read-only transformed annotation document', () => {
  assert.match(page, /const shapes = item\.annotation_document\.shapes \?\? \[\]/);
  assert.match(page, /className=\{`pipeline-preview-annotation-layer \$\{screenAligned \? 'screen-aligned' : ''\}`\} viewBox=\{`0 0 \$\{item\.width\} \$\{item\.height\}`\} preserveAspectRatio="xMidYMid meet"/);
  assert.match(page, /renderPipelinePreviewAnnotationLayer\(item\.result, true\)/);
  assert.match(page, /renderPipelinePreviewAnnotationLayer\(item\.result\)/);
  assert.match(css, /\.pipeline-preview-annotation-layer\{position:absolute;z-index:6;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none/);
  assert.match(css, /\.pipeline-preview-annotation-layer \*\{pointer-events:none!important\}/);
});

test('split panes use screen-space pixel canvases with label collision and synchronized crosshairs', () => {
  assert.match(page, /pipelinePaneMetrics\(paneWidth, paneHeight, item\.width, item\.height, view, stageWidth, stageHeight, pixelGridEnabled\)/);
  assert.match(page, /const labelLayouts = canvasLabelLayouts\(/);
  assert.match(page, /metrics\.imageUnitsPerScreenPixel/);
  assert.match(page, /metrics\.visibleBounds \?\? undefined/);
  assert.match(page, /function PipelinePixelGridCanvas\(/);
  assert.match(page, /const dpr = Math\.max\(1, window\.devicePixelRatio \|\| 1\)/);
  assert.match(page, /snapPipelineGridCoordinate\(metrics\.display\.left \+ index \* stepX, dpr\)/);
  assert.match(page, /context\.lineWidth = 1 \/ dpr/);
  assert.match(page, /new ResizeObserver\(schedule\)/);
  assert.match(page, /<PipelinePixelGridCanvas imageWidth=\{item\.width\} imageHeight=\{item\.height\} view=\{view\} referenceWidth=\{stageRef\.current\?\.clientWidth \?\? 840\} referenceHeight=\{stageRef\.current\?\.clientHeight \?\? 592\} enabled=\{showPixel\} \/>/);
  assert.match(page, /className="horizontal" x1="0" x2=\{item\.width\}/);
  assert.match(page, /className="vertical" x1=\{point\.x\} x2=\{point\.x\}/);
  assert.match(css, /\.pipeline-preview-pane\.source-pixels-visible \.pipeline-preview-image\{image-rendering:pixelated\}/);
  assert.match(css, /\.pipeline-pixel-grid-canvas\{position:absolute;z-index:5;inset:0;width:100%;height:100%;pointer-events:none\}/);
  assert.match(css, /\.pipeline-shared-cursor line\{stroke-dasharray:4 5;stroke-opacity:\.62\}/);
});

test('existing canvas controls remain above the full-canvas pipeline surface', () => {
  assert.match(css, /\.pipeline-preview-surface\{position:absolute;z-index:1;/);
  assert.match(css, /\.pipeline-view-controls\{position:absolute;z-index:17;/);
  assert.match(css, /\.pipeline-stale-notice\{position:absolute;z-index:16;/);
  assert.match(css, /\.canvas-page-navigation\{position:absolute;z-index:14;/);
  assert.match(css, /\.navigator\{position:absolute;z-index:4;/);
});
