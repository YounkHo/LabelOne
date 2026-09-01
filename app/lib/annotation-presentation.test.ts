import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');
const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const virtualFileList = readFileSync(new URL('../components/virtual-file-list.tsx', import.meta.url), 'utf8');

test('canvas and list colors come from annotation categories rather than shape types', () => {
  assert.match(page, /const categoryStyle = annotationCategoryStyle\(shape\.label, annotationCategoryColorOverrides\)/);
  assert.match(page, /className=\{typeClass\} style=\{categoryStyle\}/);
  assert.match(page, /className=\{`annotation-object-row[\s\S]*?style=\{annotationCategoryStyle\(shape\.label, annotationCategoryColorOverrides\)\}/);
  assert.doesNotMatch(page, /renderDemoShapeGeometry|renderDemoCanvasLabel|demoAnnotationShapes/);
  for (const shapeType of ['rectangle', 'rotation', 'polygon', 'point', 'line', 'circle']) {
    assert.doesNotMatch(css, new RegExp(`\\.shape-${shapeType}\\{--shape-color:`));
  }
});

test('single-object visibility keeps every object row available for recovery', () => {
  assert.ok(page.indexOf('className="side-section annotation-category-panel"') < page.indexOf('className="side-section annotation-object-panel"'));
  assert.match(page, /<div className="annotation-category-icon">[\s\S]*?<label className="annotation-category-color"[\s\S]*?<button className=\{`annotation-category-visibility/);
  assert.match(page, /<input type="color"[^>]*value=\{categoryColors\.stroke\}[^>]*aria-label=\{`\u8bbe\u7f6e\u7c7b\u522b \$\{category\} \u7684\u989c\u8272`\}/);
  assert.match(page, /className=\{`annotation-category-visibility/);
  assert.match(page, /role="checkbox" aria-checked=\{categoryVisibilityState\}/);
  assert.match(page, /deleteAnnotationCategory\(category\)/);
  assert.match(page, /\u5220\u9664\u5f53\u524d\u56fe\u7c7b\u522b \$\{category\} \u53ca\u5176 \$\{categoryIndexes\.length\} \u4e2a\u6807\u6ce8\u6846/);
  assert.doesNotMatch(page, /annotationCategoryFilter|activeAnnotationCategoryFilter|filteredAnnotation|className="annotation-category-filter"|ariaLabel="筛选标注类别"/);
  assert.match(page, /<button[^>]*className="annotation-category-main"[\s\S]*?<strong>\{category\}<\/strong><small>\{categoryIndexes\.length\} 个框<\/small><\/button>/);
  assert.doesNotMatch(page, /categoryVisibleCount\} 可见/);
  assert.doesNotMatch(page, /if \(hiddenShapeIndexes\.has\(index\)\) return null/);
  assert.match(page, /objectListShapes\.length \? <div ref=\{annotationObjectListRef\} className=\{`annotation-object-list/);
  assert.match(page, /const objectListShapes = annotationObjects/);
  assert.match(page, /aria-label="当前图全部标注框"/);
  assert.match(page, /const visible = !hiddenShapeIndexes\.has\(index\)/);
  assert.match(page, /visible \? '' : 'hidden-object'/);
  assert.match(page, /aria-checked=\{visible\}/);
  assert.match(page, /onClick=\{\(\) => changeShapeVisibility\(index, !visible\)\}/);
  assert.match(page, /changeAnnotationIndexesVisibility\(annotationObjectIndexes, true\)/);
  assert.match(page, /changeAnnotationIndexesVisibility\(annotationObjectIndexes, false\)/);
  assert.match(page, /if \(!visible\) setSelectedShapeIndex/);
  assert.doesNotMatch(page, /标注框均已隐藏/);
  assert.match(css, /\.annotation-category-list\{max-height:min\(168px,26vh\);overflow:auto/);
  assert.match(css, /\.annotation-object-list\{border-color:#334255;background:#0d151f\}/);
  assert.match(css, /\.annotation-object-row\{background:#101923/);
  assert.match(css, /\.annotation-category-row\{grid-template-columns:26px minmax\(0,1fr\) 30px\}/);
  assert.match(css, /\.annotation-category-icon,\.annotation-object-icon\{position:relative;width:22px;height:22px\}/);
  assert.match(css, /\.annotation-object-shape\{width:22px;height:22px;display:grid;place-items:center/);
  assert.match(css, /\.annotation-category-icon \.annotation-category-visibility,\.annotation-object-icon \.annotation-object-visibility\{position:absolute;z-index:4;right:-3px;bottom:-3px;width:14px;height:14px/);
  assert.match(css, /\.annotation-category-icon \.visibility-eye-icon,\.annotation-object-icon \.visibility-eye-icon\{position:static;width:11px;height:11px/);
  assert.match(css, /\.annotation-category-row\.hidden-category \.annotation-category-color\{opacity:\.34;filter:grayscale\(1\) saturate\(0\)\}/);
  assert.match(css, /\.annotation-category-icon \.annotation-category-visibility:not\(\.on\),\.annotation-object-icon \.annotation-object-visibility:not\(\.on\)\{opacity:1;border-color:#748091;color:#9aa6b4\}/);
  assert.match(css, /\.annotation-object-row\.hidden-object \.annotation-object-main\{opacity:\.64;filter:grayscale\(1\) saturate\(0\)\}/);
  assert.match(css, /\.annotation-object-row\.hidden-object \.annotation-object-shape\{opacity:\.42;filter:grayscale\(1\) saturate\(0\)/);
  assert.match(css, /\.annotation-object-row\.hidden-object \.annotation-object-main strong\{color:#8794a4;text-decoration:none\}/);
  assert.match(css, /\.annotation-object-row\.hidden-object \.annotation-object-visibility\{opacity:1;color:#9aa6b4\}/);
});

test('annotation object rows reuse the category composite visibility icon in one polished line', () => {
  assert.match(page, /className="annotation-object-main" aria-label=\{`\$\{label\}，标注 \$\{index \+ 1\}，\$\{shapeTypeLabels\[shape\.shape_type\]/);
  assert.match(page, /<div className="annotation-object-icon">[\s\S]*?className=\{`annotation-object-shape[\s\S]*?className=\{`annotation-object-visibility/);
  assert.match(page, /<small className="annotation-object-index" aria-label=\{`标注编号 \$\{index \+ 1\}`\}>#[\s\S]*?<\/small><strong title=\{label\}>\{label\}<\/strong>/);
  assert.match(page, /className=\{`annotation-object-visibility \$\{visible \? 'on' : ''\}`\}/);
  assert.doesNotMatch(page, /#\{String\(index \+ 1\)\.padStart\(2, '0'\)\} · \{shapeTypeLabels\[shape\.shape_type\]/);
  assert.match(page, /className="shape-delete-button"[^>]+aria-label=\{`删除标注框[\s\S]*?>×<\/button>/);
  assert.match(css, /\.annotation-object-row\{grid-template-columns:26px minmax\(0,1fr\) 22px;gap:5px\}/);
  assert.match(css, /\.annotation-object-row:not\(:last-child\)::after\{content:'';position:absolute;left:36px;right:5px;bottom:0;height:1px;background:#202c3a;pointer-events:none\}/);
  assert.match(css, /\.annotation-object-main\{height:28px;display:grid;grid-template-columns:29px minmax\(0,1fr\)/);
  assert.match(css, /\.annotation-object-row\.selected\{background:color-mix\(in srgb,var\(--shape-color/);
  assert.match(css, /\.shape-delete-button\{[^}]*color:#6b7481;font-size:14px;line-height:1;opacity:0;pointer-events:none/);
  assert.match(css, /\.annotation-object-row\.selected \.shape-delete-button\{opacity:\.55\}/);
  assert.match(css, /\.annotation-object-row:hover \.shape-delete-button:not\(:disabled\)\{opacity:\.68;pointer-events:auto\}/);
});

test('category and object names expose inline double-click and keyboard relabeling', () => {
  const shapeEditor = page.match(/const openShapeLabelEditor = \(shapeIndex: number\) => \{[\s\S]*?\n  \};/)?.[0] ?? '';
  const selectedEditor = page.match(/const openSelectedShapeLabelEditor = \(\) => \{[\s\S]*?\n  \};/)?.[0] ?? '';
  const categoryEditor = page.match(/const openCategoryLabelEditor = \(category: string, count: number\) => \{[\s\S]*?\n  \};/)?.[0] ?? '';
  const documentCommit = page.match(/const commitAnnotationDocument = \([\s\S]*?\n  \};/)?.[0] ?? '';
  const shapeLabelUpdate = page.match(/const updateShapeLabel = \(shapeIndex: number, rawLabel: string\) => \{[\s\S]*?\n  \};/)?.[0] ?? '';
  assert.ok(shapeEditor && selectedEditor && categoryEditor);
  assert.doesNotMatch(`${shapeEditor}${selectedEditor}${categoryEditor}`, /showingPipelineImage/);
  assert.doesNotMatch(documentCommit, /showingPipelineImage|allowWhileShowingPipelineImage/);
  assert.doesNotMatch(shapeLabelUpdate, /showingPipelineImage|allowWhileShowingPipelineImage/);
  assert.match(shapeLabelUpdate, /if \(!committed\) return false;\s+setLastManualLabel\(label\);\s+return true/);
  assert.match(page, /className="annotation-category-main"[^>]+aria-keyshortcuts=\{shortcutAriaLabel\(shortcuts\['edit\.changeCategory'\]/);
  assert.match(page, /onDoubleClick=\{\(event\) => \{ event\.preventDefault\(\); event\.stopPropagation\(\); openCategoryLabelEditor\(category, categoryIndexes\.length\); \}\}/);
  assert.match(page, /className="annotation-object-main"[\s\S]*?onDoubleClick=\{\(event\) => \{ event\.preventDefault\(\); event\.stopPropagation\(\); openShapeLabelEditor\(index\); \}\}/);
  assert.match(page, /resolveShortcutAction\(event\.nativeEvent, shortcuts, 'app'\) === 'edit\.changeCategory'/);
  assert.match(page, /shortcutAction === 'edit\.changeCategory'[\s\S]*?openShapeLabelEditor\(selectedShapeIndex\)/);
  assert.match(page, /shortcutAction === 'edit\.changeCategory'/);
  assert.match(page, /if \(pendingShapeLabelEdit \|\| pendingCategoryLabelEdit\) \{\s+if \(event\.key === 'Escape' && !event\.isComposing\) \{/);
  assert.doesNotMatch(page, /if \(pendingShapeLabelEdit \|\| pendingCategoryLabelEdit\) \{\s+event\.preventDefault\(\)/);
  assert.match(page, /className="annotation-inline-label-editor batch"/);
  assert.match(page, /全数据集 · 将修改所有图片中的同名类别；已有名称会合并/);
  assert.match(page, /className="annotation-inline-label-editor object"/);
  assert.match(page, /仅修改此标注框 · Enter 确认 · Esc 取消/);
  assert.match(page, /kind: 'category_rename',[\s\S]*?source_category: sourceCategory,[\s\S]*?target_category: targetLabel/);
  assert.match(page, /documentIsDirty\(envelope\.document\)[\s\S]*?persistCurrentAnnotation\(true\)/);
  assert.match(css, /\.annotation-inline-label-editor\{min-width:0;display:grid/);
  assert.doesNotMatch(page, /pendingShapeLabelEdit && <div className="recovery-overlay"|pendingCategoryLabelEdit && <div className="recovery-overlay"/);
  assert.match(css, /\.annotation-category-row \.annotation-category-main:not\(:disabled\)\{cursor:text\}/);
});

test('image navigation sits at the canvas midline and exposes focused shortcuts', () => {
  assert.match(page, /className="canvas-page-navigation" aria-label="切换图片"/);
  assert.match(page, /className="canvas-page-button previous"/);
  assert.match(page, /className="canvas-page-button next"/);
  assert.match(page, /shortcutAction === 'navigation\.previous'[^}]+stepFile\(-1\)/);
  assert.match(page, /shortcutAction === 'navigation\.next'[^}]+stepFile\(1\)/);
  assert.match(page, /className="canvas-page-navigation" aria-label="切换图片" onDoubleClick=/);
  assert.match(css, /\.canvas-page-button\{position:absolute;top:50%/);
  assert.match(page, /data-tooltip-title="上一张图片"[^>]+data-shortcut=\{`\$\{displayShortcut\(shortcuts\['navigation\.previous'\], useMacShortcutSymbols\)\} \/ ←`\}/);
  assert.match(page, /data-tooltip-title="下一张图片"[^>]+data-shortcut=\{`\$\{displayShortcut\(shortcuts\['navigation\.next'\], useMacShortcutSymbols\)\} \/ →`\}/);
  assert.match(page, /<span className="canvas-page-arrow" aria-hidden="true">‹<\/span><\/button>/);
  assert.match(page, /<span className="canvas-page-arrow" aria-hidden="true">›<\/span><\/button>/);
  assert.doesNotMatch(page, /canvas-page-button[^>]*>[\s\S]{0,200}<kbd/);
  assert.doesNotMatch(page, /<strong>上一张<\/strong>|<strong>下一张<\/strong>/);
  assert.doesNotMatch(page, /index \+ direction \+ validFiles\.length\) % validFiles\.length/);
  assert.doesNotMatch(page, /<IconButton label="上一张"/);
  assert.doesNotMatch(page, /<IconButton label="下一张"/);
});

test('canvas image navigation keeps the active virtual file row visible', () => {
  assert.match(page, /activeItemKey=\{currentFile\?\.id \?\? null\}/);
  assert.match(page, /itemKey=\{fileItemKey\}/);
  assert.match(page, /aria-current=\{currentFile\?\.id === file\.id \? 'true' : undefined\}/);
  assert.match(virtualFileList, /items\.findIndex\(\(item\) => itemKey\(item\) === activeItemKey\)/);
  assert.match(virtualFileList, /scrollTopForActiveListItem\(\{/);
  assert.match(virtualFileList, /element\.scrollTop = nextScrollTop/);
});

test('selection tool blocks native canvas selection and image dragging', () => {
  assert.match(page, /const preventCanvasNativeDrag = \(event: React\.DragEvent<HTMLDivElement>\) => \{/);
  assert.match(page, /event\.button === 0 && tool === 'select' && !spaceDown/);
  assert.match(page, /tool === 'select' && event\.button === 0 && !spaceDown/);
  assert.doesNotMatch(page, /\(tool === 'select' \|\| showingPipelinePreview\)/);
  assert.match(page, /event\.preventDefault\(\);\s+}\s+startRealDraw\(event\)/);
  assert.match(page, /onDragStart=\{preventCanvasNativeDrag\}/);
  assert.match(page, /if \(tool !== 'select' \|\| spaceDown\) return/);
  assert.match(css, /\.canvas-stage \{[^}]*user-select:none;-webkit-user-select:none/);
});

test('rotation styling uses a top-right corner point without the legacy external arm', () => {
  assert.match(css, /\.rotation-corner-handle/);
  assert.match(css, /\.rotation-corner-handle:hover/);
  assert.doesNotMatch(css, /\.real-rotation-handle/);
  assert.doesNotMatch(css, /\.rotation-handle\{/);
});

test('toolbar separates canvas modes from drawing tools and uses a rotated rectangle icon', () => {
  assert.match(page, /role="group" aria-label="画布操作"/);
  assert.match(page, /role="group" aria-label="绘制标注"/);
  assert.match(page, /function ShapeTypeIcon\(/);
  assert.match(page, /shapeType === 'rotation' \? <rect x="4" y="5" width="12" height="10"[^>]+rotate\(-14 10 10\)/);
  assert.match(page, /<ShapeTypeIcon shapeType=\{id === 'brush' \? 'linestrip' : id\} \/>/);
  assert.doesNotMatch(page, /\['rotation',\s*'[◇▱]'/);
});

test('line drawing uses two clicks instead of pointer drag completion', () => {
  const startDraw = page.slice(page.indexOf('const startRealDraw'), page.indexOf('const moveRealDraw'));
  const lineBranch = startDraw.slice(startDraw.indexOf("if (tool === 'line')"), startDraw.lastIndexOf('event.currentTarget.setPointerCapture'));
  const endDraw = page.slice(page.indexOf('const endRealDraw'), page.indexOf('const cancelRealDraw'));

  assert.match(lineBranch, /if \(!start\) \{[\s\S]*drawRef\.current = \{ startX: point\.x, startY: point\.y \}/);
  assert.match(lineBranch, /createDragShape\('line', \[start\.startX, start\.startY\], coordinate\)/);
  assert.match(lineBranch, /drawRef\.current = null;[\s\S]*appendDraftShape\(shape as AnnotationShape, pointerAnchor\)/);
  assert.ok(lineBranch.indexOf('if (!shape)') < lineBranch.indexOf('drawRef.current = null'));
  assert.doesNotMatch(lineBranch, /setPointerCapture/);
  assert.match(endDraw, /\['rect', 'rotation', 'circle'\]\.includes\(tool\)/);
  assert.doesNotMatch(endDraw, /\['rect', 'rotation', 'line', 'circle'\]/);
  assert.match(page, /const handleCanvasDoubleClick = [\s\S]*if \(tool === 'line'\) \{[\s\S]*event\.preventDefault\(\);[\s\S]*event\.stopPropagation\(\);/);
  assert.match(page, /onDoubleClick=\{handleCanvasDoubleClick\}/);
  assert.match(page, /'点击确定直线起点 · 再点击确定终点'/);
  assert.match(page, /'移动指针预览 · 点击确定终点 · Esc 取消'/);
});

test('file rows do not imply a nested destination', () => {
  assert.doesNotMatch(page, /className="chevron"/);
});

test('canvas crosshair follows the pointer without blocking canvas interactions', () => {
  assert.match(page, /onPointerMoveCapture=\{updateCanvasCrosshair\}/);
  assert.match(page, /onPointerLeave=\{hideCanvasGuides\}/);
  assert.match(page, /event\.pointerType === 'touch' \|\| dragRef\.current/);
  assert.match(page, /className="canvas-crosshair" aria-hidden="true"/);
  assert.doesNotMatch(page, /className="pixel-cursor"/);
  assert.match(page, /<PixelReadout cursor=\{cursor\} \/>/);
  assert.match(css, /\.canvas-crosshair\{[^}]*pointer-events:none/);
  assert.match(css, /\.canvas-crosshair::before\{width:100%;height:1px/);
  assert.match(css, /\.canvas-crosshair::after\{width:1px;height:100%/);
  assert.match(css, /\.canvas-crosshair\[data-visible=true\]\{opacity:\.48\}/);
  assert.match(css, /\.canvas-crosshair::before\{[^}]*repeating-linear-gradient\(90deg/);
  assert.match(css, /\.canvas-crosshair::after\{[^}]*repeating-linear-gradient\(180deg/);
  assert.doesNotMatch(css, /\.canvas-crosshair::before,.canvas-crosshair::after\{[^}]*filter:/);
  assert.doesNotMatch(css, /\.canvas-crosshair\{[^}]*transition:/);
});

test('pixel coordinates and compact RGBA values live in the bottom status bar', () => {
  assert.match(page, /<output className=\{`pixel-readout/);
  assert.match(page, /className="channel-r">R<\/i><data>/);
  assert.match(page, /className="channel-g">G<\/i><data>/);
  assert.match(page, /className="channel-b">B<\/i><data>/);
  assert.match(page, /className="channel-a">A<\/i><data>/);
  assert.match(page, /className="channel-a">A<\/i><data>[^<]+<\/data><\/span><span className="pixel-field"><i className="channel-v">V<\/i><data>/);
  assert.match(page, /<footer className="statusbar">[^\n]*<PixelReadout cursor=\{cursor\} \/>/);
  assert.doesNotMatch(page, /className="pixel-cursor"|R=|G=|B=|A=/);
  assert.match(css, /\.channel-r\{color:#ff8a8a\}/);
  assert.match(css, /\.channel-g\{color:#71d49a\}/);
  assert.match(css, /\.channel-b\{color:#81afff\}/);
});

test('statusbar uses three balanced groups without exposing annotation revision', () => {
  assert.match(page, /className="statusbar-group statusbar-context"/);
  assert.match(page, /className="statusbar-group statusbar-pixels"><PixelReadout cursor=\{cursor\} \/>/);
  assert.match(page, /className="statusbar-group statusbar-actions"/);
  assert.match(page, /className="statusbar-group statusbar-context">[^\n]*<span className="healthy">/);
  assert.match(page, /className="statusbar-group statusbar-actions"><label className=\{`statusbar-grid-toggle/);
  assert.match(page, /<input type="checkbox" aria-label="显示真实像素网格" checked=\{showPixel\}/);
  assert.match(page, /pixelGridVisible \? '真实网格已显示' : '真实网格待显示'/);
  assert.doesNotMatch(page, /'网格 关'|'网格 显示中'|'网格 待显示'/);
  assert.doesNotMatch(page, /annotationDraft \? `revision \$\{annotationDraft\.revision/);
  assert.match(css, /\.statusbar\{display:grid;[^}]*grid-template-columns:minmax\(0,1fr\) auto minmax\(0,1fr\)/);
  assert.match(css, /\.statusbar-pixels\{justify-content:center\}/);
  assert.match(css, /\.statusbar-actions\{justify-content:flex-end/);
  assert.match(css, /\.statusbar-grid-toggle>input:checked\+\.statusbar-grid-check::after\{content:'✓'\}/);
  assert.match(css, /\.statusbar-grid-toggle>span:last-child\{min-width:5em\}/);
  assert.doesNotMatch(css, /@media\(max-width:1320px\)\{\.statusbar-source\{display:none\}\}/);
});

test('zoom percentage edits on click and resets to 100% on double-click', () => {
  assert.match(page, /onClick=\{\(event\) => \{ if \(event\.detail === 1 && !zoomEditing\) beginZoomEditing\(\); \}\}/);
  assert.match(page, /onDoubleClick=\{\(event\) => \{ event\.preventDefault\(\); resetZoomPercent\(\); \}\}/);
  assert.match(page, /setView\(\(old\) => zoomCanvasView\(old, 1 \/ old\.scale, maximumCanvasScale\(\)\)\)/);
  assert.match(page, /role="spinbutton" value=\{zoomEditing \? zoomDraft : String\(Math\.round\(view\.scale \* 100\)\)\} readOnly=\{!zoomEditing\}/);
  assert.match(page, /aria-valuemin=\{25\}/);
  assert.match(page, /aria-valuemax=\{Math\.round\(maximumCanvasScale\(\) \* 100\)\}/);
  assert.match(page, /event\.key === 'Enter'/);
  assert.match(page, /event\.key === 'Escape'/);
  assert.match(css, /\.zoom-value-control\{width:52px;[^}]*flex:none/);
  assert.match(css, /\.zoom-value-control:hover\{[^}]*border-color:[^}]*background:/);
  assert.match(css, /\.zoom-value-control\.is-editing\{[^}]*border-color:/);
});

test('pixel grid maps every cell to one real source pixel', () => {
  assert.doesNotMatch(css, /\.image-surface\.pixel-grid::after|background-size:8px 8px/);
  assert.match(page, /MIN_SOURCE_PIXEL_GRID_SIZE/);
  assert.match(page, /MAX_SOURCE_PIXEL_INSPECTION_SIZE/);
  assert.match(page, /sourcePixelScreenSize\(displayedWidth \?\? 0, annotationSurfaceWidth, view\.scale\)/);
  assert.match(page, /<pattern id="source-pixel-grid" width="1" height="1" patternUnits="userSpaceOnUse">/);
  assert.match(page, /preserveAspectRatio="none"/);
  assert.match(page, /data-cell-x=\{sourcePixelWidthOnScreen\} data-cell-y=\{sourcePixelHeightOnScreen\}/);
  assert.match(css, /\.source-pixel-grid-line\{[^}]*stroke-width:\.5;[^}]*vector-effect:non-scaling-stroke;shape-rendering:crispEdges/);
  assert.match(css, /\.image-surface\.source-pixels-visible img,[^}]+image-rendering:pixelated/);
});

test('file JSON state uses the shared custom select and the row status slot', () => {
  assert.match(page, /<CustomSelect className=\{`annotation-filter-select/);
  assert.match(page, /fileAnnotationFilters\.map\(\(option\) => \(\{ value: option, label:/);
  assert.doesNotMatch(page, /<select/);
  assert.match(page, /resolveFileStatusIndicator\(file\.annotationFileExists, file\.annotations/);
  assert.match(page, /className={`file-status-indicator \$\{statusView\.kind\}`}/);
  assert.match(page, /statusView\.kind === 'check' \? '✓'/);
  assert.match(css, /\.annotation-filter-select/);
  assert.match(css, /\.file-status-indicator\{box-sizing:border-box;width:16px;height:16px;[^}]*border-radius:50%/);
  assert.match(css, /\.file-status-indicator\.check\{background:#53d9b5;color:#06251d/);
  assert.match(css, /\.file-status-indicator\.failed\{border:1px solid #c86670;background:#351b20;color:#f5a0a8/);
  assert.match(css, /\.file-status-indicator\.progress>i\{background:conic-gradient/);
  assert.match(css, /\.file-status-indicator\.running>i\{[^}]*animation:file-status-spin \.8s linear infinite/);
  assert.match(page, /const rowTitle = statusView\.kind !== 'empty' \? statusView\.label/);
  assert.match(page, /data-tooltip=\{rowTitle\}/);
  assert.doesNotMatch(page, /title=\{rowTitle\}|title=\{statusView\.label\}/);
  assert.doesNotMatch(page, /nextFileAnnotationFilter/);
  assert.doesNotMatch(css, /\.annotation-status-mark/);
  assert.doesNotMatch(page, /filter === 'exceptions'/);
  assert.doesNotMatch(page, /异常恢复中心/);
  assert.doesNotMatch(page, /className="filter-row"/);
});

test('canvas cursor styling differentiates selection, pan, drawing, resizing and rotation', () => {
  assert.match(page, /resolveCanvasCursorMode\(tool, \{/);
  assert.match(page, /resize-\$\{resizeCursor\}/);
  assert.match(css, /\.cursor-select \.real-shape:not\(\.prediction\)\{cursor:move\}/);
  assert.match(css, /\.cursor-pan \.canvas-stage[^}]+\{cursor:grab\}/);
  assert.match(css, /\.cursor-panning \.canvas-stage[^}]+\{cursor:grabbing\}/);
  assert.match(css, /\.cursor-draw \.canvas-stage[^}]+\{cursor:crosshair\}/);
  assert.match(css, /\.cursor-draw \.canvas-stage[^}]+\.cursor-draw \.annotation-box/);
  assert.match(css, /\.shape-control-point\{[^}]+cursor:nwse-resize\}/);
  assert.match(css, /\.shape-control-point:active\{cursor:nwse-resize\}/);
  assert.match(css, /\.shape-control-point\.resize-nesw-resize\{cursor:nesw-resize\}/);
  assert.match(css, /\.rotation-corner-handle\{cursor:url\('\/cursors\/rotate\.svg'\) 12 12,crosshair/);
});

test('current-image auto-save lives in the statusbar without legacy save actions', () => {
  assert.doesNotMatch(page, /文件⌄|工具⌄|fileMenuOpen|toolsOpen|className="tools-menu/);
  assert.doesNotMatch(css, /\.tools-menu|\.tools-button|\.file-autosave-toggle/);
  assert.equal(page.match(/className="open-dataset-button"/g)?.length, 1);
  assert.match(page, /className={`statusbar-autosave \$\{annotationAutoSave \? 'active' : ''\}/);
  assert.match(page, /role="switch" aria-label="当前图自动保存" aria-describedby="current-image-save-status" aria-checked=\{annotationAutoSave\}/);
  assert.match(page, /id="current-image-save-status" aria-live="polite"/);
  assert.match(page, /<div className="top-actions">/);
  assert.match(page, /const saveModeLabel = annotationSaving/);
  assert.match(page, /\? '本机草稿'/);
  assert.match(page, /\? '未保存'/);
  assert.match(page, /: '已保存'/);
  assert.doesNotMatch(page, /top-quick-action|save-state|save-mode-button/);
  assert.doesNotMatch(page, /设置标注目录|导入当前标注 JSON|导出当前标注 JSON|annotationImportPreview/);
  assert.doesNotMatch(page, /state\.annotationAutoSave/);
  assert.match(page, /annotationAutoSaveRef\.current = DEFAULT_ANNOTATION_AUTO_SAVE/);
  assert.match(css, /\.statusbar-autosave\{/);
  assert.match(css, /\.statusbar-autosave>i\{[^}]+border-radius:999px/);
  assert.match(css, /\.statusbar-autosave\{height:24px\}/);
});

test('top-right actions expose settings and a stateful fullscreen control', () => {
  assert.match(page, /function SettingsIcon\(\)/);
  assert.match(page, /function FullscreenIcon\(\{ active \}/);
  assert.match(page, /ref=\{settingsButtonRef\}[^>]+aria-label="设置"[^>]+aria-expanded=\{settingsOpen\}[^>]+aria-controls="global-settings-page"/);
  assert.match(page, /settingsOpen && <GlobalSettingsPage/);
  assert.match(page, /className=\{`top-action-button \$\{isFullscreen \? 'active' : ''\}`\} aria-label=\{isFullscreen \? '退出全屏' : '进入全屏'\}/);
  assert.match(page, /document\.addEventListener\('fullscreenchange', syncFullscreenState\)/);
  assert.match(page, /appShellRef\.current\.requestFullscreen\(\)/);
  assert.match(page, /document\.exitFullscreen\(\)/);
  assert.match(css, /\.top-action-button\{width:30px;height:30px/);
  assert.match(page, /closeButtonRef=\{settingsCloseRef\}/);
});

test('display controls live with the objects and model results they affect', () => {
  assert.doesNotMatch(page, /显示图层|className="layer-row"|className="layer-swatch/);
  assert.doesNotMatch(css, /\.layer-row|\.layer-swatch|\.layers-panel>\.side-section:first-child/);
  assert.match(page, /\['layers', '对象'\]/);
  assert.match(page, /className={`annotation-master-visibility \$\{showGT \? 'on' : ''\}`}/);
  assert.match(page, /aria-label=\{showGT \? '隐藏画布中的全部标注' : '显示画布中的全部标注'\}/);
  assert.match(page, /className="annotation-category-list prediction-category-list"/);
  assert.match(page, /aria-label="当前图 AI 预测类别"/);
  assert.match(page, /<strong>分割 \/ 像素结果<\/strong>/);
  assert.match(page, /<strong>分类浮层<\/strong>/);
  assert.doesNotMatch(page, /画布模型预测|showPred|setShowPred/);
  assert.match(page, /className={`statusbar-grid-toggle \$\{showPixel \? 'active' : ''\}/);
  assert.match(page, /<input type="checkbox" aria-label="显示真实像素网格" checked=\{showPixel\}/);
  assert.match(css, /\.annotation-master-visibility\{/);
  assert.match(css, /\.prediction-display-control\{/);
  assert.match(css, /\.statusbar-grid-toggle\{/);
});

test('canvas pipeline entry exposes a compact read-only preview without changing the right tab', () => {
  const trigger = page.match(/<button[^>]*className={`pipeline-chip[\s\S]*?<\/button>/)?.[0];
  const preview = page.match(/<section id="pipeline-summary-popover"[\s\S]*?<\/section>/)?.[0];
  assert.ok(trigger);
  assert.ok(preview);
  assert.match(trigger, /aria-expanded=\{pipelineSummaryOpen\}/);
  assert.match(trigger, /aria-controls="pipeline-summary-popover"/);
  assert.match(trigger, /aria-pressed=\{pipelineEnabled\}/);
  assert.match(trigger, /changePipelineEnabled\(!pipelineEnabled\)/);
  assert.match(trigger, /className="pipeline-chip-switch"/);
  assert.match(trigger, /className="pipeline-chip-state"/);
  assert.match(trigger, /className="pipeline-chip-action"/);
  assert.doesNotMatch(trigger, /openRightTab|setRightTab/);
  assert.match(page, /onMouseEnter=\{\(\) => setPipelineSummaryOpen\(true\)\}/);
  assert.match(page, /onMouseLeave=\{\(\) => setPipelineSummaryOpen\(false\)\}/);
  assert.match(page, /onFocusCapture=\{\(\) => setPipelineSummaryOpen\(true\)\}/);
  assert.match(page, /event\.currentTarget\.contains\(next\)/);
  assert.match(page, /event\.key === 'Escape'/);
  assert.match(preview, /role="region" aria-label="处理流步骤预览"/);
  assert.match(preview, /pipelineLinearNodes\.map\(\(node\)/);
  assert.match(preview, /执行顺序自上而下；顶部固定原图像，底部固定显示/);
  assert.match(preview, /isDisplay \? `D\$\{displayIndex \+ 1\}` : mainIndex \+ 1/);
  assert.doesNotMatch(preview, /<button|<input|<select|<textarea|onClick|contentEditable/);
  assert.match(page, /rightTab === 'pipeline' && <section className="pipeline-panel">/);
  assert.doesNotMatch(page, /openRightTab\('pipeline'\)|pipelineDrawer|setPipelineDrawer|pipeline-drawer|画布快捷栏|处理流快捷控制|编辑完整流程/);
  assert.doesNotMatch(css, /\.pipeline-drawer|\.quick-flow|\.quick-node|\.drawer-close|drawer-up|\.pipeline-chip\.panel-open/);
  assert.match(css, /\.pipeline-summary-popover\{position:absolute;[^}]*width:min\(244px/);
  assert.match(css, /\.pipeline-summary-popover ol\{[^}]*overflow-y:auto/);
  assert.match(css, /\.pipeline-chip\.summary-open\{/);
  assert.match(css, /\.pipeline-chip:hover \.pipeline-chip-state/);
  assert.match(css, /\.pipeline-chip:hover \.pipeline-chip-action/);
  assert.match(css, /\.switch\{/);
  assert.match(css, /\.pipeline-background-toggle\{/);
});

test('canvas labels keep screen-sized typography and adapt at image edges', () => {
  assert.match(page, /canvasLabelLayouts\(/);
  assert.match(page, /canvasVisibleImageBounds\(/);
  assert.match(page, /fitCanvasLabelText\(fullLabelText\)/);
  assert.match(page, /sourceWidth \/ \(imageSurfaceWidth \* Math\.max\(view\.scale, 0\.01\)\)/);
  assert.match(page, /data-placement=\{labelLayout\.placement\}/);
  assert.match(page, /transform={`translate\(\$\{labelLayout\.x\} \$\{labelLayout\.y\}\) scale\(\$\{labelLayout\.unit\}\)`}/);
  assert.match(page, /shape\.shape_type === 'circle'/);
  const geometryLayer = page.indexOf("renderRealShape(shape, index, false, 'geometry')");
  const labelLayer = page.indexOf("renderRealShape(entry.shape, entry.index, entry.prediction, 'label')");
  const controlLayer = page.indexOf("renderRealShape(shape, index, false, 'controls')");
  assert.ok(geometryLayer >= 0 && geometryLayer < labelLayer && labelLayer < controlLayer);
  assert.match(page, /const canvasPresentation = resolveCanvasPresentation/);
  assert.match(page, /const displayedShapes = canvasPresentation\.shapes/);
  assert.match(page, /annotationHitCandidates\(displayedShapes,/);
  assert.match(page, /const currentIndex = selectedShapeIndex;\s+const hitIndex = selectAnnotationHitIndex\(candidates, currentIndex, event\.detail > 1\)/);
  assert.match(page, /startShapeMove\(hitIndex, event\)/);
  assert.doesNotMatch(page, /onPointerDown: \(event: React\.PointerEvent<SVGElement>\) => startShapeMove\(index, event\)/);
  assert.match(page, /onDoubleClick=\{handleCanvasDoubleClick\}/);
  assert.match(page, /data-shape-index=\{index\}/);
  assert.match(page, /shape-canvas-label[^`]+\$\{selected \? 'selected' : ''\}/);
  assert.match(page, /list\.scrollTo\(\{ top, behavior: reduceMotion \? 'auto' : 'smooth' \}\)/);
  assert.match(page, /const safelyVisible = minX >= labelVisibleBounds\.left \+ padding/);
  assert.match(page, /revealShapeFromList\(shape\)/);
  assert.match(page, /aria-current=\{rowSelected \? 'true' : undefined\}/);
  assert.match(css, /@keyframes canvas-selection-confirm/);
  assert.doesNotMatch(css, /@keyframes canvas-selection-confirm\{[^}]*transform:/);
  assert.doesNotMatch(css, /@keyframes object-row-selection-confirm|\.annotation-object-row\.selected\{animation:/);
  assert.match(css, /@media\(prefers-reduced-motion:reduce\)\{\.real-shape\.selected,[^}]+animation:none/);
  assert.doesNotMatch(page, /labelFontSize = Math\.max\(10, \(displayedWidth/);
  assert.match(css, /\.shape-canvas-label text\{[^}]*vector-effect:non-scaling-stroke/);
  assert.doesNotMatch(page, /demo-annotation-layer|renderDemoCanvasLabel|demoAnnotationShapes/);
  assert.match(page, /selectedShapeIndex === index/);
  assert.match(page, /canvasAnnotationOpticalScale\(view\.scale\)/);
  assert.match(page, /CANVAS_CONTROL_POINT_RADIUS_PX \* screenUnit/);
  assert.doesNotMatch(page, /canvasControlPointRadiusPx\(view\.scale\)/);
  assert.match(page, /CANVAS_CONTROL_POINT_RADIUS_PX \* rotationPreviewScreenUnit/);
  assert.match(page, /'--annotation-optical-scale': annotationOpticalScale/);
  assert.match(page, /'--annotation-stroke': `\$\{1\.1 \* annotationOpticalScale\}px`/);
  assert.match(page, /'--annotation-control-stroke': '1\.6px'/);
  assert.match(page, /'--annotation-vertex-stroke': `\$\{CANVAS_VERTEX_CONTROL_DIAMETER_PX\}px`/);
  assert.doesNotMatch(page, /'--annotation-(?:control|vertex)[^']*': `\$\{[^}]*annotationOpticalScale/);
  assert.doesNotMatch(css, /\.image-surface \{[^}]*will-change:transform/);
  assert.match(page, /className={`real-annotation-layer screen-annotation-layer/);
  assert.match(page, /left: annotationStageWidth \/ 2 - annotationLayerWidth \/ 2 \+ view\.x/);
  assert.match(page, /width: annotationLayerWidth/);
  assert.match(css, /\.screen-annotation-layer\{inset:auto;z-index:5;overflow:visible;shape-rendering:geometricPrecision;text-rendering:geometricPrecision\}/);
  assert.match(css, /\.real-shape\{stroke-width:var\(--annotation-stroke,1\.1px\);vector-effect:non-scaling-stroke\}/);
  assert.match(css, /\.real-shape\.selected\{stroke-width:var\(--annotation-selected-stroke,1\.45px\);/);
  assert.match(css, /\.real-shape\.prediction\{stroke-width:var\(--annotation-prediction-stroke,1px\)\}/);
  assert.match(css, /\.shape-control-vertices\{[^}]*stroke-width:var\(--annotation-vertex-stroke,6px\)/);
  assert.match(css, /\.shape-control-point:hover,\.rotation-corner-handle:hover\{[^}]*stroke-width:var\(--annotation-control-hover-stroke,2\.2px\)/);
  assert.match(css, /\.shape-canvas-label text\{paint-order:normal;stroke:none;text-rendering:geometricPrecision\}/);
  assert.doesNotMatch(css, /\.demo-annotation-layer|\.wafer-core|\.box-one/);
});
