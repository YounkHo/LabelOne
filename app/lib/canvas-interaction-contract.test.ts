import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('pipeline image uses transformed labels while source annotation editing stays mapped', () => {
  const enable = page.slice(page.indexOf('const changePipelineEnabled'), page.indexOf('const changePipelineScope'));
  const scope = page.slice(page.indexOf('const changePipelineScope'), page.indexOf('const loadAnnotationFor'));
  const draw = page.slice(page.indexOf('const startRealDraw'), page.indexOf('const moveRealDraw'));
  const automaticPreviewCall = page.indexOf("void previewPipeline({ dataset_id: dataset.id, asset_id: currentFile.id, priority: 'interactive'");
  const automaticPreviewStart = page.lastIndexOf('useEffect(() => {', automaticPreviewCall);
  const automaticPreviewEnd = page.indexOf('\n  useEffect(() => {', automaticPreviewCall);
  const automaticPreview = page.slice(automaticPreviewStart, automaticPreviewEnd);

  assert.doesNotMatch(enable, /cancelAnnotationDrafting|setTool/);
  assert.match(enable, /if \(enabled\) \{\s+setSinglePipelineSource\(finalPipelineSource\);\s+setVisualizationDisplayMode\('source'\)/);
  assert.doesNotMatch(scope, /cancelAnnotationDrafting|setTool/);
  assert.doesNotMatch(draw, /showingPipelinePreview|showingPipelineImage/);
  assert.equal(automaticPreviewCall >= 0 && automaticPreviewStart >= 0 && automaticPreviewEnd > automaticPreviewCall, true);
  assert.ok(automaticPreview);
  assert.doesNotMatch(automaticPreview, /cancelAnnotationDrafting|annotationDirty|请先保存当前标注/);
  assert.doesNotMatch(page, /const runPipeline|▶ 运行|重新校验处理流/);
  assert.match(page, /const canvasPresentation = resolveCanvasPresentation\(\{/);
  assert.match(page, /annotationShapes: draftShapes/);
  assert.match(page, /pipelineAnnotationShapes: liveDerivedAnnotationShapes/);
  assert.match(page, /pipelineWidth: activeStandaloneRaster\?\.width \?\? canvasPipelineItem\?\.width/);
  assert.match(page, /pipelineHeight: activeStandaloneRaster\?\.height \?\? canvasPipelineItem\?\.height/);
  assert.match(page, /canvasCoordinateTransformFromPipelineMapping\(pipelineCoordinateMappingForItem\(canvasPipelineItem\)\)/);
  assert.match(page, /draftShapes\.map\(\(shape\) => transformCanvasShape\(shape, activeCanvasCoordinateTransform\)\)/);
  assert.match(page, /const displayedShapes = canvasPresentation\.shapes/);
  assert.match(page, /const showingPipelineImage = canvasPresentation\.showingPipelineImage/);
  assert.match(page, /const objectListShapes = annotationObjects/);
  assert.match(page, /resolvePipelineCoordinateTransform\(/);
  assert.match(page, /draftShapes\.map\(\(shape\) => transformCanvasShape/);
  assert.match(page, /canvasEditCoordinateTransform \? inverseTransformCanvasDelta/);
  assert.match(page, /const sourceAnnotationPoint =/);
  assert.match(page, /const showControlPoints = !prediction && selectedShapeIndex === index/);
  assert.doesNotMatch(page, /selectedPreviewShapeIndex|处理流标注 · 只读副本|处理流只读副本/);
  assert.match(page, /handlePipelineImageError\(activePipelineArtifactUrl\)/);
  assert.match(page, /pipelineImageRetryExhausted\(pipelineImageAttempts\[activePipelineArtifactUrl\] \?\? 0\)/);
  assert.match(page, /已回退原图，画布仍可编辑/);
});

test('completed drawing arms a one-shot blank click transition to select', () => {
  const commit = page.slice(page.indexOf('const commitPendingManualShape'), page.indexOf('const finishPolygonDraft'));
  const capture = page.slice(page.indexOf('const focusCanvasFromPointer'), page.indexOf('const hideCanvasCrosshair'));

  assert.match(commit, /selectOnNextCanvasBlankRef\.current = true/);
  assert.doesNotMatch(commit, /showingPipelineImage|allowWhileShowingPipelineImage/);
  assert.match(commit, /if \(!committed\) \{[\s\S]*return;[\s\S]*pendingManualShapeRef\.current = null/);
  assert.ok(commit.indexOf('const committed = commitAnnotationDocument') < commit.indexOf('pendingManualShapeRef.current = null'));
  assert.doesNotMatch(commit, /setTool\('select'\)/);
  assert.match(capture, /target === event\.currentTarget \|\| target\?\.classList\.contains\('screen-annotation-layer'\) === true/);
  assert.match(capture, /shouldSwitchToSelectAfterBlankClick\(\{/);
  assert.match(capture, /event\.preventDefault\(\);\s+event\.stopPropagation\(\);\s+setTool\('select'\)/);
  assert.match(page, /const activateTool = \(nextTool: AnnotationTool\) => \{\s+cancelAnnotationDrafting\(\)/);
});
