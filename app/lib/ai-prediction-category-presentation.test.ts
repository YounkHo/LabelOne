import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../globals.css', import.meta.url), 'utf8');

test('AI categories reuse the manual category row structure while staying read-only', () => {
  const layers = page.match(/rightTab === 'layers' && <div className="layers-panel">[\s\S]*?rightTab === 'pipeline'/)?.[0] ?? '';
  const aiCategoryStart = layers.indexOf('className="annotation-category-list prediction-category-list"');
  const aiCategoryGroup = aiCategoryStart >= 0 ? layers.slice(aiCategoryStart, layers.indexOf('</section>', aiCategoryStart)) : '';
  assert.match(aiCategoryGroup, /annotation-category-row prediction-category-row/);
  assert.match(aiCategoryGroup, /className="annotation-category-icon"/);
  assert.match(aiCategoryGroup, /annotation-category-color prediction-readonly-color/);
  assert.match(aiCategoryGroup, /className={`annotation-category-visibility/);
  assert.match(aiCategoryGroup, /annotation-category-main prediction-category-main/);
  assert.match(aiCategoryGroup, /annotation-category-source-badge/);
  assert.match(aiCategoryGroup, /最高 \{\(category\.maxScore \* 100\)\.toFixed\(1\)\}%/);
  assert.doesNotMatch(aiCategoryGroup, /type="color"|openCategoryLabelEditor|annotation-category-delete/);
  assert.doesNotMatch(page, /ai-prediction-category-group|ai-prediction-category-row|ai-prediction-badge/);
  assert.doesNotMatch(css, /\.ai-prediction-category-group|\.ai-prediction-category-row|\.ai-prediction-badge/);
});

test('AI category and single-box visibility use stable prediction keys and reset together', () => {
  assert.doesNotMatch(page, /showDetections|setShowDetections/);
  assert.match(page, /const \[hiddenPredictionKeys, setHiddenPredictionKeys\] = useState<Set<string>>/);
  assert.match(page, /const availablePredictionEntries = currentDetectionPredictions\.flatMap/);
  assert.match(page, /const visibleCurrentPredictionEntries = availablePredictionEntries\.filter/);
  assert.match(page, /!hiddenPredictionCategories\.has\(entry\.label\) && !hiddenPredictionKeys\.has\(entry\.key\)/);
  assert.match(page, /setHiddenPredictionCategories\(new Set\(\)\);\s+setHiddenPredictionKeys\(new Set\(\)\);[\s\S]*?\[aiPredictionCategories\.length, aiPredictionCategoryKey\]/);
  assert.match(page, /hiddenPredictionKeys\.has\(key\)\) hidden\.add\(index\)/);
  assert.match(css, /\.annotation-category-groups\{[^}]*overflow:auto/);
});

test('AI boxes reuse manual object rows and preserve visibility, selection, and promotion', () => {
  assert.match(page, /const \[objectSourceTab, setObjectSourceTab\] = useState<'manual' \| 'ai'>\('manual'\)/);
  assert.doesNotMatch(page, /const \[, setObjectSourceTab\]/);
  assert.match(page, /className="object-source-tabs" role="tablist"/);
  assert.match(page, />人工标注 <b>\{objectListShapes\.length\}<\/b>/);
  assert.match(page, />AI 预测 <b>\{availablePredictionEntries\.length\}<\/b>/);
  assert.match(page, /className="annotation-object-list prediction-object-list"/);
  assert.match(page, /annotation-object-row prediction-object-row/);
  assert.match(page, /className="annotation-object-icon"/);
  assert.match(page, /className={`annotation-object-shape/);
  assert.match(page, /className={`annotation-object-visibility/);
  assert.match(page, /className="annotation-object-main"/);
  assert.match(page, /className="prediction-promote-button"/);
  assert.doesNotMatch(page, /ai-prediction-object-list|ai-prediction-object-row|ai-prediction-object-main/);
  assert.match(page, /const promotePredictionToManual = \(predictionIndex: number\)/);
  assert.match(page, /commitAnnotationDocument\(\{ \.\.\.document, shapes: \[\.\.\.\(document\.shapes \?\? \[\]\), promoted\] \}\)/);
  assert.match(page, /delete promoted\.score/);
  assert.match(page, /prediction_key: inferencePredictionKey/);
  assert.match(page, /setSelectedShapeIndex\(nextIndex\);\s+setObjectSourceTab\('manual'\)/);
  assert.match(page, /if \(aiPredictionCategories\.length === 0\) setObjectSourceTab\('manual'\)/);
  assert.match(css, /\.prediction-object-row \.annotation-object-main\{/);
  assert.match(css, /\.prediction-promote-button\{/);
  assert.match(css, /\.real-shape\.prediction\.selected\{/);
});
