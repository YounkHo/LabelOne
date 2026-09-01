import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../hooks/use-local-backend.ts', import.meta.url), 'utf8');

test('the inference card renders the selected model schema instead of fixed confidence controls', () => {
  assert.match(page, /parametersSchema: normalizeInferenceParameterSchema\(model\.capabilities\.parameters_schema\)/);
  assert.match(page, /Object\.entries\(selectedModel\.parametersSchema\)\.map\(\(\[name, schema\]\) => <PipelineParameterControl/);
  assert.match(page, /setInferenceParameters\(inferenceParameterDefaults\(model\.parametersSchema\)\)/);
  assert.match(page, /parameters: singleInferenceParameters/);
  assert.match(page, /parameters: \{ \.\.\.inferenceParameters, \.\.\.\(isSamModel \? samPromptParameters : \{\}\) \}/);
  assert.doesNotMatch(page, /confidenceThreshold|setConfidenceThreshold|iou_threshold: 0\.45|切片推理/);
});

test('model selection loads and automatically runs the latest current-image intent', () => {
  assert.match(page, /const chooseModel = \(id: string, closePicker = true, autoLoad = true\)/);
  assert.match(page, /void loadModelById\(model, operationId\)/);
  assert.match(page, /lastAutoInferenceSignatureRef\.current === signature/);
  assert.match(page, /window\.setTimeout\(\(\) => \{[\s\S]*?autoInferenceRunnerRef\.current\(\);[\s\S]*?\}, 400\)/);
  assert.match(page, /completedInferenceSignature === currentInferenceRequestSignature/);
  assert.doesNotMatch(page, /className="run-button model-run"/);
});

test('runtime and inference requests are generation guarded and abort superseded work', () => {
  assert.match(hook, /const runtimeController = useRef<AbortController \| null>\(null\)/);
  assert.match(hook, /const runtimeRequestId = useRef\(0\)/);
  assert.match(hook, /runtimeController\.current\?\.abort\(\)/);
  assert.match(hook, /requestId !== runtimeRequestId\.current \|\| controller\.signal\.aborted/);
  assert.match(hook, /const inferenceController = useRef<AbortController \| null>\(null\)/);
  assert.match(hook, /requestId !== inferenceRequestId\.current \|\| controller\.signal\.aborted/);
});

test('SAM prompts are cleared across every model change and gate automatic inference', () => {
  const chooseModel = page.match(/const chooseModel = \(id: string[\s\S]*?\n  \};/)?.[0] ?? '';
  assert.match(chooseModel, /setSamPoints\(\[\]\)/);
  assert.match(chooseModel, /setSamBoxes\(\[\]\)/);
  assert.match(page, /if \(isSamModel && samPromptCount === 0\) return/);
});
