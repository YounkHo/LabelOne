import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('all scope creates one revision and registry keyed background preview job after validation', () => {
  assert.match(page, /pipelineScope !== 'all'/);
  assert.match(page, /pipelineValidationReady/);
  assert.match(page, /currentPipelinePrecomputeJob && !pipelinePrecomputeMayAutoResume\)\) return/);
  assert.match(page, /priority: 'background'/);
  assert.match(page, /searchedDatasetIndexRevision[\s\S]*?backend\.datasets\.data\.datasets/);
  assert.match(page, /currentPipelineRegistryHash = validatedPipelineKey === currentPipelineValidationKey \? pipelineValidationState\.data\?\.registry_hash/);
  assert.match(page, /pipelinePrecomputeKey\(dataset\.id, `\$\{currentPipelineExecutionSignature\}:format-\$\{PIPELINE_PREVIEW_FORMAT\}`\)/);
  assert.doesNotMatch(page, /pipelineOutputMode|pipelineOutputRoot|pipelineOutputFormat|pipelineOutputConflict/);
  assert.match(page, /ensurePipelinePrecompute\(\{/);
  assert.match(page, /output_policy: \{ mode: 'preview', image_format: PIPELINE_PREVIEW_FORMAT, conflict: 'reuse' \}/);
  assert.doesNotMatch(page, /frontend_precompute_key:/);
});

test('pipeline preprocessing has no destructive output policy UI or duplicate static contract', () => {
  assert.doesNotMatch(page, /批处理输出|Derived dataset|output_root|文件冲突|className={`pipeline-output-policy|className="flow-contract"/);
});

test('completed batch item previews enter the signature cache and may restore only the current graph', () => {
  assert.match(page, /pipelinePreviewResultFromJobItem\(item\.result\)/);
  assert.match(page, /cachePipelinePreview\(result, currentPipelineExecutionSignature\)/);
  assert.match(page, /pipelineScope === 'all'[\s\S]*?assetId === currentFile\?\.id[\s\S]*?restorePipelinePreview\(result\)/);
  assert.match(page, /pipelinePreviewCacheKey\(dataset\.id, currentFile\.id, currentPipelineExecutionSignature\)/);
});

test('reload discovers the matching server-versioned automatic job and safely pauses stale automatic work', () => {
  assert.match(page, /reusablePipelinePrecomputeRecord = currentPipelinePrecomputeKey \? backendJobs\.find/);
  assert.match(page, /job\.request\.priority === 'background'/);
  assert.match(page, /job\.request\.output_policy\?\.mode === 'preview'/);
  assert.match(page, /job\.request\.pipeline_context\?\.dataset_index_revision === currentDatasetIndexRevision/);
  assert.match(page, /job\.request\.pipeline_context\.registry_hash === currentPipelineRegistryHash/);
  assert.match(page, /pipelineRequestNodesEqual\(job\.request\.pipeline_nodes, pipelineRequestNodes\)/);
  assert.match(page, /job\.request\.pipeline_context/);
  assert.match(page, /job\.request\.priority === 'background'/);
  assert.match(page, /job\.request\.output_policy\?\.mode === 'preview'/);
  assert.match(page, /!job\.request\.asset_ids\?\.length/);
  assert.match(page, /const stillCurrent = pipelineEnabled[\s\S]*?pipelineRequestNodesEqual/);
  assert.doesNotMatch(page, /stillCurrent[\s\S]{0,300}!annotationDirty/);
  assert.match(page, /controlPipelineJob\(job\.job_id, 'pause'\)/);
  assert.match(page, /pipelinePrecomputePausePendingRef/);
});

test('only frontend-paused jobs auto-resume when returning from current to all scope', () => {
  assert.match(page, /automaticallyPausedPipelineJobs/);
  assert.match(page, /pipelinePrecomputeMayAutoResume/);
  assert.match(page, /currentPipelinePrecomputeJob && !pipelinePrecomputeMayAutoResume/);
  assert.match(page, /setAutomaticallyPausedPipelineJobs\(\(old\) => new Set\(old\)\.add\(job\.job_id\)\)/);
  assert.match(page, /then\(\(\{ job, resumed \}\)/);
  assert.match(page, /if \(resumed\) setAutomaticallyPausedPipelineJobs/);
});

test('all preview remains automatic without a duplicate manual run button', () => {
  assert.doesNotMatch(page, /pipelinePrecomputeButtonLabel|pipelineAllPreviewAutomatic|▶ 运行|className="run-button compact"/);
  assert.match(page, /实时 \+ 全库/);
  assert.match(page, /实时当前图/);
});

test('annotation edits do not block or pause image-only pipeline precompute', () => {
  assert.doesNotMatch(page, /pipelineScope !== 'all'[\s\S]{0,260}\|\| annotationDirty/);
  assert.doesNotMatch(page, /const stillCurrent = pipelineEnabled[\s\S]{0,260}!annotationDirty/);
  assert.doesNotMatch(page, /等待保存后自动预计算|保存当前标注后才会启动全库预计算/);
});

test('automatic precompute never opens the background activity dock', () => {
  const effect = page.match(/pipelineScope !== 'all'[\s\S]*?ensurePipelinePrecompute\([\s\S]*?\}, \[[^\]]*watchPipelineJobEvents[^\]]*\]\);/)?.[0] ?? '';
  assert.ok(effect);
  assert.doesNotMatch(effect, /setTaskStreamOpen/);
});
