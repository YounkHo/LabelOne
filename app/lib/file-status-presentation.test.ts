import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../page.tsx', import.meta.url), 'utf8');

test('file-row progress selection follows the live pipeline toggle', () => {
  assert.match(page, /const fileProgressJob = selectFileProgressJob\([\s\S]*?\{ pipelineEnabled, pipelineNodes: pipelineRequestNodes \},\s*\);/);
  assert.match(page, /const jobItem = fileProgressJob \? backend\.jobItemSnapshots\[`\$\{fileProgressJob\.job_id\}:\$\{file\.id\}`\] : undefined/);
  assert.match(page, /resolveFileStatusIndicator\(file\.annotationFileExists, file\.annotations, fileProgressJob\?\.kind, jobItem\)/);
});

test('only the right status slot owns the file progress tooltip', () => {
  const row = page.match(/return <button className=\{`file-row[\s\S]*?<\/button>;/)?.[0];
  assert.ok(row);
  const openingTag = row.match(/^return <button[\s\S]*?>/)?.[0] ?? '';
  assert.doesNotMatch(openingTag, /data-tooltip|title=/);
  assert.match(row, /className="file-status-slot" data-tooltip=\{statusView\.kind !== 'empty' \? statusView\.label : undefined\} data-tooltip-anchor="pointer"/);
  assert.doesNotMatch(row, /title=\{statusView\.label\}/);
});
