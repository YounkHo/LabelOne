import { useRef, useState } from 'react';

import { CustomSelect } from './custom-select';
import type { PipelineParameterSchema } from '../lib/contracts';
import { clampPipelineNumericValue, pipelineParameterControlKind, pipelineParameterSchemaForContext, pipelineRangeStep, type PipelineParameterContext } from '../lib/pipeline-parameters';

export function PipelineParameterControl({
  name,
  label,
  schema,
  value,
  context,
  onChange,
}: {
  name: string;
  label: string;
  schema: PipelineParameterSchema;
  value: unknown;
  context?: PipelineParameterContext;
  onChange: (name: string, value: unknown) => void;
}) {
  const effectiveSchema = pipelineParameterSchemaForContext(schema, context);
  const kind = pipelineParameterControlKind(effectiveSchema);
  const unit = effectiveSchema['x-ui']?.unit;
  const externalNumberText = value === undefined ? '' : String(value);
  const [numberDraftState, setNumberDraftState] = useState({ external: externalNumberText, value: externalNumberText });
  const cancelNumberCommitRef = useRef(false);
  const numberDraft = numberDraftState.external === externalNumberText ? numberDraftState.value : externalNumberText;
  const setNumberDraft = (next: string) => setNumberDraftState({ external: externalNumberText, value: next });
  const commitNumber = () => {
    if (cancelNumberCommitRef.current) {
      cancelNumberCommitRef.current = false;
      return;
    }
    if (numberDraft.trim() === '') {
      onChange(name, undefined);
      return;
    }
    const normalized = clampPipelineNumericValue(numberDraft, effectiveSchema);
    setNumberDraft(String(normalized));
    onChange(name, normalized);
  };
  const heading = <span className="pipeline-parameter-heading"><strong>{label}</strong><small>{effectiveSchema.description ?? '该参数暂未提供说明。'}</small></span>;
  if (kind === 'enum') {
    const selectedIndex = effectiveSchema.enum!.findIndex((candidate) => Object.is(candidate, value));
    return <label className="pipeline-parameter-control enum">{heading}<CustomSelect ariaLabel={label} value={selectedIndex >= 0 ? String(selectedIndex) : ''} placeholder="请选择" options={effectiveSchema.enum!.map((option, index) => ({ value: String(index), label: String(option) }))} onChange={(index) => onChange(name, effectiveSchema.enum![Number(index)])} /></label>;
  }
  if (kind === 'checkbox') return <label className="pipeline-parameter-control checkbox">{heading}<input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(name, event.target.checked)} /></label>;
  if (kind === 'range') {
    const normalized = clampPipelineNumericValue(typeof value === 'number' ? value : Number(value), effectiveSchema);
    return <label className="pipeline-parameter-control range">{heading}<input type="range" min={effectiveSchema.minimum} max={effectiveSchema.maximum} step={pipelineRangeStep(effectiveSchema)} value={normalized} onChange={(event) => onChange(name, clampPipelineNumericValue(event.target.value, effectiveSchema))} /><b>{Number.isInteger(normalized) ? normalized : Number(normalized.toFixed(3))}{unit ? ` ${unit}` : ''}</b></label>;
  }
  if (kind === 'number') return <label className="pipeline-parameter-control number">{heading}<span className="pipeline-parameter-number-field"><input type="number" inputMode={effectiveSchema.type === 'integer' ? 'numeric' : 'decimal'} value={numberDraft} min={effectiveSchema.minimum} max={effectiveSchema.maximum} step={effectiveSchema.multipleOf ?? (effectiveSchema.type === 'integer' ? 1 : 'any')} placeholder={effectiveSchema.default === undefined ? '可选' : String(effectiveSchema.default)} onChange={(event) => setNumberDraft(event.target.value)} onBlur={commitNumber} onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); else if (event.key === 'Escape') { cancelNumberCommitRef.current = true; setNumberDraft(externalNumberText); event.currentTarget.blur(); } }} />{unit && <em>{unit}</em>}</span></label>;
  return <label className="pipeline-parameter-control string">{heading}<input value={value === undefined ? '' : String(value)} placeholder={effectiveSchema.default === undefined ? '可选' : String(effectiveSchema.default)} onChange={(event) => onChange(name, event.target.value === '' ? undefined : event.target.value)} /></label>;
}
