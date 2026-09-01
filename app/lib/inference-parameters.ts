import type { PipelineParameterSchema } from './contracts';

export type InferenceParameterSchema = Record<string, PipelineParameterSchema>;

export function normalizeInferenceParameterSchema(value: unknown): InferenceParameterSchema {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const properties = (value as { properties?: unknown }).properties;
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) return {};
  const entries: Array<[string, PipelineParameterSchema]> = [];
  for (const [name, raw] of Object.entries(properties).slice(0, 64)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name) || !raw || typeof raw !== 'object' || Array.isArray(raw)) continue;
    const candidate = raw as Record<string, unknown>;
    const type = candidate.type;
    if (!['integer', 'number', 'string', 'boolean'].includes(String(type))) continue;
    const schema: PipelineParameterSchema = { type: type as PipelineParameterSchema['type'] };
    if (typeof candidate.title === 'string') schema.title = candidate.title.slice(0, 128);
    if (typeof candidate.description === 'string') schema.description = candidate.description.slice(0, 512);
    if (typeof candidate.minimum === 'number' && Number.isFinite(candidate.minimum)) schema.minimum = candidate.minimum;
    if (typeof candidate.maximum === 'number' && Number.isFinite(candidate.maximum)) schema.maximum = candidate.maximum;
    if (Array.isArray(candidate.enum) && candidate.enum.length <= 64 && candidate.enum.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) {
      schema.enum = candidate.enum as Array<string | number | boolean>;
    }
    if ('default' in candidate) schema.default = candidate.default;
    entries.push([name, schema]);
  }
  return Object.fromEntries(entries);
}

export function inferenceParameterDefaults(schema: InferenceParameterSchema): Record<string, unknown> {
  return Object.fromEntries(Object.entries(schema).flatMap(([name, property]) => {
    if (property.default !== undefined) return [[name, property.default]];
    if (property.enum?.length) return [[name, property.enum[0]]];
    if (property.type === 'boolean') return [[name, false]];
    return [];
  }));
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, child]) => [key, canonical(child)]));
}

export function inferenceRequestSignature(value: unknown): string {
  return JSON.stringify(canonical(value));
}
