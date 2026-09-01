import type { PipelineParameterSchema } from './contracts';

export type PipelineParameterControlKind = 'range' | 'checkbox' | 'enum' | 'number' | 'string';

export type PipelineParameterContext = {
  inputWidth?: number;
  inputHeight?: number;
  parameters?: Record<string, unknown>;
  schemas?: Record<string, PipelineParameterSchema>;
};

const exactNumericRoles = new Set([
  'region-x', 'region-y', 'region-width', 'region-height', 'target-width', 'target-height',
]);

export function pipelineParameterControlKind(schema: PipelineParameterSchema): PipelineParameterControlKind {
  if (schema.type === 'boolean') return 'checkbox';
  if (schema.enum?.length) return 'enum';
  const numeric = schema.type === 'integer' || schema.type === 'number';
  const control = schema['x-ui']?.control;
  if (numeric && (control === 'number' || exactNumericRoles.has(schema['x-ui']?.role ?? ''))) return 'number';
  if (numeric && control === 'slider' && Number.isFinite(schema.minimum) && Number.isFinite(schema.maximum)) return 'range';
  if (numeric && Number.isFinite(schema.minimum) && Number.isFinite(schema.maximum)) {
    const span = schema.maximum! - schema.minimum!;
    if (schema.type === 'integer' && span > 1000) return 'number';
    return 'range';
  }
  if (schema.type === 'integer' || schema.type === 'number') return 'number';
  return 'string';
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function parameterNameForRole(schemas: Record<string, PipelineParameterSchema>, role: string): string | undefined {
  return Object.entries(schemas).find(([, schema]) => schema['x-ui']?.role === role)?.[0];
}

export function pipelineParameterSchemaForContext(
  schema: PipelineParameterSchema,
  context?: PipelineParameterContext,
): PipelineParameterSchema {
  const role = schema['x-ui']?.role;
  if (!role || !context) return schema;
  const inputWidth = finiteNumber(context.inputWidth, 0);
  const inputHeight = finiteNumber(context.inputHeight, 0);
  const parameters = context.parameters ?? {};
  const schemas = context.schemas ?? {};
  const xName = parameterNameForRole(schemas, 'region-x');
  const yName = parameterNameForRole(schemas, 'region-y');
  const targetWidthName = parameterNameForRole(schemas, 'target-width');
  const targetHeightName = parameterNameForRole(schemas, 'target-height');
  let maximum = schema.maximum;
  if (role === 'region-x' && inputWidth > 0) maximum = Math.min(maximum ?? Infinity, Math.max(0, Math.floor(inputWidth) - 1));
  if (role === 'region-y' && inputHeight > 0) maximum = Math.min(maximum ?? Infinity, Math.max(0, Math.floor(inputHeight) - 1));
  if (role === 'region-width' && inputWidth > 0) maximum = Math.min(maximum ?? Infinity, Math.max(1, Math.floor(inputWidth - finiteNumber(xName ? parameters[xName] : undefined, 0))));
  if (role === 'region-height' && inputHeight > 0) maximum = Math.min(maximum ?? Infinity, Math.max(1, Math.floor(inputHeight - finiteNumber(yName ? parameters[yName] : undefined, 0))));
  if (role === 'target-width' && targetHeightName) {
    const targetHeight = finiteNumber(parameters[targetHeightName], 0);
    if (targetHeight > 0) maximum = Math.min(maximum ?? Infinity, Math.floor(64_000_000 / targetHeight));
  }
  if (role === 'target-height' && targetWidthName) {
    const targetWidth = finiteNumber(parameters[targetWidthName], 0);
    if (targetWidth > 0) maximum = Math.min(maximum ?? Infinity, Math.floor(64_000_000 / targetWidth));
  }
  return Number.isFinite(maximum) ? { ...schema, maximum } : schema;
}

export function clampPipelineNumericValue(raw: string | number, schema: PipelineParameterSchema): number {
  const parsed = typeof raw === 'number' ? raw : Number(raw);
  const fallback = typeof schema.default === 'number' && Number.isFinite(schema.default)
    ? schema.default
    : Number.isFinite(schema.minimum) ? schema.minimum!
      : Number.isFinite(schema.maximum) ? schema.maximum!
        : 0;
  let value = Number.isFinite(parsed) ? parsed : fallback;
  const minimum = Number.isFinite(schema.minimum)
    ? schema.type === 'integer' ? Math.ceil(schema.minimum!) : schema.minimum!
    : undefined;
  const maximum = Number.isFinite(schema.maximum)
    ? schema.type === 'integer' ? Math.floor(schema.maximum!) : schema.maximum!
    : undefined;
  if (schema.type === 'integer') value = Math.round(value);
  if (minimum !== undefined) value = Math.max(minimum, value);
  if (maximum !== undefined) value = Math.min(maximum, value);
  return value;
}

export function pipelineRangeStep(schema: PipelineParameterSchema): number {
  if (typeof schema.multipleOf === 'number' && Number.isFinite(schema.multipleOf) && schema.multipleOf > 0) return schema.multipleOf;
  if (schema.type === 'integer') return 1;
  if (!Number.isFinite(schema.minimum) || !Number.isFinite(schema.maximum)) return 0.01;
  return Math.max((schema.maximum! - schema.minimum!) / 200, 0.001);
}
