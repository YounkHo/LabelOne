export type AnnotationHistory<T> = {
  past: T[];
  present: T;
  future: T[];
};

export const ANNOTATION_HISTORY_CAPACITY = 100;

export function cloneAnnotationValue<T>(value: T): T {
  return structuredClone(value);
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, child]) => [key, canonicalize(child)]));
}

export function annotationFingerprint(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

export function createAnnotationHistory<T>(present: T): AnnotationHistory<T> {
  return { past: [], present: cloneAnnotationValue(present), future: [] };
}

export function commitAnnotationHistory<T>(history: AnnotationHistory<T>, next: T, capacity = ANNOTATION_HISTORY_CAPACITY): AnnotationHistory<T> {
  if (annotationFingerprint(history.present) === annotationFingerprint(next)) return history;
  const boundedCapacity = Math.max(100, Math.floor(capacity));
  return {
    past: [...history.past, cloneAnnotationValue(history.present)].slice(-boundedCapacity),
    present: cloneAnnotationValue(next),
    future: [],
  };
}

export function undoAnnotationHistory<T>(history: AnnotationHistory<T>): AnnotationHistory<T> {
  const previous = history.past.at(-1);
  if (previous === undefined) return history;
  return {
    past: history.past.slice(0, -1),
    present: cloneAnnotationValue(previous),
    future: [cloneAnnotationValue(history.present), ...history.future],
  };
}

export function redoAnnotationHistory<T>(history: AnnotationHistory<T>): AnnotationHistory<T> {
  const next = history.future[0];
  if (next === undefined) return history;
  return {
    past: [...history.past, cloneAnnotationValue(history.present)],
    present: cloneAnnotationValue(next),
    future: history.future.slice(1),
  };
}

export function annotationDraftKey(datasetId: string, assetId: string): string {
  return `${datasetId}:${assetId}`;
}
