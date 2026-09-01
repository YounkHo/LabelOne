export type AnnotationCategoryColors = {
  stroke: string;
  fill: string;
  labelBackground: string;
};

export const DEFAULT_ANNOTATION_CATEGORY_STROKES = [
  '#6fc2e8',
  '#e6b566',
  '#a995e8',
  '#75c98e',
  '#df8fb3',
  '#7aa9ed',
  '#e28d61',
  '#5dbcb1',
  '#e47d77',
  '#a8c85f',
  '#879bd8',
  '#cf91d1',
];

function stableCategoryHash(label: string): number {
  let hash = 2166136261;
  for (const character of label.trim()) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function normalizeAnnotationCategoryColor(color: string | null | undefined): string | null {
  const normalized = color?.trim().toLowerCase() ?? '';
  return /^#[0-9a-f]{6}$/.test(normalized) ? normalized : null;
}

function mixHexColors(foreground: string, background: string, foregroundWeight: number): string {
  const channel = (color: string, offset: number) => Number.parseInt(color.slice(offset, offset + 2), 16);
  const mixed = [1, 3, 5].map((offset) => Math.round(
    channel(foreground, offset) * foregroundWeight + channel(background, offset) * (1 - foregroundWeight),
  ));
  return `#${mixed.map((value) => value.toString(16).padStart(2, '0')).join('')}`;
}

export function annotationCategoryColors(label: string, customStroke?: string | null): AnnotationCategoryColors {
  const normalizedLabel = label.trim().normalize('NFC') || '未命名';
  const fallbackStroke = DEFAULT_ANNOTATION_CATEGORY_STROKES[stableCategoryHash(normalizedLabel) % DEFAULT_ANNOTATION_CATEGORY_STROKES.length];
  const normalizedCustomStroke = normalizeAnnotationCategoryColor(customStroke);
  const stroke = normalizedCustomStroke ?? fallbackStroke;
  return {
    stroke,
    fill: `${stroke}30`,
    labelBackground: mixHexColors(stroke, '#090f16', normalizedCustomStroke ? 0.34 : 0.32),
  };
}
