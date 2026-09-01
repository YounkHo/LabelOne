export const MAX_ANNOTATION_LABEL_LENGTH = 128;

export type ScreenPoint = { x: number; y: number };
export type FloatingMenuPosition = ScreenPoint & { transformOrigin: string };

export function normalizeAnnotationLabel(rawLabel: string): string | null {
  const label = rawLabel.trim();
  return label && label.length <= MAX_ANNOTATION_LABEL_LENGTH ? label : null;
}

export function buildAnnotationLabelChoices(labels: string[], preferredLabel = '', limit = 24): string[] {
  const counts = new Map<string, number>();
  for (const rawLabel of labels) {
    const label = normalizeAnnotationLabel(rawLabel);
    if (label) counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const preferred = normalizeAnnotationLabel(preferredLabel);
  const sorted = [...counts].sort(([left, leftCount], [right, rightCount]) => rightCount - leftCount || left.localeCompare(right, 'zh-Hans-CN')).map(([label]) => label);
  const choices = preferred ? [preferred, ...sorted.filter((label) => label !== preferred)] : sorted;
  return choices.slice(0, Math.max(1, Math.min(100, Math.floor(limit))));
}

export function positionFloatingLabelMenu(
  anchor: ScreenPoint,
  viewport: { width: number; height: number },
  menu = { width: 320, height: 360 },
  gap = 10,
  margin = 8,
): FloatingMenuPosition {
  const anchorX = Number.isFinite(anchor.x) ? anchor.x : margin;
  const anchorY = Number.isFinite(anchor.y) ? anchor.y : margin;
  const openLeft = anchorX + gap + menu.width > viewport.width - margin && anchorX - gap - menu.width >= margin;
  const openAbove = anchorY + gap + menu.height > viewport.height - margin && anchorY - gap - menu.height >= margin;
  const desiredX = openLeft ? anchorX - gap - menu.width : anchorX + gap;
  const desiredY = openAbove ? anchorY - gap - menu.height : anchorY + gap;
  const maxX = Math.max(margin, viewport.width - margin - menu.width);
  const maxY = Math.max(margin, viewport.height - margin - menu.height);
  return {
    x: Math.max(margin, Math.min(maxX, desiredX)),
    y: Math.max(margin, Math.min(maxY, desiredY)),
    transformOrigin: `${openLeft ? 'right' : 'left'} ${openAbove ? 'bottom' : 'top'}`,
  };
}
