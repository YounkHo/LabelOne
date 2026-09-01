export type CanvasLabelPlacement = 'above' | 'inside' | 'below' | 'left' | 'right' | 'clamped' | 'sticky';
export type CanvasVisibleBounds = { left: number; top: number; right: number; bottom: number };
export type CanvasViewportView = { scale: number; x: number; y: number };

export type CanvasLabelLayout = {
  unit: number;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
  paddingX: number;
  radius: number;
  placement: CanvasLabelPlacement;
};

export type CanvasLabelRequest = {
  points: number[][];
  text: string;
  priority?: number;
  anchor?: { x: number; y: number; align?: 'start' | 'center' };
};

export type CanvasLabelCollisionPolicy = {
  maxOffsetSlots?: number;
  hideWhenCrowded?: boolean;
};

export function canvasLabelTopVertexAnchor(points: number[][]): CanvasLabelRequest['anchor'] | undefined {
  const valid = points.filter((point) => point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]));
  if (!valid.length) return undefined;
  const top = valid.reduce((best, point) => point[1] < best[1] || (point[1] === best[1] && point[0] < best[0]) ? point : best);
  return { x: top[0], y: top[1], align: 'center' };
}

const clamp = (value: number, minimum: number, maximum: number) => Math.max(minimum, Math.min(maximum, value));

export function canvasAnnotationOpticalScale(canvasScale: number): number {
  if (!Number.isFinite(canvasScale) || canvasScale <= 1) return 1;
  return clamp(1 / Math.pow(canvasScale, 0.16), 0.72, 1);
}

export function canvasLabelOpacity(canvasScale: number, selected = false): number {
  if (!Number.isFinite(canvasScale) || canvasScale <= 0) return 0;
  const hiddenAt = selected ? 0.28 : 0.48;
  const fullyVisibleAt = selected ? 0.46 : 0.72;
  const progress = clamp((canvasScale - hiddenAt) / (fullyVisibleAt - hiddenAt), 0, 1);
  return progress * progress * (3 - 2 * progress);
}

export const CANVAS_CONTROL_POINT_RADIUS_PX = 7;
export const CANVAS_VERTEX_CONTROL_DIAMETER_PX = 14;

function labelOverlapArea(candidate: CanvasLabelLayout, occupied: CanvasLabelLayout, padding: number): number {
  const left = Math.max(candidate.x, occupied.x - padding);
  const top = Math.max(candidate.y, occupied.y - padding);
  const right = Math.min(candidate.x + candidate.width, occupied.x + occupied.width + padding);
  const bottom = Math.min(candidate.y + candidate.height, occupied.y + occupied.height + padding);
  return Math.max(0, right - left) * Math.max(0, bottom - top);
}

function glyphWidth(char: string): number {
  if (/\p{Extended_Pictographic}|[\u2e80-\u9fff\uff01-\uff60]/u.test(char)) return 1;
  if (/[ilI1.,:;'·\s]/.test(char)) return 0.36;
  if (/[MW@%#]/.test(char)) return 0.9;
  return 0.62;
}

function estimatedTextWidth(value: string, fontSize: number): number {
  return Array.from(value).reduce((width, char) => width + glyphWidth(char) * fontSize, 0);
}

export function truncateCanvasLabel(value: string, maximumGlyphs = 32): string {
  const glyphs = Array.from(value);
  return glyphs.length <= maximumGlyphs ? value : `${glyphs.slice(0, maximumGlyphs - 1).join('')}…`;
}

export function fitCanvasLabelText(value: string, maximumLabelWidth = 220, fontSize = 11, horizontalPadding = 10): string {
  const available = Math.max(0, maximumLabelWidth - horizontalPadding);
  if (estimatedTextWidth(value, fontSize) <= available) return value;
  const ellipsisWidth = estimatedTextWidth('…', fontSize);
  let used = 0;
  const result: string[] = [];
  for (const char of Array.from(value)) {
    const advance = glyphWidth(char) * fontSize;
    if (used + advance + ellipsisWidth > available) break;
    result.push(char);
    used += advance;
  }
  return `${result.join('')}…`;
}

export function canvasVisibleImageBounds(
  sourceWidth: number,
  sourceHeight: number,
  surfaceWidth: number,
  surfaceHeight: number,
  viewportWidth: number,
  viewportHeight: number,
  view: CanvasViewportView,
): CanvasVisibleBounds | null {
  if (![sourceWidth, sourceHeight, surfaceWidth, surfaceHeight, viewportWidth, viewportHeight, view.scale].every((value) => Number.isFinite(value) && value > 0) || !Number.isFinite(view.x) || !Number.isFinite(view.y)) return null;
  const scaledWidth = surfaceWidth * view.scale;
  const scaledHeight = surfaceHeight * view.scale;
  const imageLeft = viewportWidth / 2 - scaledWidth / 2 + view.x;
  const imageTop = viewportHeight / 2 - scaledHeight / 2 + view.y;
  const left = clamp(-imageLeft / scaledWidth * sourceWidth, 0, sourceWidth);
  const top = clamp(-imageTop / scaledHeight * sourceHeight, 0, sourceHeight);
  const right = clamp((viewportWidth - imageLeft) / scaledWidth * sourceWidth, 0, sourceWidth);
  const bottom = clamp((viewportHeight - imageTop) / scaledHeight * sourceHeight, 0, sourceHeight);
  return right >= left && bottom >= top ? { left, top, right, bottom } : null;
}

export function canvasLabelLayout(
  points: number[][],
  text: string,
  sourceWidth: number,
  sourceHeight: number,
  imageUnitsPerScreenPixel: number,
  visibleBounds?: CanvasVisibleBounds,
  stickySlot = 0,
  stickyTopInsetPx = 52,
  occupiedLayouts: CanvasLabelLayout[] = [],
  anchor?: CanvasLabelRequest['anchor'],
  collisionPolicy: CanvasLabelCollisionPolicy = {},
): CanvasLabelLayout | null {
  if (!points.length || !Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight) || sourceWidth <= 0 || sourceHeight <= 0 || !Number.isFinite(imageUnitsPerScreenPixel) || imageUnitsPerScreenPixel <= 0) return null;
  const validPoints = points.filter((point) => point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]));
  if (!validPoints.length) return null;

  const unit = imageUnitsPerScreenPixel;
  const fontSize = 11 * unit;
  const paddingX = 5 * unit;
  const height = 19 * unit;
  const gap = 3 * unit;
  const bounds = visibleBounds
    ? {
      left: clamp(visibleBounds.left, 0, sourceWidth),
      top: clamp(visibleBounds.top, 0, sourceHeight),
      right: clamp(visibleBounds.right, 0, sourceWidth),
      bottom: clamp(visibleBounds.bottom, 0, sourceHeight),
    }
    : { left: 0, top: 0, right: sourceWidth, bottom: sourceHeight };
  if (bounds.right < bounds.left || bounds.bottom < bounds.top) return null;
  const minX = Math.min(...validPoints.map((point) => point[0]));
  const minY = Math.min(...validPoints.map((point) => point[1]));
  const maxX = Math.max(...validPoints.map((point) => point[0]));
  const maxY = Math.max(...validPoints.map((point) => point[1]));
  if (maxX < bounds.left || minX > bounds.right || maxY < bounds.top || minY > bounds.bottom) return null;
  const visibleWidth = bounds.right - bounds.left;
  const visibleHeight = bounds.bottom - bounds.top;
  const margin = Math.min(4 * unit, visibleWidth / 2, visibleHeight / 2);
  const desiredWidth = clamp(estimatedTextWidth(text, 11) + 10, 36, 220) * unit;
  const availableWidth = Math.max(0, visibleWidth - margin * 2);
  const width = Math.min(desiredWidth, availableWidth || visibleWidth);
  const minimumX = bounds.left + margin;
  const maximumX = Math.max(minimumX, bounds.right - margin - width);
  const requestedAnchorX = anchor && Number.isFinite(anchor.x) ? anchor.x : minX;
  const requestedAnchorY = anchor && Number.isFinite(anchor.y) ? anchor.y : minY;
  const desiredX = requestedAnchorX - (anchor?.align === 'center' ? width / 2 : 0);
  const x = clamp(desiredX, minimumX, maximumX);
  const anchorY = Math.max(requestedAnchorY, bounds.top);
  const sticky = Boolean(visibleBounds && (minX < bounds.left || minY < bounds.top || maxX > bounds.right || maxY > bounds.bottom));
  const defaultMinimumY = bounds.top + margin;
  const availableMaximumY = Math.max(defaultMinimumY, bounds.bottom - margin - height);
  const slotHeight = height + gap;
  const availableSlots = Math.max(1, Math.floor(Math.max(0, availableMaximumY - bounds.top - stickyTopInsetPx * unit) / slotHeight) + 1);
  const normalizedSlot = Math.max(0, Math.floor(stickySlot)) % availableSlots;
  const minimumY = sticky
    ? clamp(bounds.top + stickyTopInsetPx * unit + normalizedSlot * slotHeight, defaultMinimumY, availableMaximumY)
    : defaultMinimumY;
  const maximumY = Math.max(minimumY, availableMaximumY);
  const aboveY = anchorY - gap - height;
  const insideY = Math.max(anchorY + gap, minimumY);
  const preferred = aboveY >= minimumY
    ? { x, y: aboveY, placement: sticky ? 'sticky' : 'above' as CanvasLabelPlacement }
    : insideY + height <= bounds.bottom - margin
      ? { x, y: insideY, placement: sticky ? 'sticky' : 'inside' as CanvasLabelPlacement }
      : { x, y: clamp(aboveY, minimumY, maximumY), placement: sticky ? 'sticky' : 'clamped' as CanvasLabelPlacement };
  const radius = 4 * unit;
  const candidates: CanvasLabelLayout[] = [];
  const seen = new Set<string>();
  const addCandidate = (candidateX: number, candidateY: number, placement: CanvasLabelPlacement) => {
    if (candidateY < defaultMinimumY || candidateY > availableMaximumY) return;
    const nextX = clamp(candidateX, minimumX, maximumX);
    const key = `${nextX.toFixed(4)}:${candidateY.toFixed(4)}`;
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push({ unit, x: nextX, y: candidateY, width, height, fontSize, paddingX, radius, placement: sticky ? 'sticky' : placement });
  };
  addCandidate(preferred.x, preferred.y, preferred.placement);
  const rightAlignedX = clamp(maxX - width, minimumX, maximumX);
  const belowY = maxY + gap;
  addCandidate(x, belowY, 'below');
  addCandidate(rightAlignedX, aboveY, 'above');
  addCandidate(rightAlignedX, belowY, 'below');
  addCandidate(x, Math.max(minY + gap, defaultMinimumY), 'inside');
  addCandidate(rightAlignedX, Math.max(minY + gap, defaultMinimumY), 'inside');
  addCandidate(x, Math.min(maxY - height - gap, availableMaximumY), 'inside');
  addCandidate(rightAlignedX, Math.min(maxY - height - gap, availableMaximumY), 'inside');
  addCandidate(maxX + gap, clamp(minY, defaultMinimumY, availableMaximumY), 'right');
  addCandidate(minX - gap - width, clamp(minY, defaultMinimumY, availableMaximumY), 'left');
  const maxOffsetSlots = Math.max(0, Math.floor(collisionPolicy.maxOffsetSlots ?? 4));
  for (let offset = 1; offset <= maxOffsetSlots; offset += 1) {
    addCandidate(preferred.x, preferred.y - offset * slotHeight, preferred.placement);
    addCandidate(preferred.x, preferred.y + offset * slotHeight, preferred.placement);
  }
  if (!candidates.length) candidates.push({ unit, x, y: clamp(aboveY, minimumY, maximumY), width, height, fontSize, paddingX, radius, placement: sticky ? 'sticky' : 'clamped' });
  if (!occupiedLayouts.length) return candidates[0];
  const collisionPadding = 2 * unit;
  const collisionFree = candidates.find((candidate) => occupiedLayouts.every((occupied) => labelOverlapArea(candidate, occupied, collisionPadding) === 0));
  if (collisionFree) return collisionFree;
  if (collisionPolicy.hideWhenCrowded) return null;
  return candidates.reduce((best, candidate) => {
    const score = occupiedLayouts.reduce((total, occupied) => total + labelOverlapArea(candidate, occupied, collisionPadding), 0);
    const bestScore = occupiedLayouts.reduce((total, occupied) => total + labelOverlapArea(best, occupied, collisionPadding), 0);
    return score < bestScore ? candidate : best;
  });
}

export function canvasLabelLayouts(
  requests: CanvasLabelRequest[],
  sourceWidth: number,
  sourceHeight: number,
  imageUnitsPerScreenPixel: number,
  visibleBounds?: CanvasVisibleBounds,
): Array<CanvasLabelLayout | null> {
  const results: Array<CanvasLabelLayout | null> = Array.from({ length: requests.length }, () => null);
  const occupied: CanvasLabelLayout[] = [];
  const ordered = requests.map((request, index) => ({ request, index })).sort((left, right) => (right.request.priority ?? 0) - (left.request.priority ?? 0) || left.index - right.index);
  ordered.forEach(({ request, index }) => {
    const selected = (request.priority ?? 0) >= 2;
    const layout = canvasLabelLayout(request.points, request.text, sourceWidth, sourceHeight, imageUnitsPerScreenPixel, visibleBounds, selected ? occupied.length : 0, 52, occupied, request.anchor, { maxOffsetSlots: 1, hideWhenCrowded: !selected });
    results[index] = layout;
    if (layout) occupied.push(layout);
  });
  return results;
}
