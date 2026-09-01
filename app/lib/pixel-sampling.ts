export type PixelSample = {
  r: number;
  g: number;
  b: number;
  a: number;
  v: number;
};

export type DisplayRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export function pixelValue(r: number, g: number, b: number): number {
  return Math.max(r, g, b);
}

export function pixelSampleFromRgba(data: ArrayLike<number>): PixelSample {
  const r = Math.max(0, Math.min(255, Math.round(data[0] ?? 0)));
  const g = Math.max(0, Math.min(255, Math.round(data[1] ?? 0)));
  const b = Math.max(0, Math.min(255, Math.round(data[2] ?? 0)));
  const a = Math.max(0, Math.min(255, Math.round(data[3] ?? 255)));
  return { r, g, b, a, v: pixelValue(r, g, b) };
}

export function sourcePixelAtDisplayPoint(
  clientX: number,
  clientY: number,
  rect: DisplayRect,
  naturalWidth: number,
  naturalHeight: number,
): { x: number; y: number } | null {
  if (rect.width <= 0 || rect.height <= 0 || naturalWidth <= 0 || naturalHeight <= 0) return null;
  if (clientX < rect.left || clientY < rect.top || clientX > rect.left + rect.width || clientY > rect.top + rect.height) return null;
  return {
    x: Math.min(naturalWidth - 1, Math.max(0, Math.floor((clientX - rect.left) / rect.width * naturalWidth))),
    y: Math.min(naturalHeight - 1, Math.max(0, Math.floor((clientY - rect.top) / rect.height * naturalHeight))),
  };
}
