export type CanvasView = {
  scale: number;
  x: number;
  y: number;
};

export type CanvasPoint = {
  x: number;
  y: number;
};

export type CanvasWheelInput = {
  deltaX: number;
  deltaY: number;
  deltaMode: number;
  ctrlKey: boolean;
  metaKey: boolean;
};

export type CanvasKeyboardCommand = 'zoom-in' | 'zoom-out' | 'fit' | 'actual-size';

export type KeyboardZoomInput = {
  key: string;
  code?: string;
  ctrlKey: boolean;
  metaKey: boolean;
};

export const MIN_CANVAS_SCALE = 0.25;
export const CANVAS_ZOOM_STEP = 1.18;
export const MIN_SOURCE_PIXEL_GRID_SIZE = 8;
export const MAX_SOURCE_PIXEL_INSPECTION_SIZE = 32;

const WHEEL_LINE_PIXELS = 16;

export function clampCanvasScale(scale: number, maximumScale: number): number {
  return Math.max(MIN_CANVAS_SCALE, Math.min(maximumScale, scale));
}

export function canvasScaleFromPercent(percent: number, maximumScale: number, fallbackScale = 1): number {
  const requestedScale = Number.isFinite(percent) ? percent / 100 : fallbackScale;
  return clampCanvasScale(requestedScale, maximumScale);
}

export function sourcePixelScreenSize(sourceSize: number, surfaceSize: number, canvasScale: number): number {
  if (![sourceSize, surfaceSize, canvasScale].every((value) => Number.isFinite(value) && value > 0)) return 0;
  return surfaceSize * canvasScale / sourceSize;
}

export function canvasScaleForSourcePixelSize(sourceSize: number, surfaceSize: number, targetPixelSize: number): number {
  if (![sourceSize, surfaceSize, targetPixelSize].every((value) => Number.isFinite(value) && value > 0)) return 0;
  return targetPixelSize * sourceSize / surfaceSize;
}

export function maximumCanvasScaleForPixelInspection(sourceSize: number, surfaceSize: number, targetPixelSize: number, cap = 256, fallback = 8): number {
  const targetScale = canvasScaleForSourcePixelSize(sourceSize, surfaceSize, targetPixelSize);
  if (!targetScale) return fallback;
  return Math.min(cap, Math.max(1, targetScale));
}

export function isSourcePixelGridVisible(enabled: boolean, pixelWidth: number, pixelHeight: number, minimumSize = MIN_SOURCE_PIXEL_GRID_SIZE): boolean {
  return Boolean(enabled && Number.isFinite(pixelWidth) && Number.isFinite(pixelHeight) && pixelWidth >= minimumSize && pixelHeight >= minimumSize);
}

/** Zoom while keeping the same image point underneath the stage-space anchor. */
export function zoomCanvasView(
  view: CanvasView,
  factor: number,
  maximumScale: number,
  anchor: CanvasPoint = { x: 0, y: 0 },
): CanvasView {
  const nextScale = clampCanvasScale(view.scale * factor, maximumScale);
  if (nextScale === view.scale) return view;
  const ratio = nextScale / view.scale;
  return {
    scale: nextScale,
    x: anchor.x - (anchor.x - view.x) * ratio,
    y: anchor.y - (anchor.y - view.y) * ratio,
  };
}

export function panCanvasView(view: CanvasView, deltaX: number, deltaY: number): CanvasView {
  return { ...view, x: view.x - deltaX, y: view.y - deltaY };
}

export function normalizeWheelDelta(delta: number, deltaMode: number, pagePixels = 800): number {
  if (deltaMode === 1) return delta * WHEEL_LINE_PIXELS;
  if (deltaMode === 2) return delta * pagePixels;
  return delta;
}

export function applyCanvasWheel(
  view: CanvasView,
  input: CanvasWheelInput,
  maximumScale: number,
  anchor: CanvasPoint,
  pagePixels = 800,
): { view: CanvasView; mode: 'zoom' | 'pan' } {
  if (input.ctrlKey || input.metaKey) {
    const deltaY = normalizeWheelDelta(input.deltaY, input.deltaMode, pagePixels);
    return {
      mode: 'zoom',
      view: zoomCanvasView(view, Math.exp(-deltaY * 0.002), maximumScale, anchor),
    };
  }
  return {
    mode: 'pan',
    view: panCanvasView(
      view,
      normalizeWheelDelta(input.deltaX, input.deltaMode, pagePixels),
      normalizeWheelDelta(input.deltaY, input.deltaMode, pagePixels),
    ),
  };
}

export function resolveCanvasKeyboardCommand(key: string, code = ''): CanvasKeyboardCommand | null {
  if (key === '+' || key === '=' || code === 'NumpadAdd') return 'zoom-in';
  if (key === '-' || key === '_' || code === 'NumpadSubtract') return 'zoom-out';
  if (key === '0' || code === 'Numpad0') return 'fit';
  if (key === '1' || code === 'Numpad1') return 'actual-size';
  return null;
}

export function isBrowserZoomKeyboardShortcut(input: KeyboardZoomInput): boolean {
  if (!input.ctrlKey && !input.metaKey) return false;
  const command = resolveCanvasKeyboardCommand(input.key, input.code);
  return command === 'zoom-in' || command === 'zoom-out' || command === 'fit';
}
