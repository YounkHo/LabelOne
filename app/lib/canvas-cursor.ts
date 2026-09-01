export type CanvasCursorTool = 'select' | 'rect' | 'rotation' | 'polygon' | 'point' | 'line' | 'circle' | 'brush' | 'pan';
export type CanvasCursorMode = 'default' | 'select' | 'pan' | 'panning' | 'draw';
export type ResizeCursor = 'ew-resize' | 'ns-resize' | 'nwse-resize' | 'nesw-resize';

export function resolveCanvasCursorMode(
  tool: CanvasCursorTool,
  state: { temporaryPan: boolean; panning: boolean; drawingEnabled: boolean },
): CanvasCursorMode {
  if (state.panning) return 'panning';
  if (state.temporaryPan || tool === 'pan') return 'pan';
  if (tool === 'select') return 'select';
  return state.drawingEnabled ? 'draw' : 'default';
}

export function resolveResizeCursor(points: number[][], pointIndex: number): ResizeCursor {
  const point = points[pointIndex];
  if (!point || points.length < 2) return 'nwse-resize';
  const opposite = points.length === 4
    ? points[(pointIndex + 2) % 4]
    : points[pointIndex === 0 ? 1 : 0];
  if (!opposite) return 'nwse-resize';
  const angle = ((Math.atan2(opposite[1] - point[1], opposite[0] - point[0]) * 180 / Math.PI) + 180) % 180;
  if (angle < 22.5 || angle >= 157.5) return 'ew-resize';
  if (angle < 67.5) return 'nwse-resize';
  if (angle < 112.5) return 'ns-resize';
  return 'nesw-resize';
}
