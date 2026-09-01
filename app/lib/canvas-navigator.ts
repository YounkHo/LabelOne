export type CanvasNavigatorView = { scale: number; x: number; y: number };

export type CanvasNavigatorMetrics = {
  navigatorWidth: number;
  navigatorHeight: number;
  imageWidth: number;
  imageHeight: number;
  viewportWidth: number;
  viewportHeight: number;
};

export type NavigatorViewport = { left: number; top: number; width: number; height: number };

const clamp = (value: number, minimum: number, maximum: number) => Math.max(minimum, Math.min(maximum, value));

function fittedImage(metrics: CanvasNavigatorMetrics) {
  const fit = Math.min(metrics.navigatorWidth / metrics.imageWidth, metrics.navigatorHeight / metrics.imageHeight);
  const width = metrics.imageWidth * fit;
  const height = metrics.imageHeight * fit;
  return { left: (metrics.navigatorWidth - width) / 2, top: (metrics.navigatorHeight - height) / 2, width, height };
}

function valid(metrics: CanvasNavigatorMetrics, scale: number) {
  return [metrics.navigatorWidth, metrics.navigatorHeight, metrics.imageWidth, metrics.imageHeight, metrics.viewportWidth, metrics.viewportHeight, scale]
    .every((value) => Number.isFinite(value) && value > 0);
}

export function navigatorViewport(view: CanvasNavigatorView, metrics: CanvasNavigatorMetrics, minimumSize = 8): NavigatorViewport {
  if (!valid(metrics, view.scale)) return { left: 0, top: 0, width: 0, height: 0 };
  const fitted = fittedImage(metrics);
  const scaledWidth = metrics.imageWidth * view.scale;
  const scaledHeight = metrics.imageHeight * view.scale;
  const width = Math.min(fitted.width, Math.max(Math.min(minimumSize, fitted.width), fitted.width * Math.min(1, metrics.viewportWidth / scaledWidth)));
  const height = Math.min(fitted.height, Math.max(Math.min(minimumSize, fitted.height), fitted.height * Math.min(1, metrics.viewportHeight / scaledHeight)));
  const centerX = fitted.width * (0.5 - view.x / scaledWidth);
  const centerY = fitted.height * (0.5 - view.y / scaledHeight);
  return {
    left: fitted.left + clamp(centerX - width / 2, 0, fitted.width - width),
    top: fitted.top + clamp(centerY - height / 2, 0, fitted.height - height),
    width,
    height,
  };
}

export function navigatorPointToView(
  point: { x: number; y: number },
  view: CanvasNavigatorView,
  metrics: CanvasNavigatorMetrics,
): CanvasNavigatorView {
  if (!valid(metrics, view.scale)) return view;
  const fitted = fittedImage(metrics);
  const normalizedX = clamp((point.x - fitted.left) / fitted.width, 0, 1);
  const normalizedY = clamp((point.y - fitted.top) / fitted.height, 0, 1);
  const scaledWidth = metrics.imageWidth * view.scale;
  const scaledHeight = metrics.imageHeight * view.scale;
  const maximumX = Math.max(0, (scaledWidth - metrics.viewportWidth) / 2);
  const maximumY = Math.max(0, (scaledHeight - metrics.viewportHeight) / 2);
  return {
    scale: view.scale,
    x: maximumX === 0 ? 0 : clamp((0.5 - normalizedX) * scaledWidth, -maximumX, maximumX),
    y: maximumY === 0 ? 0 : clamp((0.5 - normalizedY) * scaledHeight, -maximumY, maximumY),
  };
}
