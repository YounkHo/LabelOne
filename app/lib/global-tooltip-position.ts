export type TooltipRect = { left: number; top: number; right: number; bottom: number; width: number; height: number };

export type TooltipPosition = { left: number; top: number; placement: 'top' | 'bottom' };

export function positionGlobalTooltip(
  anchor: TooltipRect,
  tooltip: { width: number; height: number },
  viewport: { width: number; height: number },
  gap = 8,
  margin = 8,
): TooltipPosition {
  const maxLeft = Math.max(margin, viewport.width - tooltip.width - margin);
  const left = Math.max(margin, Math.min(maxLeft, anchor.left + anchor.width / 2 - tooltip.width / 2));
  const fitsBelow = anchor.bottom + gap + tooltip.height <= viewport.height - margin;
  const fitsAbove = anchor.top - gap - tooltip.height >= margin;
  const placement = fitsBelow || !fitsAbove ? 'bottom' : 'top';
  const rawTop = placement === 'bottom' ? anchor.bottom + gap : anchor.top - gap - tooltip.height;
  const maxTop = Math.max(margin, viewport.height - tooltip.height - margin);
  return { left, top: Math.max(margin, Math.min(maxTop, rawTop)), placement };
}
