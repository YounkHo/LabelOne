export type PipelineInsertAnchorRect = {
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
};

export type PipelineInsertPopoverPosition = {
  left: number;
  top: number;
  width: number;
  maxHeight: number;
  placement: 'top' | 'bottom';
};

const clamp = (value: number, minimum: number, maximum: number) => Math.max(minimum, Math.min(maximum, value));

export function pipelineInsertPopoverPosition(
  anchor: PipelineInsertAnchorRect,
  viewportWidth: number,
  viewportHeight: number,
  desiredWidth = 288,
  desiredHeight = 244,
): PipelineInsertPopoverPosition {
  const margin = 8;
  const gap = 6;
  const width = Math.max(120, Math.min(desiredWidth, Math.max(120, viewportWidth - margin * 2)));
  const left = clamp(anchor.left + anchor.width / 2 - width / 2, margin, Math.max(margin, viewportWidth - width - margin));
  const below = Math.max(0, viewportHeight - anchor.bottom - gap - margin);
  const above = Math.max(0, anchor.top - gap - margin);
  const comfortableHeight = Math.min(160, desiredHeight);
  const placement = below >= comfortableHeight || below >= above ? 'bottom' : 'top';
  const availableHeight = placement === 'bottom' ? below : above;
  const maxHeight = Math.max(48, Math.min(desiredHeight, availableHeight));
  const top = placement === 'bottom'
    ? clamp(anchor.bottom + gap, margin, Math.max(margin, viewportHeight - maxHeight - margin))
    : clamp(anchor.top - gap - maxHeight, margin, Math.max(margin, viewportHeight - maxHeight - margin));
  return { left, top, width, maxHeight, placement };
}
