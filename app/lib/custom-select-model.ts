export type SelectModelOption = { disabled?: boolean };

export function nextEnabledOption(options: SelectModelOption[], current: number, direction: 1 | -1): number {
  if (!options.length || options.every((option) => option.disabled)) return -1;
  let index = current;
  for (let step = 0; step < options.length; step += 1) {
    index = (index + direction + options.length) % options.length;
    if (!options[index].disabled) return index;
  }
  return -1;
}

export function selectMenuPlacement(
  trigger: { top: number; bottom: number },
  viewportHeight: number,
  desiredHeight: number,
  gap = 5,
): { placement: 'top' | 'bottom'; top: number; maxHeight: number } {
  const below = Math.max(0, viewportHeight - trigger.bottom - gap);
  const above = Math.max(0, trigger.top - gap);
  const placement = below >= Math.min(180, desiredHeight) || below >= above ? 'bottom' : 'top';
  const available = placement === 'bottom' ? below : above;
  const maxHeight = Math.max(48, Math.min(desiredHeight, available));
  return {
    placement,
    top: placement === 'bottom' ? trigger.bottom + gap : trigger.top - gap - maxHeight,
    maxHeight,
  };
}
