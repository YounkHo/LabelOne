export type ActiveListScrollInput = {
  activeIndex: number;
  currentScrollTop: number;
  itemCount: number;
  rowHeight: number;
  viewportHeight: number;
};

export function scrollTopForActiveListItem({
  activeIndex,
  currentScrollTop,
  itemCount,
  rowHeight,
  viewportHeight,
}: ActiveListScrollInput): number {
  if (!Number.isInteger(activeIndex) || activeIndex < 0 || activeIndex >= itemCount || rowHeight <= 0 || viewportHeight <= 0) return currentScrollTop;
  const maximumScrollTop = Math.max(0, itemCount * rowHeight - viewportHeight);
  const boundedScrollTop = Math.min(maximumScrollTop, Math.max(0, currentScrollTop));
  const rowTop = activeIndex * rowHeight;
  const rowBottom = rowTop + rowHeight;
  if (rowTop < boundedScrollTop) return Math.min(maximumScrollTop, rowTop);
  if (rowBottom > boundedScrollTop + viewportHeight) return Math.min(maximumScrollTop, rowBottom - viewportHeight);
  return boundedScrollTop;
}
