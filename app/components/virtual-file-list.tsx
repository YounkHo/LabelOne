'use client';

import { useEffect, useMemo, useRef, useState, type ReactNode, type UIEvent } from 'react';

import { scrollTopForActiveListItem } from '../lib/virtual-list-scroll';

type VirtualFileListProps<T> = {
  items: T[];
  total: number;
  itemKey: (item: T) => string;
  renderItem: (item: T, index: number) => ReactNode;
  hasMore: boolean;
  loadingMore: boolean;
  activeItemKey?: string | null;
  onEndReached?: () => void;
  emptyState?: ReactNode;
  rowHeight?: number;
  overscan?: number;
};

export function VirtualFileList<T>({
  items,
  total,
  itemKey,
  renderItem,
  hasMore,
  loadingMore,
  activeItemKey = null,
  onEndReached,
  emptyState,
  rowHeight = 55,
  overscan = 10,
}: VirtualFileListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const endReachedRef = useRef(onEndReached);
  const requestedLengthRef = useRef<number | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  useEffect(() => { endReachedRef.current = onEndReached; }, [onEndReached]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const update = () => setViewportHeight(element.clientHeight);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (requestedLengthRef.current !== null && requestedLengthRef.current !== items.length) requestedLengthRef.current = null;
  }, [items.length]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element || !activeItemKey) return;
    const activeIndex = items.findIndex((item) => itemKey(item) === activeItemKey);
    const nextScrollTop = scrollTopForActiveListItem({
      activeIndex,
      currentScrollTop: element.scrollTop,
      itemCount: items.length,
      rowHeight,
      viewportHeight: element.clientHeight,
    });
    if (nextScrollTop === element.scrollTop) return;
    element.scrollTop = nextScrollTop;
    setScrollTop(nextScrollTop);
  }, [activeItemKey, itemKey, items, rowHeight, viewportHeight]);

  const range = useMemo(() => {
    const visibleCount = Math.max(1, Math.ceil(viewportHeight / rowHeight));
    const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
    const end = Math.min(items.length, start + visibleCount + overscan * 2);
    return { start, end };
  }, [items.length, overscan, rowHeight, scrollTop, viewportHeight]);

  useEffect(() => {
    if (!hasMore || loadingMore || !items.length || range.end < items.length - Math.min(20, overscan * 2)) return;
    if (requestedLengthRef.current === items.length) return;
    requestedLengthRef.current = items.length;
    endReachedRef.current?.();
  }, [hasMore, items.length, loadingMore, overscan, range.end]);

  const handleScroll = (event: UIEvent<HTMLDivElement>) => setScrollTop(event.currentTarget.scrollTop);
  const visibleItems = items.slice(range.start, range.end);

  return <div ref={containerRef} className="file-list virtual-file-list" role="list" aria-busy={loadingMore} onScroll={handleScroll}>
    {items.length === 0 ? emptyState : <div className="virtual-file-spacer" style={{ height: items.length * rowHeight }}>
      {visibleItems.map((item, visibleIndex) => {
        const index = range.start + visibleIndex;
        return <div key={itemKey(item)} className="virtual-file-row" role="listitem" aria-posinset={index + 1} aria-setsize={total} style={{ height: rowHeight, transform: `translateY(${index * rowHeight}px)` }}>{renderItem(item, index)}</div>;
      })}
    </div>}
    {loadingMore && <div className="virtual-file-loading"><span className="spinner" />正在加载下一页…</div>}
  </div>;
}
