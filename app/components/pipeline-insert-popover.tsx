'use client';

import { createPortal } from 'react-dom';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';

import { pipelineInsertPopoverPosition, type PipelineInsertPopoverPosition } from '../lib/pipeline-insert-popover';
import { fullscreenPortalTarget } from '../lib/portal-target';

export type PipelineInsertOperator = {
  kind: string;
  name: string;
  icon: ReactNode;
  color: string;
  disabled?: boolean;
  disabledReason?: string;
};

type PipelineInsertPopoverProps = {
  id: string;
  anchor: HTMLButtonElement | null;
  search: string;
  operators: PipelineInsertOperator[];
  showVisualization: boolean;
  visualizationDisabled: boolean;
  visualizationName: string;
  visualizationTitle: string;
  visualizationSourceName: string;
  onSearchChange: (value: string) => void;
  onAddOperator: (kind: string) => void;
  onAddVisualization: () => void;
  onClose: () => void;
};

export function PipelineInsertPopover({ id, anchor, search, operators, showVisualization, visualizationDisabled, visualizationName, visualizationTitle, visualizationSourceName, onSearchChange, onAddOperator, onAddVisualization, onClose }: PipelineInsertPopoverProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef(0);
  const [position, setPosition] = useState<PipelineInsertPopoverPosition | null>(null);

  const updatePosition = useCallback(() => {
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    const canvasWidth = anchor.closest('.flow-canvas')?.getBoundingClientRect().width ?? 300;
    const desiredWidth = Math.min(320, Math.max(240, canvasWidth - 12));
    const estimatedHeight = 38 + (operators.length + Number(showVisualization)) * 31;
    const desiredHeight = Math.min(244, Math.max(96, menuRef.current?.scrollHeight ?? estimatedHeight));
    setPosition(pipelineInsertPopoverPosition(rect, window.innerWidth, window.innerHeight, desiredWidth, desiredHeight));
  }, [anchor, operators.length, showVisualization]);

  const schedulePosition = useCallback(() => {
    window.cancelAnimationFrame(frameRef.current);
    frameRef.current = window.requestAnimationFrame(updatePosition);
  }, [updatePosition]);

  const restoreAnchorFocus = useCallback(() => {
    window.requestAnimationFrame(() => anchor?.focus({ preventScroll: true }));
  }, [anchor]);

  const chooseOperator = (kind: string) => {
    onAddOperator(kind);
    restoreAnchorFocus();
  };

  const chooseVisualization = () => {
    onAddVisualization();
    restoreAnchorFocus();
  };

  useLayoutEffect(() => { schedulePosition(); }, [schedulePosition]);

  useEffect(() => {
    const closeOutside = (event: Event) => {
      const target = event.target as Node | null;
      if (target && !anchor?.contains(target) && !menuRef.current?.contains(target)) onClose();
    };
    window.addEventListener('resize', schedulePosition);
    window.addEventListener('scroll', schedulePosition, true);
    window.visualViewport?.addEventListener('resize', schedulePosition);
    window.visualViewport?.addEventListener('scroll', schedulePosition);
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('focusin', closeOutside);
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedulePosition);
    if (anchor) observer?.observe(anchor);
    if (menuRef.current) observer?.observe(menuRef.current);
    return () => {
      window.cancelAnimationFrame(frameRef.current);
      window.removeEventListener('resize', schedulePosition);
      window.removeEventListener('scroll', schedulePosition, true);
      window.visualViewport?.removeEventListener('resize', schedulePosition);
      window.visualViewport?.removeEventListener('scroll', schedulePosition);
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('focusin', closeOutside);
      observer?.disconnect();
    };
  }, [anchor, onClose, schedulePosition]);

  if (!anchor) return null;
  return createPortal(<div ref={menuRef} id={id} className="flow-insert-menu flow-insert-popover" data-placement={position?.placement ?? 'bottom'} role="dialog" aria-label="添加处理流算子" style={{ left: position?.left ?? -9999, top: position?.top ?? -9999, width: position?.width ?? 288, maxHeight: position?.maxHeight ?? 244, visibility: position ? 'visible' : 'hidden' }} onKeyDown={(event) => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose(); restoreAnchorFocus(); } }}>
    <input autoFocus type="search" value={search} onChange={(event) => onSearchChange(event.target.value)} placeholder="搜索并插入算子" aria-label="搜索要插入的算子" spellCheck={false} />
    {operators.map((item) => <button key={item.kind} type="button" className={item.disabled ? 'blocked-option' : ''} disabled={item.disabled} title={item.disabledReason} onClick={() => chooseOperator(item.kind)}><span className={`operator-icon ${item.color}`}>{item.icon}</span><strong>{item.name}</strong><small>{item.disabledReason ?? (item.kind.startsWith('opencv.') ? 'OpenCV' : 'Image → Image')}</small></button>)}
    {showVisualization && <button type="button" className={`visualization-option ${visualizationDisabled ? 'blocked-option' : ''}`} disabled={visualizationDisabled} title={visualizationTitle} onClick={chooseVisualization}><span className="operator-icon blue">◉</span><strong>{visualizationName}</strong><small>{visualizationDisabled ? visualizationTitle : `接收 ${visualizationSourceName} 的输出`}</small></button>}
    {operators.length === 0 && !showVisualization && <small className="empty">无匹配算子</small>}
  </div>, fullscreenPortalTarget(document));
}
