'use client';

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { positionGlobalTooltip, type TooltipPosition, type TooltipRect } from '../lib/global-tooltip-position';
import { fullscreenPortalTarget } from '../lib/portal-target';

const TOOLTIP_ID = 'labelone-global-tooltip';
const INITIAL_DELAY_MS = 320;
const WARM_WINDOW_MS = 700;

type ActiveTooltip = {
  target: HTMLElement;
  title: string | null;
  description: string;
  shortcut: string | null;
  anchorPoint: { x: number; y: number } | null;
};

function tooltipTarget(node: EventTarget | null): HTMLElement | null {
  if (!(node instanceof Element)) return null;
  return node.closest<HTMLElement>('[data-tooltip]') ?? node.closest<HTMLElement>('[title]');
}

function tooltipContent(target: HTMLElement) {
  const nativeTitle = target.getAttribute('title')?.trim() ?? '';
  const description = target.dataset.tooltip?.trim() || nativeTitle;
  if (!description) return null;
  return {
    title: target.dataset.tooltipTitle?.trim() || null,
    description,
    shortcut: target.dataset.shortcut?.trim() || null,
  };
}

export function GlobalTooltip() {
  const [active, setActive] = useState<ActiveTooltip | null>(null);
  const [position, setPosition] = useState<TooltipPosition | null>(null);
  const activeRef = useRef<ActiveTooltip | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const targetRef = useRef<HTMLElement | null>(null);
  const timerRef = useRef<number | null>(null);
  const pointerRef = useRef<{ x: number; y: number } | null>(null);
  const warmUntilRef = useRef(0);
  const savedTitleRef = useRef(new WeakMap<HTMLElement, string>());
  const savedDescriptionRef = useRef(new WeakMap<HTMLElement, string | null>());

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    const restoreTarget = (target: HTMLElement | null) => {
      if (!target) return;
      const title = savedTitleRef.current.get(target);
      if (title !== undefined && !target.hasAttribute('title')) target.setAttribute('title', title);
      savedTitleRef.current.delete(target);
      const describedBy = savedDescriptionRef.current.get(target);
      if (describedBy !== undefined) {
        if (describedBy) target.setAttribute('aria-describedby', describedBy);
        else target.removeAttribute('aria-describedby');
      }
      savedDescriptionRef.current.delete(target);
    };
    const hide = () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      timerRef.current = null;
      if (activeRef.current) warmUntilRef.current = Date.now() + WARM_WINDOW_MS;
      restoreTarget(targetRef.current);
      targetRef.current = null;
      pointerRef.current = null;
      setActive(null);
      setPosition(null);
    };
    const schedule = (target: HTMLElement, immediate: boolean, pointer: { x: number; y: number } | null = null) => {
      const content = tooltipContent(target);
      if (!content) return;
      if (targetRef.current === target) return;
      hide();
      targetRef.current = target;
      pointerRef.current = target.dataset.tooltipAnchor === 'pointer' ? pointer : null;
      const nativeTitle = target.getAttribute('title');
      if (nativeTitle) {
        savedTitleRef.current.set(target, nativeTitle);
        target.removeAttribute('title');
      }
      savedDescriptionRef.current.set(target, target.getAttribute('aria-describedby'));
      target.setAttribute('aria-describedby', TOOLTIP_ID);
      const show = () => {
        timerRef.current = null;
        if (targetRef.current !== target || !target.isConnected) return;
        setActive({ target, ...content, anchorPoint: pointerRef.current });
      };
      const delay = immediate || Date.now() < warmUntilRef.current ? 0 : INITIAL_DELAY_MS;
      if (delay === 0) show();
      else timerRef.current = window.setTimeout(show, delay);
    };
    const onPointerOver = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return;
      const target = tooltipTarget(event.target);
      if (target) schedule(target, false, { x: event.clientX, y: event.clientY });
    };
    const onPointerMove = (event: PointerEvent) => {
      const target = targetRef.current;
      if (!target || target.dataset.tooltipAnchor !== 'pointer') return;
      const anchorPoint = { x: event.clientX, y: event.clientY };
      pointerRef.current = anchorPoint;
      setActive((current) => current?.target === target ? { ...current, anchorPoint } : current);
    };
    const onPointerOut = (event: PointerEvent) => {
      const current = targetRef.current;
      if (!current) return;
      if (event.relatedTarget instanceof Node && current.contains(event.relatedTarget)) return;
      hide();
    };
    const onFocusIn = (event: FocusEvent) => {
      const target = tooltipTarget(event.target);
      if (target) schedule(target, true);
    };
    const onFocusOut = (event: FocusEvent) => {
      const current = targetRef.current;
      if (!current) return;
      if (event.relatedTarget instanceof Node && current.contains(event.relatedTarget)) return;
      hide();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') hide();
    };
    document.addEventListener('pointerover', onPointerOver);
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerout', onPointerOut);
    document.addEventListener('focusin', onFocusIn);
    document.addEventListener('focusout', onFocusOut);
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    return () => {
      document.removeEventListener('pointerover', onPointerOver);
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerout', onPointerOut);
      document.removeEventListener('focusin', onFocusIn);
      document.removeEventListener('focusout', onFocusOut);
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('resize', hide);
      hide();
    };
  }, []);

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!active || !tooltip || !active.target.isConnected) return;
    const anchor: TooltipRect = active.anchorPoint
      ? { left: active.anchorPoint.x, right: active.anchorPoint.x, top: active.anchorPoint.y, bottom: active.anchorPoint.y, width: 0, height: 0 }
      : active.target.getBoundingClientRect();
    const size = tooltip.getBoundingClientRect();
    setPosition(positionGlobalTooltip(anchor, size, { width: window.innerWidth, height: window.innerHeight }));
  }, [active]);

  if (!active || typeof document === 'undefined') return null;
  return createPortal(<div ref={tooltipRef} id={TOOLTIP_ID} className="global-tooltip" data-placement={position?.placement ?? 'bottom'} role="tooltip" style={{ left: position?.left ?? -9999, top: position?.top ?? -9999, visibility: position ? 'visible' : 'hidden' }}>
    <span className="global-tooltip-copy">{active.title && <strong>{active.title}</strong>}<span>{active.description}</span></span>
    {active.shortcut && <kbd>{active.shortcut}</kbd>}
  </div>, fullscreenPortalTarget(document));
}
