'use client';

import { createPortal } from 'react-dom';
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';

import { nextEnabledOption, selectMenuPlacement } from '../lib/custom-select-model';
import { fullscreenPortalTarget } from '../lib/portal-target';

export type CustomSelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

type CustomSelectProps = {
  value: string;
  options: CustomSelectOption[];
  onChange: (value: string) => void;
  ariaLabel: string;
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  title?: string;
};

type MenuPosition = { left: number; top: number; width: number; maxHeight: number; placement: 'top' | 'bottom' };

export function CustomSelect({ value, options, onChange, ariaLabel, className = '', disabled = false, placeholder = '请选择', title }: CustomSelectProps) {
  const id = useId().replaceAll(':', '');
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const typeaheadRef = useRef({ value: '', timer: 0 });
  const [open, setOpen] = useState(false);
  const selectedIndex = options.findIndex((option) => option.value === value);
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const [position, setPosition] = useState<MenuPosition>({ left: 0, top: 0, width: 120, maxHeight: 240, placement: 'bottom' });
  const selected = selectedIndex >= 0 ? options[selectedIndex] : null;

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const width = Math.min(Math.max(rect.width, 120), Math.max(120, window.innerWidth - 16));
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
    const placement = selectMenuPlacement(rect, window.innerHeight, Math.min(280, Math.max(48, options.length * 31 + 8)));
    setPosition({ left, width, ...placement });
  }, [options.length]);

  const openMenu = useCallback((preferredIndex?: number) => {
    if (disabled || !options.length) return;
    const fallback = selectedIndex >= 0 && !options[selectedIndex]?.disabled ? selectedIndex : nextEnabledOption(options, -1, 1);
    setActiveIndex(preferredIndex ?? fallback);
    setOpen(true);
  }, [disabled, options, selectedIndex]);

  const choose = (index: number) => {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    setOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus({ preventScroll: true }));
  };

  const move = (direction: 1 | -1) => {
    const next = nextEnabledOption(options, activeIndex, direction);
    if (next >= 0) setActiveIndex(next);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!open) openMenu(event.key === 'ArrowDown' ? nextEnabledOption(options, -1, 1) : nextEnabledOption(options, 0, -1));
      else move(event.key === 'ArrowDown' ? 1 : -1);
      return;
    }
    if (event.key === 'Home' || event.key === 'End') {
      if (!open) return;
      event.preventDefault();
      setActiveIndex(event.key === 'Home' ? nextEnabledOption(options, -1, 1) : nextEnabledOption(options, 0, -1));
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open && activeIndex >= 0) choose(activeIndex);
      else openMenu();
      return;
    }
    if (event.key === 'Escape' && open) {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === 'Tab') { setOpen(false); return; }
    if (event.key.length === 1 && /\S/.test(event.key)) {
      window.clearTimeout(typeaheadRef.current.timer);
      typeaheadRef.current.value += event.key.toLocaleLowerCase();
      const start = Math.max(-1, activeIndex);
      for (let offset = 1; offset <= options.length; offset += 1) {
        const index = (start + offset) % options.length;
        if (!options[index].disabled && options[index].label.toLocaleLowerCase().startsWith(typeaheadRef.current.value)) { setActiveIndex(index); if (!open) openMenu(index); break; }
      }
      typeaheadRef.current.timer = window.setTimeout(() => { typeaheadRef.current.value = ''; }, 650);
    }
  };

  useLayoutEffect(() => { if (open) updatePosition(); }, [open, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: Event) => {
      const target = event.target as Node | null;
      if (target && !rootRef.current?.contains(target) && !menuRef.current?.contains(target)) setOpen(false);
    };
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('focusin', closeOutside);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('focusin', closeOutside);
    };
  }, [open, updatePosition]);

  useEffect(() => () => window.clearTimeout(typeaheadRef.current.timer), []);

  return <div ref={rootRef} className={`custom-select ${open ? 'open' : ''} ${disabled ? 'disabled' : ''} ${className}`} title={title}>
    <button ref={triggerRef} type="button" className="custom-select-trigger" role="combobox" aria-label={ariaLabel} aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-listbox`} aria-activedescendant={open && activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined} disabled={disabled} onClick={() => open ? setOpen(false) : openMenu()} onKeyDown={onKeyDown}>
      <span className={selected ? '' : 'placeholder'}>{selected?.label ?? placeholder}</span><i aria-hidden="true" />
    </button>
    {open && createPortal(<div ref={menuRef} id={`${id}-listbox`} className="custom-select-menu" data-placement={position.placement} role="listbox" aria-label={ariaLabel} style={{ left: position.left, top: position.top, width: position.width, maxHeight: position.maxHeight }}>
      {options.map((option, index) => <button key={option.value} id={`${id}-option-${index}`} type="button" className={`custom-select-option ${index === activeIndex ? 'active' : ''} ${option.value === value ? 'selected' : ''}`} role="option" aria-selected={option.value === value} disabled={option.disabled} onPointerEnter={() => !option.disabled && setActiveIndex(index)} onPointerDown={(event) => event.preventDefault()} onClick={() => choose(index)}><span>{option.label}</span>{option.value === value && <b aria-hidden="true">✓</b>}</button>)}
    </div>, fullscreenPortalTarget(document))}
  </div>;
}
