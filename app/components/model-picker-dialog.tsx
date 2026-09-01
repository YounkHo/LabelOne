'use client';

import { useDeferredValue, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';

export type ModelPickerItem = {
  id: string;
  name: string;
  task: string;
  runtime: string;
  badge: string;
  availability?: string;
  runtimeState: 'unloaded' | 'loading' | 'loaded' | 'failed';
  usageCount: number;
};

export function ModelPickerDialog({
  models,
  tasks,
  selectedTask,
  selectedModelId,
  refreshing,
  downloadPending,
  downloadActive,
  downloadProgress,
  onTaskChange,
  onSelect,
  onDownload,
  onClose,
}: {
  models: ModelPickerItem[];
  tasks: string[];
  selectedTask: string;
  selectedModelId: string;
  refreshing: boolean;
  downloadPending: boolean;
  downloadActive: boolean;
  downloadProgress: number | null;
  onTaskChange: (task: string) => void;
  onSelect: (modelId: string) => void;
  onDownload: () => void;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query.trim().toLocaleLowerCase());
  const selectedModel = models.find((model) => model.id === selectedModelId);
  const taskFiltered = selectedTask === '全部' ? models : models.filter((model) => model.task === selectedTask);
  const filtered = deferredQuery
    ? taskFiltered.filter((model) => `${model.name} ${model.id} ${model.task} ${model.runtime}`.toLocaleLowerCase().includes(deferredQuery))
    : taskFiltered;

  useEffect(() => {
    searchRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not(:disabled),[tabindex]:not([tabindex="-1"])')];
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return <div className="model-picker-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section ref={dialogRef} className="model-picker-dialog" role="dialog" aria-modal="true" aria-labelledby="model-picker-dialog-title">
      <header><div><span className="eyebrow">推理模型</span><h2 id="model-picker-dialog-title">选择模型</h2><p>已下载模型选择后自动加载；缺少权重的模型需要先下载。</p></div><button ref={closeRef} type="button" aria-label="关闭模型选择" onClick={onClose}>×</button></header>
      <label className="model-picker-search" htmlFor="model-picker-search-input"><span className="sr-only">搜索模型</span><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.25" /><path d="m10.25 10.25 3 3" /></svg><input ref={searchRef} id="model-picker-search-input" type="search" value={query} placeholder="搜索名称、任务、运行时或模型 ID" autoComplete="off" spellCheck={false} onChange={(event) => setQuery(event.target.value)} />{query && <button type="button" aria-label="清空模型搜索" onClick={() => { setQuery(''); searchRef.current?.focus(); }}>清空</button>}</label>
      <div className="model-picker-task-tabs" role="tablist" aria-label="模型类别">{['全部', ...tasks].map((task) => <button type="button" role="tab" aria-selected={selectedTask === task} className={selectedTask === task ? 'active' : ''} key={task} onClick={() => onTaskChange(task)}>{task}<b>{task === '全部' ? models.length : models.filter((model) => model.task === task).length}</b></button>)}</div>
      <div className="model-picker-options" role="listbox" aria-label="可用模型">{filtered.map((model) => {
        const selected = selectedModelId === model.id;
        const loaded = model.runtimeState === 'loaded';
        const missing = model.availability === 'missing_weights';
        return <button type="button" role="option" aria-selected={selected} className={`${selected ? 'selected' : ''} ${loaded ? 'loaded' : ''} ${missing ? 'missing' : ''}`} key={model.id} onClick={() => onSelect(model.id)}><span>{model.badge.slice(0, 2)}</span><div><strong>{model.name}</strong><small>{model.task} · {model.runtime}{model.usageCount > 0 ? ` · 使用 ${model.usageCount} 次` : ''}</small></div><i>{loaded ? '已加载' : model.runtimeState === 'loading' ? '加载中' : missing ? '需下载' : model.availability === 'available' ? '已下载' : model.availability ?? '不可用'}</i></button>;
      })}{filtered.length === 0 && <p className="model-picker-empty">{deferredQuery ? `没有匹配“${query.trim()}”的模型。` : '这个分类下没有可用模型。'}</p>}</div>
      <footer aria-live="polite"><span>{refreshing ? '正在刷新模型状态…' : selectedModel?.availability === 'missing_weights' ? `“${selectedModel.name}”缺少本地权重。` : '选择已下载模型后会立即关闭并自动加载。'}</span>{selectedModel?.availability === 'missing_weights' && <button type="button" className={`model-picker-download ${downloadActive ? 'downloading' : ''}`} style={downloadProgress !== null ? ({ '--model-download-progress': `${downloadProgress}%` } as CSSProperties) : undefined} disabled={downloadPending || downloadActive} onClick={onDownload}>{downloadActive ? downloadProgress !== null ? `下载中 ${downloadProgress}%` : '下载中…' : downloadPending ? '准备下载…' : '下载权重'}</button>}</footer>
    </section>
  </div>;
}
