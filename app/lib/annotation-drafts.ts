import { annotationDraftKey } from './annotation-history.ts';
import type { AnnotationEnvelope } from './contracts.ts';

const DATABASE_NAME = 'labelone-annotation-drafts';
const DATABASE_VERSION = 1;
const STORE_NAME = 'drafts';

export type PersistedAnnotationDraft = {
  key: string;
  dataset_id: string;
  asset_id: string;
  base_revision: string;
  document: AnnotationEnvelope['document'];
  updated_at: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function validatePersistedAnnotationDraft(value: unknown, datasetId?: string, assetId?: string): PersistedAnnotationDraft | null {
  if (!isRecord(value) || typeof value.dataset_id !== 'string' || typeof value.asset_id !== 'string' || typeof value.base_revision !== 'string' || typeof value.updated_at !== 'number' || !Number.isFinite(value.updated_at) || !isRecord(value.document) || !Array.isArray(value.document.shapes)) return null;
  const shapesAreValid = value.document.shapes.every((shape) => isRecord(shape)
    && typeof shape.label === 'string'
    && typeof shape.shape_type === 'string'
    && Array.isArray(shape.points)
    && shape.points.every((point) => Array.isArray(point) && point.length === 2 && point.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate))));
  if (!shapesAreValid) return null;
  const key = annotationDraftKey(value.dataset_id, value.asset_id);
  if (value.key !== key || (datasetId !== undefined && value.dataset_id !== datasetId) || (assetId !== undefined && value.asset_id !== assetId)) return null;
  return value as PersistedAnnotationDraft;
}

function openDraftDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('此浏览器不支持 IndexedDB 草稿恢复'));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB 打开失败'));
    request.onblocked = () => reject(new Error('IndexedDB 升级被其他页面阻塞，请关闭其他 LabelOne 页面后重试'));
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME, { keyPath: 'key' });
    };
    request.onsuccess = () => resolve(request.result);
  });
}

async function withStore<T>(mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await openDraftDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = operation(transaction.objectStore(STORE_NAME));
    let result: T;
    request.onsuccess = () => { result = request.result; };
    request.onerror = () => reject(request.error ?? new Error('IndexedDB 草稿操作失败'));
    transaction.oncomplete = () => {
      database.close();
      resolve(result!);
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error('IndexedDB 草稿事务失败'));
    };
  });
}

export async function getAnnotationDraft(datasetId: string, assetId: string): Promise<PersistedAnnotationDraft | null> {
  const key = annotationDraftKey(datasetId, assetId);
  const value = await withStore<unknown>('readonly', (store) => store.get(key));
  if (value === undefined) return null;
  const record = validatePersistedAnnotationDraft(value, datasetId, assetId);
  if (record) return record;
  await deleteAnnotationDraft(datasetId, assetId);
  return null;
}

export async function putAnnotationDraft(record: Omit<PersistedAnnotationDraft, 'key' | 'updated_at'>): Promise<PersistedAnnotationDraft> {
  const stored: PersistedAnnotationDraft = {
    ...structuredClone(record),
    key: annotationDraftKey(record.dataset_id, record.asset_id),
    updated_at: Date.now(),
  };
  await withStore<IDBValidKey>('readwrite', (store) => store.put(stored));
  return stored;
}

export async function deleteAnnotationDraft(datasetId: string, assetId: string): Promise<void> {
  await withStore<undefined>('readwrite', (store) => store.delete(annotationDraftKey(datasetId, assetId)) as IDBRequest<undefined>);
}
