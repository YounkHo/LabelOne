import type { AssetCursorPage, DatasetScanItem, DatasetScanResult } from './contracts.ts';

export function directoryBasename(path: string): string {
  return path.replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean).at(-1) ?? '本地数据集';
}

export function mergeAssetCursorPage(current: AssetCursorPage, incoming: AssetCursorPage, append: boolean): AssetCursorPage {
  if (!append || current.index_revision !== incoming.index_revision) return structuredClone(incoming);
  const seen = new Set(current.items.map((item) => item.asset_id));
  return {
    ...incoming,
    items: [...current.items, ...incoming.items.filter((item) => !seen.has(item.asset_id))],
  };
}

export function summarizeScanItems(items: DatasetScanItem[]): DatasetScanResult['summary'] {
  const summary: DatasetScanResult['summary'] = {
    valid: 0,
    duplicate_match: 0,
    orphan_annotation: 0,
    corrupt_image: 0,
    corrupt_annotation: 0,
    hidden_image_only: 0,
  };
  for (const item of items) summary[item.status] += 1;
  return summary;
}

export function formatDownloadProgress(receivedBytes: number, totalBytes: number | null): { percent: number | null; label: string } {
  const safeReceived = Math.max(0, Number.isFinite(receivedBytes) ? receivedBytes : 0);
  const safeTotal = totalBytes !== null && Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : null;
  return {
    percent: safeTotal === null ? null : Math.min(100, Math.round(safeReceived / safeTotal * 100)),
    label: `${formatBytes(safeReceived)} / ${safeTotal === null ? '未知大小' : formatBytes(safeTotal)}`,
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
