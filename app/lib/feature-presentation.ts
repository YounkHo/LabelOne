import type { FeatureLayer } from './contracts';

export type FeatureTensorKind = 'spatial-map' | 'token-sequence' | 'vector' | 'matrix' | 'unsupported';

export function featureTensorKind(layer?: FeatureLayer): FeatureTensorKind {
  if (!layer) return 'unsupported';
  const rank = layer.shape.length;
  if (rank === 4) return 'spatial-map';
  if (rank === 3) return 'token-sequence';
  if (rank === 1 || (rank === 2 && layer.shape[0] === 1)) return 'vector';
  if (rank === 2) return 'matrix';
  return 'unsupported';
}

export function featureTensorKindLabel(kind: FeatureTensorKind): string {
  return {
    'spatial-map': '二维特征图',
    'token-sequence': 'Token 序列',
    vector: '特征向量',
    matrix: '特征矩阵',
    unsupported: '未知 Tensor',
  }[kind];
}

export function featureProjectionOptions(layer?: FeatureLayer): string[] {
  const kind = featureTensorKind(layer);
  if (kind === 'vector') return ['None'];
  const tokenCount = Number(layer?.shape[kind === 'token-sequence' ? 1 : 0]);
  const tokenGrid = Number.isInteger(tokenCount) && tokenCount > 0 && Number.isInteger(Math.sqrt(tokenCount)) ? ['Token Grid'] : [];
  if (kind === 'matrix') return ['None', 'PCA-1', 'Mean', 'Max', 'Single Channel', ...tokenGrid];
  if (kind === 'token-sequence') return ['PCA-1', 'Mean', 'Max', 'Single Channel', ...tokenGrid];
  if (kind === 'spatial-map') return ['PCA-1', 'Mean', 'Max', 'Single Channel'];
  return ['None'];
}

export function defaultFeatureProjection(layer?: FeatureLayer): string {
  const kind = featureTensorKind(layer);
  if (kind === 'spatial-map') return 'PCA-1';
  if (kind === 'token-sequence') return 'Mean';
  return 'None';
}

export function featurePreviewDescription(kind: FeatureTensorKind): string {
  if (kind === 'spatial-map') return '通道投影为二维特征图，按原始特征分辨率生成预览';
  if (kind === 'token-sequence') return 'Token 沿序列投影；仅平方 Token 数可切换网格';
  if (kind === 'vector') return '按向量原顺序生成有界折线预览，不使用空间插值';
  if (kind === 'matrix') return '矩阵可直接预览，或沿最后一维投影';
  return '当前 Tensor 暂无可用预览策略';
}
