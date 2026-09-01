export type PipelineGraphNode = {
  id: string;
  kind: string;
  enabled: boolean;
  parameters: Record<string, unknown>;
  operator_version?: string;
};

export type PipelineVisualizationNode = PipelineGraphNode & {
  kind: 'visualize';
  tap_after_node_id: string;
};

export type VisualizationDimensions = {
  width: number;
  height: number;
  overlay_compatible?: boolean;
  coordinate_space_id?: string;
};

export type PipelineInsertionGap<TNode extends PipelineGraphNode, TVisualization extends PipelineVisualizationNode> = {
  key: string;
  upstream: TNode | TVisualization;
  downstream: TNode | TVisualization;
  transformInsertionIndex: number;
  retargetVisualizationId?: string;
  visualizationTapAfterNodeId?: string;
};

export const MAX_PIPELINE_VISUALIZATIONS = 4;

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, canonical(child)]),
  );
}

function fingerprintText(value: unknown): string {
  const text = JSON.stringify(canonical(value));
  let left = 0x811c9dc5;
  let right = 0x9e3779b9;
  for (let index = 0; index < text.length; index += 1) {
    left = Math.imul(left ^ text.charCodeAt(index), 0x01000193);
    right = Math.imul(right ^ text.charCodeAt(index), 0x85ebca6b);
  }
  return `${(left >>> 0).toString(16).padStart(8, '0')}${(right >>> 0).toString(16).padStart(8, '0')}`;
}

export function insertPipelineNode<T extends PipelineGraphNode>(nodes: T[], node: T, insertionIndex: number): T[] {
  if (node.kind === 'source' || node.kind === 'visualize') throw new Error('Only transform nodes can be inserted into the main chain');
  const index = Math.max(1, Math.min(nodes.length, insertionIndex));
  return [...nodes.slice(0, index), node, ...nodes.slice(index)];
}

export function removePipelineNode<T extends PipelineGraphNode>(nodes: T[], nodeId: string): T[] {
  const target = nodes.find((node) => node.id === nodeId);
  if (!target || target.kind === 'source') return nodes;
  return nodes.filter((node) => node.id !== nodeId);
}

export function normalizeVisualizationTaps<T extends PipelineVisualizationNode>(
  nodes: PipelineGraphNode[],
  visualizations: T[],
): T[] {
  if (!nodes.length) return visualizations.slice(0, MAX_PIPELINE_VISUALIZATIONS);
  const validIds = new Set(nodes.map((node) => node.id));
  const finalNodeId = nodes[nodes.length - 1].id;
  const byTapId = new Map<string, T>();
  let firstInvalid: T | undefined;
  for (const visualization of visualizations) {
    if (!validIds.has(visualization.tap_after_node_id)) {
      firstInvalid ??= visualization;
      continue;
    }
    if (!byTapId.has(visualization.tap_after_node_id)) byTapId.set(visualization.tap_after_node_id, visualization);
  }
  if (!byTapId.has(finalNodeId)) {
    if (firstInvalid) {
      byTapId.set(finalNodeId, { ...firstInvalid, tap_after_node_id: finalNodeId });
    } else {
      const lastValid = Array.from(byTapId.entries()).at(-1);
      if (lastValid) {
        byTapId.delete(lastValid[0]);
        byTapId.set(finalNodeId, { ...lastValid[1], tap_after_node_id: finalNodeId });
      }
    }
  }
  const ordered = nodes.flatMap((node) => byTapId.has(node.id) ? [byTapId.get(node.id)!] : []);
  if (ordered.length <= MAX_PIPELINE_VISUALIZATIONS) return ordered;
  const finalVisualization = ordered.find((visualization) => visualization.tap_after_node_id === finalNodeId)!;
  const leading = ordered.filter((visualization) => visualization.id !== finalVisualization.id).slice(0, MAX_PIPELINE_VISUALIZATIONS - 1);
  return [...leading, finalVisualization];
}

export function pipelineLinearItems<
  TNode extends PipelineGraphNode,
  TVisualization extends PipelineVisualizationNode,
>(nodes: TNode[], visualizations: TVisualization[]): Array<TNode | TVisualization> {
  const normalizedVisualizations = normalizeVisualizationTaps(nodes, visualizations);
  return nodes.flatMap((node) => [
    node,
    ...normalizedVisualizations.filter((visualization) => visualization.tap_after_node_id === node.id),
  ]);
}

export function pipelineInsertionGaps<
  TNode extends PipelineGraphNode,
  TVisualization extends PipelineVisualizationNode,
>(nodes: TNode[], visualizations: TVisualization[]): Array<PipelineInsertionGap<TNode, TVisualization>> {
  const items = pipelineLinearItems(nodes, visualizations);
  return items.slice(0, -1).map((upstream, index) => {
    const downstream = items[index + 1];
    const upstreamMainIndex = upstream.kind === 'visualize'
      ? nodes.findIndex((node) => node.id === (upstream as TVisualization).tap_after_node_id)
      : nodes.findIndex((node) => node.id === upstream.id);
    const downstreamMainIndex = downstream.kind === 'visualize'
      ? -1
      : nodes.findIndex((node) => node.id === downstream.id);
    return {
      key: `${upstream.id}->${downstream.id}`,
      upstream,
      downstream,
      transformInsertionIndex: downstreamMainIndex >= 0 ? downstreamMainIndex : upstreamMainIndex + 1,
      ...(downstream.kind === 'visualize' ? { retargetVisualizationId: downstream.id } : {}),
      ...(upstream.kind !== 'visualize' && downstream.kind !== 'visualize' ? { visualizationTapAfterNodeId: upstream.id } : {}),
    };
  });
}

export function insertPipelineNodeAtGap<
  TNode extends PipelineGraphNode,
  TVisualization extends PipelineVisualizationNode,
>(
  nodes: TNode[],
  visualizations: TVisualization[],
  node: TNode,
  gap: PipelineInsertionGap<TNode, TVisualization>,
): { nodes: TNode[]; visualizations: TVisualization[] } {
  const nextNodes = insertPipelineNode(nodes, node, gap.transformInsertionIndex);
  const retargeted = gap.retargetVisualizationId
    ? visualizations.map((visualization) => visualization.id === gap.retargetVisualizationId
      ? { ...visualization, tap_after_node_id: node.id }
      : visualization)
    : visualizations;
  return { nodes: nextNodes, visualizations: normalizeVisualizationTaps(nextNodes, retargeted) };
}

export function serializePipelineNodes(
  nodes: PipelineGraphNode[],
  visualizations: PipelineVisualizationNode[],
): PipelineGraphNode[] {
  return pipelineLinearItems(nodes, visualizations).map(({ id, kind, enabled, parameters, operator_version }) => ({
    id,
    kind,
    enabled,
    parameters,
    operator_version,
  }));
}

export function pipelineSignature(
  nodes: PipelineGraphNode[],
  visualizations: PipelineVisualizationNode[],
): string {
  return `pipeline:${fingerprintText(serializePipelineNodes(nodes, visualizations))}`;
}

export function visualizationOverlayCompatibility(items: VisualizationDimensions[]): { allowed: boolean; reason: string } {
  if (items.length < 2) return { allowed: false, reason: '至少需要两个显示结果才能叠加' };
  if (items.some((item) => item.overlay_compatible === false)) {
    return { allowed: false, reason: '包含频域、向量、Token 或非空间结果，只能使用独立分屏' };
  }
  const first = items[0];
  const invalid = items.find((item) => !Number.isFinite(item.width) || !Number.isFinite(item.height) || item.width <= 0 || item.height <= 0);
  if (invalid) return { allowed: false, reason: '结果缺少有效尺寸，只能使用独立分屏' };
  const mismatch = items.find((item) => item.width !== first.width || item.height !== first.height);
  if (mismatch) {
    return {
      allowed: false,
      reason: `尺寸不一致（${first.width}×${first.height} / ${mismatch.width}×${mismatch.height}），无法像素对齐叠加`,
    };
  }
  const coordinateSpaces = items.map((item) => item.coordinate_space_id).filter((value): value is string => Boolean(value));
  if (coordinateSpaces.length === items.length && coordinateSpaces.some((value) => value !== coordinateSpaces[0])) {
    return { allowed: false, reason: '尺寸一致但坐标空间不同，无法像素对齐叠加' };
  }
  return { allowed: true, reason: '尺寸一致，可像素对齐叠加' };
}
