import type {
  GlobalWorkspaceSettings,
  WorkspacePipelineNode,
  WorkspacePipelineSettings,
  WorkspaceVisualizationNode,
} from './contracts';

type PipelineStateInput = {
  enabled: boolean;
  scope: 'current' | 'all';
  nodes: WorkspacePipelineNode[];
  visualizations: WorkspaceVisualizationNode[];
  displayMode: WorkspacePipelineSettings['display_mode'];
  singleSource: string;
  layerState: WorkspacePipelineSettings['layer_state'];
};

function cloneRecord(value: Record<string, unknown>): Record<string, unknown> {
  return structuredClone(value);
}

export function snapshotPipelineSettings(input: PipelineStateInput): WorkspacePipelineSettings {
  return {
    enabled: input.enabled,
    scope: input.scope,
    nodes: input.nodes.map((node) => ({
      id: node.id,
      kind: node.kind,
      name: node.name,
      enabled: node.enabled,
      parameters: cloneRecord(node.parameters),
      ...(node.operator_version ? { operator_version: node.operator_version } : {}),
    })),
    visualizations: input.visualizations.slice(0, 4).map((node) => ({
      id: node.id,
      kind: 'visualize',
      name: node.name,
      enabled: node.enabled,
      parameters: cloneRecord(node.parameters),
      ...(node.operator_version ? { operator_version: node.operator_version } : {}),
      tap_after_node_id: node.tap_after_node_id,
    })),
    display_mode: input.displayMode,
    single_source: input.singleSource,
    layer_state: Object.fromEntries(Object.entries(input.layerState).slice(0, 4).map(([id, state]) => [id, {
      visible: state.visible,
      opacity: Math.max(0, Math.min(100, Math.round(state.opacity))),
    }])),
  };
}

export function usablePipelineSettings(value: WorkspacePipelineSettings | null | undefined): WorkspacePipelineSettings | null {
  if (!value || !Array.isArray(value.nodes) || !Array.isArray(value.visualizations)) return null;
  const nodes = value.nodes.filter((node) => node && typeof node.id === 'string' && typeof node.kind === 'string');
  if (!nodes.length || nodes[0].kind !== 'source') return null;
  const ids = new Set(nodes.map((node) => node.id));
  if (ids.size !== nodes.length) return null;
  const visualizations = value.visualizations.filter((node) => node?.kind === 'visualize' && ids.has(node.tap_after_node_id)).slice(0, 4);
  if (!visualizations.length) return null;
  return { ...value, nodes, visualizations };
}

export function globalWorkspaceSettings(
  pipeline: WorkspacePipelineSettings,
  inference: GlobalWorkspaceSettings['inference'],
): GlobalWorkspaceSettings {
  return {
    schema_version: 1,
    pipeline,
    inference: {
      model_id: inference.model_id,
      provider: inference.provider,
      parameters: cloneRecord(inference.parameters),
    },
  };
}
