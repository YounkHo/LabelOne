export type ShortcutActionId =
  | 'edit.undo'
  | 'edit.redo'
  | 'edit.save'
  | 'edit.changeCategory'
  | 'navigation.previous'
  | 'navigation.next'
  | 'canvas.zoomIn'
  | 'canvas.zoomOut'
  | 'canvas.fit'
  | 'canvas.actualSize'
  | 'tool.select'
  | 'tool.pan'
  | 'tool.rect'
  | 'tool.rotation'
  | 'tool.polygon'
  | 'tool.point'
  | 'tool.line'
  | 'tool.circle'
  | 'tool.brush';

export type ShortcutGroup = '编辑' | '图片切换' | '画布视图' | '标注工具';

export type ShortcutDefinition = {
  id: ShortcutActionId;
  label: string;
  description: string;
  group: ShortcutGroup;
  scope: 'app' | 'canvas';
  defaultBinding: string;
  fixedAlternative?: string;
};

export type ShortcutMap = Record<ShortcutActionId, string>;
export type ShortcutOverrides = Partial<Record<ShortcutActionId, string>>;

export const shortcutDefinitions: ShortcutDefinition[] = [
  { id: 'edit.undo', label: '撤销', description: '撤销上一次标注修改', group: '编辑', scope: 'app', defaultBinding: 'Mod+Z' },
  { id: 'edit.redo', label: '重做', description: '恢复上一次撤销的修改', group: '编辑', scope: 'app', defaultBinding: 'Mod+Shift+Z' },
  { id: 'edit.save', label: '保存当前标注', description: '立即保存当前图片的标注', group: '编辑', scope: 'app', defaultBinding: 'Mod+S' },
  { id: 'edit.changeCategory', label: '修改类别', description: '修改当前聚焦类别或选中对象的类别', group: '编辑', scope: 'app', defaultBinding: 'F2' },
  { id: 'navigation.previous', label: '上一张图片', description: '切换到上一张可标注图片', group: '图片切换', scope: 'canvas', defaultBinding: 'A', fixedAlternative: 'ArrowLeft' },
  { id: 'navigation.next', label: '下一张图片', description: '切换到下一张可标注图片', group: '图片切换', scope: 'canvas', defaultBinding: 'D', fixedAlternative: 'ArrowRight' },
  { id: 'canvas.zoomIn', label: '放大画布', description: '以画布中心为锚点放大', group: '画布视图', scope: 'canvas', defaultBinding: '+' },
  { id: 'canvas.zoomOut', label: '缩小画布', description: '以画布中心为锚点缩小', group: '画布视图', scope: 'canvas', defaultBinding: '-' },
  { id: 'canvas.fit', label: '适应窗口', description: '居中并适应当前可用空间', group: '画布视图', scope: 'canvas', defaultBinding: '0' },
  { id: 'canvas.actualSize', label: '实际大小', description: '恢复到 1:1 显示', group: '画布视图', scope: 'canvas', defaultBinding: '1' },
  { id: 'tool.select', label: '选择工具', description: '选择与编辑标注对象', group: '标注工具', scope: 'canvas', defaultBinding: 'V' },
  { id: 'tool.pan', label: '平移工具', description: '拖动画布视口', group: '标注工具', scope: 'canvas', defaultBinding: 'H' },
  { id: 'tool.rect', label: '矩形框', description: '绘制轴对齐矩形框', group: '标注工具', scope: 'canvas', defaultBinding: 'R' },
  { id: 'tool.rotation', label: '旋转框', description: '绘制四点旋转框', group: '标注工具', scope: 'canvas', defaultBinding: 'O' },
  { id: 'tool.polygon', label: '多边形', description: '逐点绘制多边形', group: '标注工具', scope: 'canvas', defaultBinding: 'P' },
  { id: 'tool.point', label: '点', description: '创建点标注', group: '标注工具', scope: 'canvas', defaultBinding: 'T' },
  { id: 'tool.line', label: '直线', description: '以两次点击创建直线', group: '标注工具', scope: 'canvas', defaultBinding: 'L' },
  { id: 'tool.circle', label: '圆', description: '拖动创建圆形标注', group: '标注工具', scope: 'canvas', defaultBinding: 'C' },
  { id: 'tool.brush', label: '自由线', description: '按住拖动绘制开放连续线', group: '标注工具', scope: 'canvas', defaultBinding: 'B' },
];

const actionIds = new Set(shortcutDefinitions.map((definition) => definition.id));
const forbiddenBindings = new Set(['Escape', 'Tab', 'Enter', 'Space', 'Backspace', 'Delete', 'F5', 'Mod+R', 'Mod+W', 'Mod+T', 'Mod+P', 'Mod+L', 'Alt+F4']);

export const shortcutGroups: ShortcutGroup[] = ['编辑', '图片切换', '画布视图', '标注工具'];

export function defaultShortcutMap(): ShortcutMap {
  return Object.fromEntries(shortcutDefinitions.map((definition) => [definition.id, definition.defaultBinding])) as ShortcutMap;
}

export function resolvedShortcutMap(overrides: ShortcutOverrides): ShortcutMap {
  return { ...defaultShortcutMap(), ...sanitizeShortcutOverrides(overrides) };
}

export function normalizeShortcutKey(key: string): string {
  if (key === ' ') return 'Space';
  if (key === 'Esc') return 'Escape';
  if (key.length === 1 && /[a-z]/i.test(key)) return key.toUpperCase();
  return key;
}

export function shortcutFromKeyboardEvent(event: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'isComposing'>): string | null {
  if (event.isComposing) return null;
  const key = normalizeShortcutKey(event.key);
  if (['Meta', 'Control', 'Alt', 'Shift'].includes(key)) return null;
  const modifiers: string[] = [];
  if (event.metaKey || event.ctrlKey) modifiers.push('Mod');
  if (event.altKey) modifiers.push('Alt');
  if (event.shiftKey && (/^[A-Z0-9]$/.test(key) || key.length > 1)) modifiers.push('Shift');
  return [...modifiers, key].join('+');
}

export function isForbiddenShortcut(binding: string): boolean {
  return forbiddenBindings.has(binding);
}

export function sanitizeShortcutOverrides(value: unknown): ShortcutOverrides {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const result: ShortcutOverrides = {};
  for (const [rawAction, rawBinding] of Object.entries(value)) {
    if (!actionIds.has(rawAction as ShortcutActionId) || typeof rawBinding !== 'string') continue;
    const binding = rawBinding.trim();
    if (!binding || binding.length > 48 || isForbiddenShortcut(binding)) continue;
    result[rawAction as ShortcutActionId] = binding;
  }
  return result;
}

export function shortcutMatches(event: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'isComposing'>, binding: string): boolean {
  return shortcutFromKeyboardEvent(event) === binding;
}

export function resolveShortcutAction(
  event: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'isComposing'>,
  shortcuts: ShortcutMap,
  scope: 'app' | 'canvas',
): ShortcutActionId | null {
  const binding = shortcutFromKeyboardEvent(event);
  if (!binding) return null;
  return shortcutDefinitions.find((definition) => (
    (scope === 'canvas' || definition.scope === 'app')
    && (shortcuts[definition.id] === binding || definition.fixedAlternative === binding)
  ))?.id ?? null;
}

export function findShortcutConflict(shortcuts: ShortcutMap, action: ShortcutActionId, binding: string): ShortcutDefinition | null {
  const match = shortcutDefinitions.find((definition) => definition.id !== action && shortcuts[definition.id] === binding);
  return match ?? null;
}

export function displayShortcut(binding: string, isMac = true): string {
  return binding
    .replace('Mod', isMac ? '⌘' : 'Ctrl')
    .replace('Alt', isMac ? '⌥' : 'Alt')
    .replace('Shift', isMac ? '⇧' : 'Shift')
    .replace('ArrowLeft', '←')
    .replace('ArrowRight', '→')
    .replace('ArrowUp', '↑')
    .replace('ArrowDown', '↓');
}

export function shortcutAriaLabel(binding: string, isMac = true): string {
  return binding.replace('Mod', isMac ? 'Meta' : 'Control');
}
