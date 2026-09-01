export function normalizeAnnotationCategory(label: string): string {
  return label.trim().normalize('NFC') || '未命名';
}

export function annotationIndexesForCategory(labels: string[], category: string): number[] {
  const normalizedCategory = category.trim() ? normalizeAnnotationCategory(category) : '';
  return labels.flatMap((label, index) => !normalizedCategory || normalizeAnnotationCategory(label) === normalizedCategory ? [index] : []);
}

export function setAnnotationIndexesVisible(hiddenIndexes: ReadonlySet<number>, indexes: number[], visible: boolean): Set<number> {
  const next = new Set(hiddenIndexes);
  indexes.forEach((index) => visible ? next.delete(index) : next.add(index));
  return next;
}

export function renameAnnotationCategory<T extends { label: string }>(shapes: readonly T[], sourceCategory: string, targetLabel: string): { shapes: T[]; indexes: number[] } {
  const normalizedSource = normalizeAnnotationCategory(sourceCategory);
  const normalizedTarget = targetLabel.trim().normalize('NFC');
  const indexes: number[] = [];
  const nextShapes = shapes.map((shape, index) => {
    if (normalizeAnnotationCategory(shape.label) !== normalizedSource) return shape;
    indexes.push(index);
    return shape.label === normalizedTarget ? shape : { ...shape, label: normalizedTarget };
  });
  return { shapes: nextShapes, indexes };
}
