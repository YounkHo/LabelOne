export function setShapeVisibility(hiddenIndexes: ReadonlySet<number>, index: number, visible: boolean): Set<number> {
  const next = new Set(hiddenIndexes);
  if (visible) next.delete(index);
  else next.add(index);
  return next;
}

export function hideAllShapes(shapeCount: number): Set<number> {
  return new Set(Array.from({ length: Math.max(0, shapeCount) }, (_, index) => index));
}

export function remapHiddenShapesAfterDeletion(hiddenIndexes: ReadonlySet<number>, deletedIndex: number): Set<number> {
  return remapHiddenShapesAfterDeletions(hiddenIndexes, [deletedIndex]);
}

function sortedDeletionIndexes(deletedIndexes: Iterable<number>): number[] {
  return [...new Set(deletedIndexes)].sort((left, right) => left - right);
}

function remapShapeIndex(index: number, deletedIndexes: number[]): number | null {
  let low = 0;
  let high = deletedIndexes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (deletedIndexes[middle] < index) low = middle + 1;
    else high = middle;
  }
  return deletedIndexes[low] === index ? null : index - low;
}

export function remapHiddenShapesAfterDeletions(hiddenIndexes: ReadonlySet<number>, deletedIndexes: Iterable<number>): Set<number> {
  const deleted = sortedDeletionIndexes(deletedIndexes);
  const next = new Set<number>();
  hiddenIndexes.forEach((index) => {
    const remapped = remapShapeIndex(index, deleted);
    if (remapped !== null) next.add(remapped);
  });
  return next;
}

export function remapSelectedShapeAfterDeletion(selectedIndex: number | null, deletedIndex: number): number | null {
  return remapSelectedShapeAfterDeletions(selectedIndex, [deletedIndex]);
}

export function remapSelectedShapeAfterDeletions(selectedIndex: number | null, deletedIndexes: Iterable<number>): number | null {
  if (selectedIndex === null) return null;
  return remapShapeIndex(selectedIndex, sortedDeletionIndexes(deletedIndexes));
}
