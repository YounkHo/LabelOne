export const DEFAULT_ANNOTATION_AUTO_SAVE = false;

export function normalizeAnnotationAutoSavePreference(value: unknown): boolean {
  return typeof value === 'boolean' ? value : DEFAULT_ANNOTATION_AUTO_SAVE;
}

export function shouldWriteAnnotationFile(manual: boolean, autoSaveEnabled: boolean): boolean {
  return manual || autoSaveEnabled;
}
