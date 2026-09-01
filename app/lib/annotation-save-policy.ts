export const DEFAULT_ANNOTATION_AUTO_SAVE = false;

export function shouldWriteAnnotationFile(manual: boolean, autoSaveEnabled: boolean): boolean {
  return manual || autoSaveEnabled;
}
