export type FileAnnotationFilter = 'all' | 'with_json' | 'without_json';

export const fileAnnotationFilters: FileAnnotationFilter[] = ['all', 'with_json', 'without_json'];

export const fileAnnotationFilterLabels: Record<FileAnnotationFilter, string> = {
  all: '全部文件',
  with_json: '有 JSON',
  without_json: '无 JSON',
};

export function matchesFileAnnotationFilter(filter: FileAnnotationFilter, annotationFileExists: boolean): boolean {
  if (filter === 'with_json') return annotationFileExists;
  if (filter === 'without_json') return !annotationFileExists;
  return true;
}
