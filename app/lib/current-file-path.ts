export type CurrentFilePath = {
  fullPath: string;
  directoryPath: string;
  directoryLabel: string;
  fileName: string;
  isAbsolute: boolean;
};

type CurrentFilePathInput = {
  datasetPath: string;
  fileName?: string;
  displayPath?: string;
  imagePath?: string;
};

function joinFilePath(root: string, relative: string) {
  const separator = root.includes('\\') && !root.includes('/') ? '\\' : '/';
  return `${root.replace(/[\\/]+$/, '')}${separator}${relative.replace(/^[\\/]+/, '')}`;
}

export function resolveCurrentFilePath({ datasetPath, fileName = '', displayPath = '', imagePath = '' }: CurrentFilePathInput): CurrentFilePath {
  const relativePath = displayPath.trim() || fileName.trim();
  const fullPath = imagePath.trim() || (datasetPath.trim() && relativePath ? joinFilePath(datasetPath.trim(), relativePath) : relativePath || datasetPath.trim());
  const separatorIndex = Math.max(fullPath.lastIndexOf('/'), fullPath.lastIndexOf('\\'));
  const directoryPath = separatorIndex >= 0 ? fullPath.slice(0, separatorIndex) : '';
  const directoryLabel = separatorIndex === 0 ? fullPath[0] : directoryPath ? `${directoryPath}${fullPath[separatorIndex]}` : '';
  const resolvedName = fileName.trim() || (separatorIndex >= 0 ? fullPath.slice(separatorIndex + 1) : fullPath);
  return {
    fullPath,
    directoryPath,
    directoryLabel,
    fileName: resolvedName,
    isAbsolute: /^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(fullPath),
  };
}
