import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveCurrentFilePath } from './current-file-path.ts';

test('prefers the real image path for full-path copying', () => {
  assert.deepEqual(resolveCurrentFilePath({
    datasetPath: '/datasets/inspection',
    displayPath: 'zone/wafer.tif',
    fileName: 'wafer.tif',
    imagePath: '/mnt/images/zone/wafer.tif',
  }), {
    fullPath: '/mnt/images/zone/wafer.tif',
    directoryPath: '/mnt/images/zone',
    directoryLabel: '/mnt/images/zone/',
    fileName: 'wafer.tif',
    isAbsolute: true,
  });
});

test('combines the dataset and display paths when no real image path is available', () => {
  assert.deepEqual(resolveCurrentFilePath({
    datasetPath: '/datasets/inspection/',
    displayPath: '/zone/wafer.tif',
    fileName: 'wafer.tif',
  }), {
    fullPath: '/datasets/inspection/zone/wafer.tif',
    directoryPath: '/datasets/inspection/zone',
    directoryLabel: '/datasets/inspection/zone/',
    fileName: 'wafer.tif',
    isAbsolute: true,
  });
});

test('keeps relative project paths descriptive without claiming they are absolute', () => {
  assert.deepEqual(resolveCurrentFilePath({ datasetPath: 'inspection-project', fileName: 'image.tif' }), {
    fullPath: 'inspection-project/image.tif',
    directoryPath: 'inspection-project',
    directoryLabel: 'inspection-project/',
    fileName: 'image.tif',
    isAbsolute: false,
  });
});

test('keeps the native directory separator in the visible path segment', () => {
  assert.equal(resolveCurrentFilePath({ datasetPath: 'C:\\images', fileName: 'chip.png', imagePath: 'C:\\images\\chip.png' }).directoryLabel, 'C:\\images\\');
  assert.equal(resolveCurrentFilePath({ datasetPath: '/', fileName: 'chip.png', imagePath: '/chip.png' }).directoryLabel, '/');
});
