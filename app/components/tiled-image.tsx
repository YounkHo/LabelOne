'use client';

import { useLayoutEffect, useRef, useState, type RefObject } from 'react';

import type { TileMetadata } from '../lib/contracts';

type CanvasView = { scale: number; x: number; y: number };

type VisibleTile = {
  key: string;
  level: number;
  x: number;
  y: number;
  left: number;
  top: number;
  width: number;
  height: number;
  url: string;
};

type TiledImageProps = {
  assetKey: string;
  alt: string;
  metadata: TileMetadata;
  placeholderUrl?: string | null;
  tileUrl: (level: number, x: number, y: number, format: string) => string | null;
  view: CanvasView;
  viewportRef: RefObject<HTMLElement | null>;
};

function sameTiles(left: VisibleTile[], right: VisibleTile[]) {
  return left.length === right.length && left.every((tile, index) => tile.key === right[index]?.key);
}

export function TiledImage({
  assetKey,
  alt,
  metadata,
  placeholderUrl,
  tileUrl,
  view,
  viewportRef,
}: TiledImageProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [tiles, setTiles] = useState<VisibleTile[]>([]);
  const [renderLevel, setRenderLevel] = useState(0);

  useLayoutEffect(() => {
    let frame = 0;
    const update = () => {
      const root = rootRef.current;
      const viewport = viewportRef.current;
      if (!root || !viewport) return;
      const rootRect = root.getBoundingClientRect();
      const viewportRect = viewport.getBoundingClientRect();
      if (rootRect.width <= 0 || rootRect.height <= 0) return;

      const visibleLeft = Math.max(rootRect.left, viewportRect.left);
      const visibleTop = Math.max(rootRect.top, viewportRect.top);
      const visibleRight = Math.min(rootRect.right, viewportRect.right);
      const visibleBottom = Math.min(rootRect.bottom, viewportRect.bottom);
      if (visibleLeft >= visibleRight || visibleTop >= visibleBottom) {
        setTiles((current) => current.length ? [] : current);
        return;
      }

      const cssPixelsPerSourcePixel = rootRect.width / metadata.width;
      const idealLevel = metadata.max_level + Math.log2(Math.max(cssPixelsPerSourcePixel, Number.EPSILON));
      const level = Math.max(0, Math.min(metadata.max_level, Math.round(idealLevel)));
      const downsample = 2 ** (metadata.max_level - level);
      const levelWidth = Math.max(1, Math.ceil(metadata.width / downsample));
      const levelHeight = Math.max(1, Math.ceil(metadata.height / downsample));
      const totalColumns = Math.ceil(levelWidth / metadata.tile_size);
      const totalRows = Math.ceil(levelHeight / metadata.tile_size);
      const localLeft = (visibleLeft - rootRect.left) / rootRect.width * levelWidth;
      const localTop = (visibleTop - rootRect.top) / rootRect.height * levelHeight;
      const localRight = (visibleRight - rootRect.left) / rootRect.width * levelWidth;
      const localBottom = (visibleBottom - rootRect.top) / rootRect.height * levelHeight;
      const minX = Math.max(0, Math.floor(localLeft / metadata.tile_size) - 1);
      const minY = Math.max(0, Math.floor(localTop / metadata.tile_size) - 1);
      const maxX = Math.min(totalColumns - 1, Math.floor(Math.max(0, localRight - 1) / metadata.tile_size) + 1);
      const maxY = Math.min(totalRows - 1, Math.floor(Math.max(0, localBottom - 1) / metadata.tile_size) + 1);
      const next: VisibleTile[] = [];
      for (let y = minY; y <= maxY; y += 1) {
        for (let x = minX; x <= maxX; x += 1) {
          const url = tileUrl(level, x, y, metadata.format || 'webp');
          if (!url) continue;
          const pixelLeft = x * metadata.tile_size;
          const pixelTop = y * metadata.tile_size;
          const pixelWidth = Math.min(metadata.tile_size, levelWidth - pixelLeft);
          const pixelHeight = Math.min(metadata.tile_size, levelHeight - pixelTop);
          next.push({
            key: `${assetKey}:${metadata.source_etag}:${level}:${x}:${y}:${metadata.format}`,
            level,
            x,
            y,
            left: pixelLeft / levelWidth * 100,
            top: pixelTop / levelHeight * 100,
            width: pixelWidth / levelWidth * 100,
            height: pixelHeight / levelHeight * 100,
            url,
          });
        }
      }
      setRenderLevel(level);
      setTiles((current) => sameTiles(current, next) ? current : next);
    };
    const schedule = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(update);
    };
    const observer = new ResizeObserver(schedule);
    if (rootRef.current) observer.observe(rootRef.current);
    if (viewportRef.current) observer.observe(viewportRef.current);
    schedule();
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [assetKey, metadata, tileUrl, view.scale, view.x, view.y, viewportRef]);

  return (
    <div ref={rootRef} className="tiled-image" role="img" aria-label={alt} data-level={renderLevel}>
      {placeholderUrl && (
        // eslint-disable-next-line @next/next/no-img-element -- local thumbnail is a temporary low-resolution canvas placeholder
        <img className="tiled-image-placeholder" src={placeholderUrl} alt="" crossOrigin="anonymous" draggable={false} aria-hidden="true" />
      )}
      {tiles.map((tile) => (
        // eslint-disable-next-line @next/next/no-img-element -- browser-native lazy loading is required for local image tiles
        <img
          key={tile.key}
          className="tiled-image-tile"
          src={tile.url}
          alt=""
          crossOrigin="anonymous"
          aria-hidden="true"
          draggable={false}
          loading="lazy"
          decoding="async"
          data-level={tile.level}
          data-x={tile.x}
          data-y={tile.y}
          style={{
            left: `${tile.left}%`,
            top: `${tile.top}%`,
            width: `calc(${tile.width}% + .5px)`,
            height: `calc(${tile.height}% + .5px)`,
            imageRendering: tile.level === metadata.max_level && view.scale > 1 ? 'pixelated' : 'auto',
          }}
        />
      ))}
    </div>
  );
}
