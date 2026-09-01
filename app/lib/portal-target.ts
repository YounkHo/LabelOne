export function fullscreenPortalTarget<T>(documentLike: { fullscreenElement: T | null; body: T }): T {
  return documentLike.fullscreenElement ?? documentLike.body;
}
