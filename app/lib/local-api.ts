import type { ApiError } from './contracts';

export class LocalApiError extends Error {
  code: string;
  details?: Record<string, unknown>;

  constructor(error: ApiError) {
    super(error.message);
    this.name = 'LocalApiError';
    this.code = error.code;
    this.details = error.details;
  }
}

export function localApiBase(): string | null {
  if (typeof window === 'undefined') return null;
  if (!['localhost', '127.0.0.1'].includes(window.location.hostname)) return null;
  const queryOverride = new URLSearchParams(window.location.search).get('api');
  const configured = queryOverride ?? window.localStorage.getItem('labelone-api-base') ?? 'http://127.0.0.1:8766/api/v1';
  try {
    const url = new URL(configured);
    if (url.protocol !== 'http:' || !['localhost', '127.0.0.1'].includes(url.hostname)) return null;
    const normalized = url.toString().replace(/\/$/, '');
    if (queryOverride) window.localStorage.setItem('labelone-api-base', normalized);
    return normalized;
  } catch {
    return null;
  }
}

export async function localRequest<T>(
  base: string,
  path: string,
  init: RequestInit = {},
  timeoutMs = 3000,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const callerSignal = init.signal;
  const abortFromCaller = () => controller.abort();
  callerSignal?.addEventListener('abort', abortFromCaller, { once: true });
  try {
    const response = await fetch(`${base}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...init.headers,
      },
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new LocalApiError({
        code: payload?.code ?? `http_${response.status}`,
        message: payload?.message ?? `Local API request failed (${response.status})`,
        details: payload?.details,
      });
    }
    return payload as T;
  } catch (error) {
    if (error instanceof LocalApiError) throw error;
    if (controller.signal.aborted) throw new LocalApiError({ code: 'request_aborted', message: 'Local API request was cancelled or timed out' });
    throw new LocalApiError({ code: 'local_service_unreachable', message: error instanceof Error ? error.message : 'Local service is unreachable' });
  } finally {
    window.clearTimeout(timeout);
    callerSignal?.removeEventListener('abort', abortFromCaller);
  }
}
