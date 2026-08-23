/**
 * Typed client for the VectorMind API.
 *
 * Mirrors the Pydantic schemas in `backend/schemas.py`. This is the only
 * place the frontend encodes what the API looks like, so a contract
 * change should break here rather than in a component.
 *
 * Every request is bounded by a timeout and cancellable. Without one, a
 * backend that has loaded its model but not its index leaves the UI
 * spinning indefinitely — which is exactly what a cold start looks like,
 * and the state the demo is most likely to be seen in.
 */

import type {
  HealthResponse,
  SearchResponse,
  TextSearchRequest,
} from '../types/search';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

/**
 * Request timeouts, in milliseconds.
 *
 * Search is generous because the first query after startup pays for a
 * cold model: the measured p95 is ~25ms but the cold-start maximum was
 * 3.4s. Health is short because it is polled, and a slow answer to
 * "are you up" is itself the answer.
 */
export const TIMEOUTS = {
  health: 5_000,
  search: 30_000,
} as const;

/** An API call that failed, carrying the HTTP status when there was one. */
export class ApiError extends Error {
  readonly status: number | null;
  readonly isTimeout: boolean;

  constructor(
    message: string,
    options: { status?: number | null; isTimeout?: boolean } = {}
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? null;
    this.isTimeout = options.isTimeout ?? false;
  }

  /** A message worth showing a visitor, rather than a raw status code. */
  get userMessage(): string {
    if (this.isTimeout) {
      return 'The server took too long to respond. It may still be loading the model — try again in a moment.';
    }
    if (this.status === 503) {
      return 'The search service is starting up. Its model or index is not loaded yet.';
    }
    if (this.status === 413) {
      return 'That image is too large. Try one under 10MB.';
    }
    if (this.status === 429) {
      return 'Too many requests. Give it a few seconds.';
    }
    if (this.status !== null && this.status >= 500) {
      return 'The server hit an error handling that request.';
    }
    return this.message;
  }
}

async function request<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<T> {
  // AbortSignal.timeout() is cleaner but leaves callers unable to cancel
  // for their own reasons, so the controller is explicit.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${url}`, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(`Request to ${url} timed out`, { isTimeout: true });
    }
    // fetch rejects with TypeError for DNS, CORS, and offline failures,
    // none of which carry a status.
    throw new ApiError(
      error instanceof Error ? error.message : 'Network request failed'
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      (body && (body.detail || body.error)) ||
      `HTTP ${response.status}: ${response.statusText}`;
    throw new ApiError(String(detail), { status: response.status });
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { method: 'GET' }, TIMEOUTS.health);
}

export async function searchText(
  req: TextSearchRequest
): Promise<SearchResponse> {
  return request<SearchResponse>(
    '/search/text',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    },
    TIMEOUTS.search
  );
}

export async function searchImage(
  file: File,
  topK = 10
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append('file', file);

  // No Content-Type header: the browser must set it itself to include
  // the multipart boundary, and setting it manually breaks the upload.
  return request<SearchResponse>(
    `/search/image?top_k=${topK}`,
    { method: 'POST', body: formData },
    TIMEOUTS.search
  );
}
