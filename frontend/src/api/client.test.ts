/**
 * Tests for the typed API client.
 *
 * This layer is where a backend contract change shows up first: it is the
 * only place the frontend encodes what /search/text, /search/image and
 * /health look like. The Python side has 393 tests; this file exists so a
 * schema change breaks a test rather than the running demo.
 */

import { describe, expect, it, vi } from 'vitest';
import { ApiError, TIMEOUTS, getHealth, searchImage, searchText } from './client';
import type { HealthResponse, SearchResponse } from '../types/search';

const HEALTH: HealthResponse = {
  status: 'ok',
  model_loaded: true,
  index_loaded: true,
  device: 'cpu',
  num_indexed_images: 3179,
};

const SEARCH: SearchResponse = {
  results: [
    {
      rank: 1,
      index: 7,
      score: 0.84,
      filename: '022545.jpg',
      image_url: '/images/022545.jpg',
      caption: 'A crowd of people shopping at a street market.',
    },
  ],
  query: 'a busy market',
  search_type: 'text_to_image',
  total_results: 1,
  latency_ms: 14.2,
};

/** First call's (url, init). Every request carries an AbortSignal for
 *  the timeout, so exact-object assertions on init are not usable. */
function callArgs(fetchMock: unknown): [string, RequestInit] {
  return (fetchMock as ReturnType<typeof vi.fn>).mock.calls[0] as [
    string,
    RequestInit,
  ];
}

function mockFetch(body: unknown, init: Partial<Response> = {}): typeof fetch {
  const fn = vi.fn().mockResolvedValue({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: async () => body,
  });
  vi.stubGlobal('fetch', fn);
  return fn as unknown as typeof fetch;
}

describe('getHealth', () => {
  it('requests /health', async () => {
    const fetchMock = mockFetch(HEALTH);
    await getHealth();
    const [url, init] = callArgs(fetchMock);
    expect(url).toBe('/health');
    expect(init.method).toBe('GET');
  });

  it('returns the parsed body', async () => {
    mockFetch(HEALTH);
    await expect(getHealth()).resolves.toEqual(HEALTH);
  });

  it('surfaces the index size the UI displays', async () => {
    mockFetch(HEALTH);
    const health = await getHealth();
    expect(health.num_indexed_images).toBe(3179);
  });
});

describe('searchText', () => {
  it('POSTs JSON to /search/text', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchText({ query: 'a busy market', top_k: 10 });

    const [url, init] = callArgs(fetchMock);
    expect(url).toBe('/search/text');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(init.body).toBe(
      JSON.stringify({ query: 'a busy market', top_k: 10 })
    );
  });

  it('returns ranked results', async () => {
    mockFetch(SEARCH);
    const response = await searchText({ query: 'a busy market' });
    expect(response.results).toHaveLength(1);
    expect(response.results[0].rank).toBe(1);
  });

  it('preserves the image_url the grid renders', async () => {
    mockFetch(SEARCH);
    const response = await searchText({ query: 'a busy market' });
    expect(response.results[0].image_url).toBe('/images/022545.jpg');
  });

  it('omits top_k when the caller does not set one', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchText({ query: 'x' });
    const [, init] = callArgs(fetchMock);
    expect(JSON.parse(init.body as string)).toEqual({ query: 'x' });
  });
});

describe('searchImage', () => {
  const file = new File(['bytes'], 'photo.jpg', { type: 'image/jpeg' });

  it('POSTs multipart form data with top_k in the query string', async () => {
    const fetchMock = mockFetch({ ...SEARCH, search_type: 'image_to_text' });
    await searchImage(file, 5);

    const [url, init] = callArgs(fetchMock);
    expect(url).toBe('/search/image?top_k=5');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBe(file);
  });

  it('defaults top_k to 10', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchImage(file);
    expect(callArgs(fetchMock)[0]).toBe('/search/image?top_k=10');
  });

  it('does not set Content-Type, so the browser can add the boundary', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchImage(file);
    const [, init] = callArgs(fetchMock);
    expect(init.headers).toBeUndefined();
  });
});

describe('error handling', () => {
  it('throws the API detail message when present', async () => {
    mockFetch({ detail: 'Service unavailable: model or index not loaded' }, {
      ok: false,
      status: 503,
    });
    await expect(searchText({ query: 'x' })).rejects.toThrow(
      'Service unavailable: model or index not loaded'
    );
  });

  it('falls back to the error field', async () => {
    mockFetch({ error: 'boom' }, { ok: false, status: 500 });
    await expect(searchText({ query: 'x' })).rejects.toThrow('boom');
  });

  it('falls back to status text when the body is not JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: 'Bad Gateway',
        json: async () => {
          throw new Error('not json');
        },
      })
    );
    await expect(getHealth()).rejects.toThrow('HTTP 502: Bad Gateway');
  });

  it('propagates network failures rather than swallowing them', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')));
    await expect(getHealth()).rejects.toThrow('offline');
  });
});

describe('ApiError', () => {
  it('is the error type thrown, carrying the status', async () => {
    mockFetch({ detail: 'nope' }, { ok: false, status: 503 });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(503);
    expect(error.isTimeout).toBe(false);
  });

  it('explains a 503 as a cold start rather than showing the code', async () => {
    mockFetch({ detail: 'raw backend text' }, { ok: false, status: 503 });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error.userMessage).toMatch(/starting up/i);
  });

  it('explains an oversized upload', async () => {
    mockFetch({}, { ok: false, status: 413 });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error.userMessage).toMatch(/too large/i);
  });

  it('explains rate limiting', async () => {
    mockFetch({}, { ok: false, status: 429 });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error.userMessage).toMatch(/too many requests/i);
  });

  it('generalises any 5xx rather than leaking internals', async () => {
    mockFetch({ detail: 'Traceback (most recent call last)...' }, {
      ok: false,
      status: 500,
    });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error.userMessage).toBe('The server hit an error handling that request.');
  });

  it('passes a 4xx message through unchanged', async () => {
    mockFetch({ detail: 'query must not be empty' }, { ok: false, status: 422 });
    const error = await searchText({ query: 'x' }).catch((e) => e);
    expect(error.userMessage).toBe('query must not be empty');
  });
});

describe('timeouts', () => {
  it('aborts and reports a timeout rather than hanging', async () => {
    // A backend that accepts the connection and never answers is exactly
    // what a cold start looks like; without the timeout the UI spins forever.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url: string, init: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError'))
          );
        });
      })
    );

    vi.useFakeTimers();
    const pending = getHealth().catch((e) => e);
    await vi.advanceTimersByTimeAsync(TIMEOUTS.health + 10);
    const error = await pending;
    vi.useRealTimers();

    expect(error).toBeInstanceOf(ApiError);
    expect(error.isTimeout).toBe(true);
    expect(error.userMessage).toMatch(/took too long/i);
  });

  it('passes an abort signal on every request', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchText({ query: 'x' });
    const [, init] = callArgs(fetchMock);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});
