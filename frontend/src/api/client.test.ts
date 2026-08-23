/**
 * Tests for the typed API client.
 *
 * This layer is where a backend contract change shows up first: it is the
 * only place the frontend encodes what /search/text, /search/image and
 * /health look like. The Python side has 393 tests; this file exists so a
 * schema change breaks a test rather than the running demo.
 */

import { describe, expect, it, vi } from 'vitest';
import { getHealth, searchImage, searchText } from './client';
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
    expect(fetchMock).toHaveBeenCalledWith('/health');
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

    expect(fetchMock).toHaveBeenCalledWith('/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'a busy market', top_k: 10 }),
    });
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
    const body = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0][1].body;
    expect(JSON.parse(body)).toEqual({ query: 'x' });
  });
});

describe('searchImage', () => {
  const file = new File(['bytes'], 'photo.jpg', { type: 'image/jpeg' });

  it('POSTs multipart form data with top_k in the query string', async () => {
    const fetchMock = mockFetch({ ...SEARCH, search_type: 'image_to_text' });
    await searchImage(file, 5);

    const [url, init] = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0];
    expect(url).toBe('/search/image?top_k=5');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBe(file);
  });

  it('defaults top_k to 10', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchImage(file);
    expect(
      (fetchMock as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0]
    ).toBe('/search/image?top_k=10');
  });

  it('does not set Content-Type, so the browser can add the boundary', async () => {
    const fetchMock = mockFetch(SEARCH);
    await searchImage(file);
    const init = (fetchMock as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0][1];
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
