/**
 * VectorMind API Client — Typed fetch wrapper
 */

import type {
  HealthResponse,
  SearchResponse,
  TextSearchRequest,
} from '../types/search';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({
      error: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new Error(error.detail || error.error || 'Request failed');
  }
  return response.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(response);
}

export async function searchText(
  request: TextSearchRequest
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<SearchResponse>(response);
}

export async function searchImage(
  file: File,
  topK: number = 10
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/search/image?top_k=${topK}`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<SearchResponse>(response);
}
