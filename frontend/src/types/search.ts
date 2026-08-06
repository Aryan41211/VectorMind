/**
 * VectorMind Search Types — Mirroring Pydantic schemas
 */

export interface SearchResult {
  index: number;
  score: number;
  caption?: string;
  image_path?: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  search_type: 'text_to_image' | 'image_to_text';
  total_results: number;
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  index_loaded: boolean;
  device: string;
  num_indexed_images: number;
}

export interface TextSearchRequest {
  query: string;
  top_k?: number;
}

export interface ApiError {
  error: string;
  detail?: string;
}
