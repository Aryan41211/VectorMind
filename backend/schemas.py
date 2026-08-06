"""
VectorMind Pydantic Schemas — API Request/Response Models

Defines the request and response schemas for the search API endpoints.
Ensures type safety and automatic OpenAPI documentation generation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Request Schemas ---

class TextSearchRequest(BaseModel):
    """Request schema for text-based image search."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text query for image search",
        examples=["a dog playing in the park"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
        examples=[10],
    )


class ImageSearchRequest(BaseModel):
    """Request schema for image-based text search (via form data)."""
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
        examples=[10],
    )


# --- Response Schemas ---

class SearchResult(BaseModel):
    """A single search result."""
    index: int = Field(
        ...,
        description="Index of the result in the dataset",
        examples=[42],
    )
    score: float = Field(
        ...,
        description="Similarity score (higher = more similar)",
        examples=[0.85],
    )
    caption: str | None = Field(
        default=None,
        description="Associated caption (for image→text results)",
        examples=["a dog playing in the park"],
    )
    image_path: str | None = Field(
        default=None,
        description="Path to the image file (for text→image results)",
        examples=["data/flickr30k/images/12345.jpg"],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the result",
    )


class SearchResponse(BaseModel):
    """Response schema for search queries."""
    results: list[SearchResult] = Field(
        ...,
        description="Ranked search results",
    )
    query: str = Field(
        ...,
        description="Original query (text or 'image:{hash}')",
        examples=["a dog playing in the park"],
    )
    search_type: str = Field(
        ...,
        description="Type of search performed",
        examples=["text_to_image"],
    )
    total_results: int = Field(
        ...,
        description="Total number of results returned",
        examples=[10],
    )
    latency_ms: float = Field(
        ...,
        description="Search latency in milliseconds",
        examples=[15.5],
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        default="healthy",
        description="Service status",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded",
    )
    index_loaded: bool = Field(
        ...,
        description="Whether the FAISS index is loaded",
    )
    device: str = Field(
        ...,
        description="Device the model is running on",
        examples=["cuda"],
    )
    num_indexed_images: int = Field(
        ...,
        description="Number of images in the index",
        examples=[1000],
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(
        ...,
        description="Error message",
    )
    detail: str | None = Field(
        default=None,
        description="Detailed error information",
    )


# --- Configuration Schemas ---

class ModelConfig(BaseModel):
    """Model configuration for serving."""
    checkpoint_path: str = Field(
        ...,
        description="Path to the model checkpoint",
    )
    device: str = Field(
        default="auto",
        description="Device to run inference on (auto/cpu/cuda)",
    )
    index_path: str = Field(
        ...,
        description="Path to the FAISS index directory",
    )


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )
    port: int = Field(
        default=8000,
        description="Server port",
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes",
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )
