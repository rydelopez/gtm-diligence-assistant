from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


def _default_embedding_provider(explicit_provider: str | None = None) -> str:
    if explicit_provider:
        return explicit_provider.strip().lower()

    configured_provider = os.environ.get("EMBEDDING_PROVIDER")
    if configured_provider:
        return configured_provider.strip().lower()

    model_provider = os.environ.get("MODEL", "openai").strip().lower()
    if model_provider in {"openai", "google"}:
        return model_provider

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    return "openai"


def create_embedding_model(provider: str | None = None) -> Any:
    embedding_provider = _default_embedding_provider(provider)

    if embedding_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embeddings. "
                "Set EMBEDDING_PROVIDER=google to use Google embeddings instead."
            )
        return OpenAIEmbeddings(
            model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            chunk_size=int(os.environ.get("EMBEDDING_CHUNK_SIZE", "16")),
        )

    if embedding_provider == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Google embeddings. "
                "Set EMBEDDING_PROVIDER=openai to use OpenAI embeddings instead."
            )
        return GoogleGenerativeAIEmbeddings(
            model=os.environ.get("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        )

    if embedding_provider == "fake":
        return DeterministicFakeEmbedding(size=int(os.environ.get("FAKE_EMBEDDING_SIZE", "256")))

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER '{embedding_provider}'. "
        "Supported values are openai, google, and fake."
    )
