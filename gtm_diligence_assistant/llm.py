from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def create_chat_model(provider: str | None = None):
    model_provider = (provider or os.environ.get("MODEL", "openai")).strip().lower()
    temperature = float(os.environ.get("MODEL_TEMPERATURE", "0"))

    if model_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when MODEL=openai.")
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-5"), temperature=temperature)

    if model_provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required when MODEL=anthropic.")
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            temperature=temperature,
        )

    if model_provider == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required when MODEL=google.")
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro"),
            temperature=temperature,
        )

    raise RuntimeError(f"Unsupported MODEL provider: {model_provider}")
