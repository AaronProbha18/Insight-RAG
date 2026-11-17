"""Chat and embedding models chosen by `provider:model` strings in the environment."""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; expected a 'provider:model' string")
    return value


def _options() -> dict:
    # e.g. http://host.docker.internal:11434 when Ollama runs outside the container.
    base_url = os.environ.get("INSIGHT_RAG_MODEL_BASE_URL")
    return {"base_url": base_url} if base_url else {}


def chat_model():
    return init_chat_model(_env("INSIGHT_RAG_LLM"), **_options())


def embeddings():
    return init_embeddings(_env("INSIGHT_RAG_EMBEDDINGS"), **_options())


def judge_model():
    """Eval judge. Deliberately separate from INSIGHT_RAG_LLM so it never drifts with it."""
    return init_chat_model(_env("INSIGHT_RAG_JUDGE_LLM"), **_options())
