import os
import urllib.request

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from plumbline import models


class _StubEmbeddings:
    def embed_query(self, text):
        return [0.0] * 4


def test_chat_model_uses_env_string(monkeypatch):
    monkeypatch.delenv("INSIGHT_RAG_MODEL_BASE_URL", raising=False)
    seen = []
    monkeypatch.setenv("INSIGHT_RAG_LLM", "fake:x")
    monkeypatch.setattr(
        models,
        "init_chat_model",
        lambda s: seen.append(s) or FakeListChatModel(responses=["hi"]),
    )

    assert models.chat_model().invoke("hello").content == "hi"
    assert seen == ["fake:x"]


def test_embeddings_uses_env_string(monkeypatch):
    monkeypatch.delenv("INSIGHT_RAG_MODEL_BASE_URL", raising=False)
    seen = []
    monkeypatch.setenv("INSIGHT_RAG_EMBEDDINGS", "fake:y")
    monkeypatch.setattr(
        models,
        "init_embeddings",
        lambda s: seen.append(s) or _StubEmbeddings(),
    )

    assert len(models.embeddings().embed_query("hello")) == 4
    assert seen == ["fake:y"]


@pytest.mark.parametrize(
    ("factory", "var"),
    [
        (models.chat_model, "INSIGHT_RAG_LLM"),
        (models.embeddings, "INSIGHT_RAG_EMBEDDINGS"),
        (models.judge_model, "INSIGHT_RAG_JUDGE_LLM"),
    ],
)
def test_missing_env_raises_named_error(monkeypatch, factory, var):
    monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError, match=var):
        factory()


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except OSError:
        return False


@pytest.mark.llm
@pytest.mark.skipif(
    not (_ollama_up() and os.environ.get("INSIGHT_RAG_LLM", "").startswith("ollama:")),
    reason="needs Ollama on localhost:11434 and INSIGHT_RAG_LLM/EMBEDDINGS=ollama:*",
)
def test_ollama_roundtrip():
    assert models.chat_model().invoke("Reply with one word: ok").content
    assert len(models.embeddings().embed_query("hello")) > 0


def test_base_url_is_forwarded_when_set(monkeypatch):
    seen = {}
    monkeypatch.setenv("INSIGHT_RAG_LLM", "ollama:m")
    monkeypatch.setenv("INSIGHT_RAG_EMBEDDINGS", "ollama:e")
    monkeypatch.setenv(
        "INSIGHT_RAG_MODEL_BASE_URL", "http://host.docker.internal:11434"
    )
    monkeypatch.setattr(models, "init_chat_model", lambda s, **kw: seen.update(chat=kw))
    monkeypatch.setattr(models, "init_embeddings", lambda s, **kw: seen.update(emb=kw))

    models.chat_model()
    models.embeddings()

    expected = {"base_url": "http://host.docker.internal:11434"}
    assert seen == {"chat": expected, "emb": expected}
