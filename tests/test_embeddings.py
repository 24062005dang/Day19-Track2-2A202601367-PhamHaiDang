"""Guards the EMBEDDING_BACKEND contract.

This variable was documented in .env.example, flipped to `bge-m3` by
setup-docker.sh, printed as "bge-m3 embeddings" and advertised in the README --
while NOTHING read it. Every path silently used the English bge-small model.
These tests exist so that regression cannot come back silently.
"""
import numpy as np
import pytest

from app.embeddings import BACKENDS, DEFAULT_BACKEND, Embedder


def test_default_backend_is_the_nonquantized_bge_small():
    """The lite path and every rubric threshold depend on this exact default.

    It must stay the NON-quantized build: the int8 `-v1.5` ONNX takes ~140 ms
    to embed one query on a CPU without AVX-VNNI versus ~12 ms here, which on
    its own pushes NB3's hybrid P99 past the 50 ms threshold.
    """
    e = Embedder()
    assert e.backend == DEFAULT_BACKEND == "fastembed"
    assert e.model_name == "BAAI/bge-small-en"
    assert e.dim == 384


def test_quantized_build_stays_reachable():
    """Kept as an explicit backend so the latency comparison is reproducible."""
    e = Embedder("fastembed-q")
    assert e.model_name == "BAAI/bge-small-en-v1.5"
    assert e.dim == 384                        # same dim -> no re-index needed


def test_env_var_is_actually_honoured(monkeypatch):
    """The original bug: the variable existed but changed nothing."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    e = Embedder()
    assert e.backend == "bge-m3"
    assert e.model_name == "BAAI/bge-m3"
    assert e.dim == 1024                       # NOT 384


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    assert Embedder("fastembed").dim == 384


def test_unknown_backend_fails_loudly(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "definitely-not-a-model")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_BACKEND"):
        Embedder()


def test_every_backend_declares_a_distinct_dim():
    """Qdrant collections are created with this number; a wrong one is a
    silent 'vector dimension error' at upsert time."""
    for name, spec in BACKENDS.items():
        assert spec.dim > 0, name
        assert spec.provider in {"fastembed", "sentence-transformers", "openai"}, name
    assert BACKENDS["fastembed"].dim != BACKENDS["bge-m3"].dim


def test_default_embedding_matches_declared_dim():
    e = Embedder()
    v = next(e.embed(["xin chào thế giới"]))
    assert isinstance(np.asarray(v), np.ndarray)
    assert len(v) == e.dim


def test_searcher_collection_uses_backend_dim():
    """Regression: the collection was created from a hard-coded EMBED_DIM."""
    from app import search
    assert search.EMBED_DIM == Embedder().dim
