"""Sentence similarity for the counterfactual filter.

Only the similarity is stored, alongside the identity that produced it, so a rescore under a
different embedder is visible in the artifacts rather than silent.

MiniLM was the original choice for being local and small, but it scores paraphrases of clinical
text well below a general-domain reader would - "Cognitive function spared" against "Cognitive
function *not* affected" comes out at 0.70 - so rollouts that merely reword the sentence land in
the counterfactual arm. A larger embedder separates rewording from genuine divergence better;
set DCTAX_EMBED_MODEL to a sentence-transformers id to go back.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dctax.embeddings.cache import EmbeddingIdentity

MODEL = os.environ.get("DCTAX_EMBED_MODEL", "openai:text-embedding-3-large")
BATCH = 256
#: Sentences per OpenAI request. The endpoint caps a batch by tokens, not count, and openings
#: run to a couple of hundred characters, so this stays well inside it.
API_BATCH = 512
#: Batches in flight at once. Embedding a batch is almost entirely network wait - one model
#: measured 71 minutes of wall clock against 10 minutes of CPU - and the batches are
#: independent, so issuing them concurrently is the difference between hours and minutes.
#:
#: The ceiling is tokens, not requests: the account allows 1M tokens and 5000 requests per
#: minute, and a 512-sentence batch of ~13-token openings is only ~6,800 tokens, so ~147
#: batches per minute is the limit. At ~10s per batch that is ~24 in flight. Set
#: DCTAX_EMBED_CONCURRENCY lower when several scorers run at once, since they share the budget.
API_CONCURRENCY = int(os.environ.get("DCTAX_EMBED_CONCURRENCY", "12"))


#: Native width of each OpenAI embedding model, since the API does not report it up front.
OPENAI_DIMENSIONS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}


def _resolved_revision(model_id: str) -> str | None:
    """Commit sha of the cached snapshot, or None when it cannot be resolved."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:  # noqa: BLE001: provenance is recorded when available, never required
        return None


class SemanticMatcher:
    """Cosine similarity between sentences, batched and deduplicated."""

    def __init__(self, model_id: str = MODEL, *, device: str | None = None, batch: int = BATCH):
        self.model_id = model_id
        self.batch = batch
        self._openai = model_id.startswith("openai:")
        if self._openai:
            from openai import OpenAI

            from dctax.utils.llm import api_key

            self._name = model_id.split(":", 1)[1]
            self._client = OpenAI(api_key=api_key("OPENAI_API_KEY"))
            self._model = None
            self.identity = EmbeddingIdentity(
                provider="openai", model=self._name,
                dimensions=OPENAI_DIMENSIONS.get(self._name, 3072), revision=None,
            )
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_id, device=device)
            self.identity = EmbeddingIdentity(
                provider="huggingface",
                model=model_id,
                dimensions=int(self._model.get_sentence_embedding_dimension()),
                revision=_resolved_revision(model_id),
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-norm vectors, so a dot product is the cosine similarity."""
        if not texts:
            return np.empty((0, self.identity.dimensions), dtype=np.float32)
        if self._openai:
            return self._encode_openai(texts)
        return np.asarray(
            self._model.encode(
                texts, batch_size=self.batch, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        """Embeddings from the API, in order, retrying a batch that fails once."""
        out = np.empty((len(texts), self.identity.dimensions), dtype=np.float32)
        starts = list(range(0, len(texts), API_BATCH))

        def embed(start: int) -> tuple[int, list]:
            chunk = texts[start:start + API_BATCH]
            for attempt in range(4):
                try:
                    return start, self._client.embeddings.create(
                        model=self._name, input=chunk).data
                except Exception as exc:  # noqa: BLE001: retried, then raised
                    # A 400 is the request itself, not load; retrying it only wastes attempts.
                    if attempt == 3 or "400" in str(exc):
                        raise
                    time.sleep(2 ** attempt)
            raise RuntimeError("unreachable")

        with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as pool:
            for start, data in pool.map(embed, starts):
                for offset, item in enumerate(data):
                    vector = np.asarray(item.embedding, dtype=np.float32)
                    out[start + offset] = vector / (np.linalg.norm(vector) or 1.0)
        return out

    def pairwise(self, left: list[str], right: list[str]) -> np.ndarray:
        """Similarity of left[i] to right[i]; NaN where either side is empty.

        Texts are deduplicated before encoding, which matters because the sentence being
        removed repeats across every rollout of its boundary.
        """
        if len(left) != len(right):
            raise ValueError(f"{len(left)} left texts against {len(right)} right texts")
        if not left:
            return np.empty(0, dtype=np.float32)

        usable = [bool(a.strip()) and bool(b.strip()) for a, b in zip(left, right)]
        wanted = {t for t, keep in zip(left, usable) if keep}
        wanted |= {t for t, keep in zip(right, usable) if keep}

        order = sorted(wanted)
        index = {text: i for i, text in enumerate(order)}
        vectors = self.encode(order)

        out = np.full(len(left), np.nan, dtype=np.float32)
        rows = [i for i, keep in enumerate(usable) if keep]
        if rows:
            a = vectors[[index[left[i]] for i in rows]]
            b = vectors[[index[right[i]] for i in rows]]
            out[rows] = np.einsum("ij,ij->i", a, b)
        return out
