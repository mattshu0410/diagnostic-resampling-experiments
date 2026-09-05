"""Content-addressed storage for sentence embeddings.

The cache identity includes every setting that changes the embedding space.  Callers provide
the embeddings; this module only validates, indexes, reads and writes them.  Provider-specific
API and model-loading code therefore cannot leak into cache handling.

One file holds one identity, which includes the key scheme naming the function used to
address text within it.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 1

SHA256_IDENTITY = "sha256-identity"  # key = sha256(identity fields + normalised text)
SHA1_TEXT = "sha1-text"  # key = sha1(normalised text); what the first OpenAI cache wrote
KEY_SCHEMES = (SHA256_IDENTITY, SHA1_TEXT)


def normalise_text(text: str) -> str:
    """Collapse whitespace so formatting-only differences share an embedding."""
    return " ".join(text.split())


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Settings that uniquely identify one embedding space."""

    provider: str
    model: str
    dimensions: int
    revision: str | None = None
    key_scheme: str = SHA256_IDENTITY

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("embedding provider must not be empty")
        if not self.model.strip():
            raise ValueError("embedding model must not be empty")
        if self.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.key_scheme not in KEY_SCHEMES:
            raise ValueError(
                f"unknown embedding key scheme {self.key_scheme!r}; expected one of {KEY_SCHEMES}"
            )

    def as_dict(self) -> dict:
        """JSON-serialisable identity in stable field order."""
        return {
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "key_scheme": self.key_scheme,
        }

    def key(self, text: str) -> str:
        """Content key for `text` under this identity's key scheme."""
        normalised = normalise_text(text)
        if self.key_scheme == SHA1_TEXT:
            return hashlib.sha1(normalised.encode("utf-8")).hexdigest()

        payload = {"identity": self.as_dict(), "text": normalised}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class EmbeddingCache:
    """Vectors belonging to exactly one :class:`EmbeddingIdentity`."""

    identity: EmbeddingIdentity
    vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vectors = {
            str(key): self._validate_vector(vector, key=str(key))
            for key, vector in self.vectors.items()
        }

    @classmethod
    def load(cls, path: Path, expected_identity: EmbeddingIdentity) -> EmbeddingCache:
        """Load `path`, or return an empty cache when it does not exist.

        The stored identity must exactly match `expected_identity`.  Legacy cache formats are
        intentionally rejected here; their one-time interpretation belongs in the caller that
        knows which historical model produced them.
        """
        path = Path(path)
        if not path.exists():
            return cls(expected_identity)

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: embedding cache payload is not a mapping")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported embedding cache schema "
                f"{payload.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )

        raw_identity = payload.get("identity")
        if not isinstance(raw_identity, dict):
            raise ValueError(f"{path}: embedding cache has no valid identity")
        try:
            stored_identity = EmbeddingIdentity(**raw_identity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid embedding identity: {exc}") from exc
        if stored_identity != expected_identity:
            raise ValueError(
                f"{path}: embedding identity mismatch: stored {stored_identity.as_dict()}, "
                f"requested {expected_identity.as_dict()}"
            )

        vectors = payload.get("vectors")
        if not isinstance(vectors, dict):
            raise ValueError(f"{path}: embedding cache vectors are not a mapping")
        return cls(stored_identity, vectors)

    def get(self, texts: Iterable[str]) -> np.ndarray:
        """Return vectors for `texts` in input order; raise when any are absent."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.identity.dimensions), dtype=np.float32)

        keys = [self.identity.key(text) for text in texts]
        missing = [key for key in dict.fromkeys(keys) if key not in self.vectors]
        if missing:
            raise KeyError(f"embedding cache is missing {len(missing)} distinct texts")
        return np.stack([self.vectors[key] for key in keys]).astype(np.float32, copy=False)

    def missing(self, texts: Iterable[str]) -> list[str]:
        """Distinct uncached texts in first-occurrence order."""
        unseen: set[str] = set()
        missing: list[str] = []
        for text in texts:
            key = self.identity.key(text)
            if key not in self.vectors and key not in unseen:
                unseen.add(key)
                missing.append(text)
        return missing

    def update(self, texts: Iterable[str], vectors: np.ndarray) -> None:
        """Add one vector per text, validating count, shape and finite values."""
        texts = list(texts)
        array = np.asarray(vectors)
        if array.ndim != 2:
            raise ValueError(f"embedding batch must be two-dimensional, got shape {array.shape}")
        if len(texts) != len(array):
            raise ValueError(f"received {len(texts)} texts but {len(array)} vectors")
        if array.shape[1] != self.identity.dimensions:
            raise ValueError(
                f"embedding batch has {array.shape[1]} dimensions; "
                f"expected {self.identity.dimensions}"
            )

        validated = [self._validate_vector(vector) for vector in array]
        for text, vector in zip(texts, validated):
            self.vectors[self.identity.key(text)] = vector

    def save(self, path: Path) -> Path:
        """Atomically write the cache to `path`, following it if it is a symlink."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path = path.resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity.as_dict(),
            "vectors": self.vectors,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return path

    def _validate_vector(self, vector: np.ndarray, *, key: str | None = None) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        where = f" for key {key}" if key is not None else ""
        if array.shape != (self.identity.dimensions,):
            raise ValueError(
                f"embedding vector{where} has shape {array.shape}; "
                f"expected {(self.identity.dimensions,)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"embedding vector{where} contains non-finite values")
        return array
