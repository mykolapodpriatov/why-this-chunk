"""Real local embedder backed by ``sentence-transformers`` (optional extra).

Install with ``pip install why-this-chunk[st]``. This module is import-guarded:
importing it without the extra raises a clear :class:`ImportError` rather than a
bare :class:`ModuleNotFoundError`.

Unlike :class:`~why_this_chunk.embedders.fake.FakeEmbedder`, this embedder may
download a model on first use and is therefore **not** used by the offline test
suite. It conforms to the same :class:`~why_this_chunk.embedders.Embedder`
protocol (L2-normalized rows).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["SentenceTransformerEmbedder"]


class SentenceTransformerEmbedder:
    """Local embedder using a ``sentence-transformers`` model.

    Args:
        model_name: A model id understood by ``SentenceTransformer`` (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``).
        device: Optional torch device string (``"cpu"``, ``"cuda"``, ...).
            ``None`` lets the library choose.

    Raises:
        ImportError: If the ``[st]`` extra (``sentence-transformers``) is not
            installed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "SentenceTransformerEmbedder requires the optional 'st' extra. "
                "Install it with: pip install 'why-this-chunk[st]'"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        """The embedding dimensionality reported by the model."""
        return self._dim

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode ``texts`` into an ``(n, dim)`` array of L2-normalized rows."""
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
