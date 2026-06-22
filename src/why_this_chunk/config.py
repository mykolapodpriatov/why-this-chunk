"""Retrieval configuration and shared sweep constants.

:class:`RetrievalConfig` is the single typed contract carrying every axis the
counterfactual search and ``reindex`` operate on, so adapter authors and the
sweep target one concrete signature.

``CHUNK_SIZE_SWEEP`` is the *one* canonical set of chunk sizes used by **both**
the ``lost_to_chunking`` taxonomy check (:mod:`why_this_chunk.taxonomy`) and the
``chunk_size`` counterfactual axis (:mod:`why_this_chunk.counterfactual`); they
import it from here so the two can never diverge.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "ALPHA_SWEEP",
    "CHUNK_SIZE_SWEEP",
    "RERANK_COST",
    "RetrievalConfig",
    "load_project_config",
]

#: Canonical, ordered chunk sizes (characters) swept by the ``lost_to_chunking``
#: check and the ``chunk_size`` counterfactual axis. Shared so they never
#: diverge. Ordered ascending; cost on the axis is the absolute index distance.
CHUNK_SIZE_SWEEP: tuple[int, ...] = (128, 256, 384, 512, 768, 1024)

#: Ordered alpha values swept by the hybrid ``alpha`` counterfactual axis.
#: Cost is the number of steps moved from the current alpha's nearest index.
ALPHA_SWEEP: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Fixed integer cost charged for toggling the reranker on.
RERANK_COST: int = 4


class RetrievalConfig(BaseModel):
    """All retrieval axes the system can vary, as one validated contract.

    Attributes:
        top_k: Number of results to return (must be >= 1).
        chunk_size: Character window size used when (re)chunking source
            documents. Only meaningful for corpora with provenance.
        alpha: Hybrid mixing weight in ``[0, 1]`` applied to the dense modality;
            ``None`` for non-hybrid retrievers.
        rerank: Whether a cross-encoder reranker is applied after retrieval.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    top_k: int = Field(default=5, ge=1)
    chunk_size: int = Field(default=512, ge=1)
    alpha: float | None = Field(default=None)
    rerank: bool = Field(default=False)

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("alpha must be in the closed interval [0, 1]")
        return value

    def with_updates(self, **changes: object) -> RetrievalConfig:
        """Return a copy of this config with the given fields replaced."""
        return self.model_copy(update=changes)


def load_project_config(start: Path | None = None) -> RetrievalConfig:
    """Load a :class:`RetrievalConfig` from a project's ``pyproject.toml``.

    Reads the ``[tool.why-this-chunk]`` table, walking up from ``start`` (or the
    current working directory) to the first ``pyproject.toml`` found. Unknown
    keys are rejected. Returns defaults if no file or table is present.

    Args:
        start: Directory to begin searching from; defaults to the CWD.

    Returns:
        The parsed configuration, or :class:`RetrievalConfig` defaults.

    Raises:
        ValueError: If the ``[tool.why-this-chunk]`` table contains invalid
            values (propagated from pydantic with a clear message).
    """
    base = (start or Path.cwd()).resolve()
    for directory in (base, *base.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            with candidate.open("rb") as handle:
                data = tomllib.load(handle)
            table = data.get("tool", {}).get("why-this-chunk")
            if isinstance(table, dict):
                return RetrievalConfig.model_validate(table)
            return RetrievalConfig()
    return RetrievalConfig()
