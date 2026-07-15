"""
Candidate retrieval stage.

This is the mechanism that lets the pipeline respect the assignment's
CRITICAL CONSTRAINT: no single LLM call ever sees both schemas in full.

Source fields and destination fields are each turned into a normalized
bag-of-words string, independently, then embedded into a shared vector
space. Cosine similarity gives a table-level match (source table -> best
destination collection) and, within that collection, a top-k shortlist of
candidate destination fields for every individual source field.

Two interchangeable backends compute that vector space, selected
automatically:

  * **Dense embeddings** (app/providers/embeddings.py) -- OpenAI or Azure AI
    Foundry embedding models, when configured. Understands semantic
    relationships (e.g. "hire date" ~ "start date") without needing the
    hand-maintained synonym dictionary in normalize.py to bridge the gap.
    Source texts and destination texts are embedded in two *separate* API
    calls -- never combined into one request -- for the same reason the
    reasoning stage never sees both schemas in one prompt (see the module
    docstring in app/providers/embeddings.py for the detailed rationale).

  * **TF-IDF** (scikit-learn, local, no network) -- the fallback when no
    embedding provider is configured. Fully offline and deterministic, at
    the cost of needing normalize.py's synonym dictionary to catch
    non-lexical matches.

Both backends expose the exact same `top_candidates` / `best_tables`
interface, so nothing downstream (app/graph/nodes.py) needs to know or care
which one actually ran.

The reasoning stage (app/graph/reasoning.py, via app/providers) then only
ever receives ONE source field plus its top-k shortlist -- never the full
schema on either side, regardless of which retrieval backend produced that
shortlist.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.schemas import (
    SourceField, DestField, SOURCE_FIELDS, DEST_FIELDS,
    SOURCE_TABLES, DEST_COLLECTIONS,
    source_fields_for_table, dest_fields_for_collection,
)
from app.core.normalize import normalize_field_text
from app.providers.embeddings import EmbeddingProvider

TOP_K_FIELDS = 3
TOP_K_TABLES = 2


@dataclass
class Candidate:
    dest_field: DestField
    score: float


class SchemaRetriever:
    """Builds the shared vector space once and serves similarity queries."""

    def __init__(self, extra_synonyms: dict[str, list[str]] | None = None,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        self.extra_synonyms = extra_synonyms

        self.source_texts = [self._source_text(f) for f in SOURCE_FIELDS]
        self.dest_texts = [self._dest_text(f) for f in DEST_FIELDS]

        self.backend = "tfidf"
        self.source_matrix = None
        self.dest_matrix = None

        if embedding_provider is not None and embedding_provider.available():
            source_vecs = embedding_provider.embed(self.source_texts)
            dest_vecs = embedding_provider.embed(self.dest_texts)
            if source_vecs is not None and dest_vecs is not None:
                self.source_matrix = np.asarray(source_vecs)
                self.dest_matrix = np.asarray(dest_vecs)
                self.backend = f"embedding:{embedding_provider.name}"

        if self.source_matrix is None:
            # No embedding provider configured, or it failed mid-call -- fall
            # back to TF-IDF rather than raise, matching every other
            # graceful-degradation point in this codebase.
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            self.vectorizer.fit(self.source_texts + self.dest_texts)
            self.source_matrix = self.vectorizer.transform(self.source_texts)
            self.dest_matrix = self.vectorizer.transform(self.dest_texts)

        self._table_centroids = self._build_centroids(
            SOURCE_TABLES, source_fields_for_table, SOURCE_FIELDS, self.source_matrix
        )
        self._collection_centroids = self._build_centroids(
            DEST_COLLECTIONS, dest_fields_for_collection, DEST_FIELDS, self.dest_matrix
        )

    def _source_text(self, f: SourceField) -> str:
        return normalize_field_text(
            name_or_path=f.name, type_str=f.sql_type,
            comment=f.comment, constraints=f.constraints,
            extra_synonyms=self.extra_synonyms,
        )

    def _dest_text(self, f: DestField) -> str:
        return normalize_field_text(
            name_or_path=f.path, type_str=f.bson_type, comment=f.comment,
            extra_synonyms=self.extra_synonyms,
        )

    @staticmethod
    def _build_centroids(names, fields_for, all_fields, matrix):
        centroids = {}
        for name in names:
            indices = [i for i, f in enumerate(all_fields)
                       if (f.table if hasattr(f, "table") else f.collection) == name]
            sub = matrix[indices]
            centroid = np.asarray(sub.mean(axis=0))
            if centroid.ndim == 1:
                centroid = centroid.reshape(1, -1)
            centroids[name] = centroid
        return centroids

    def best_tables(self, source_table: str, top_k: int = TOP_K_TABLES):
        src_centroid = self._table_centroids[source_table]
        rows = []
        for coll in DEST_COLLECTIONS:
            sim = cosine_similarity(src_centroid, self._collection_centroids[coll])[0][0]
            rows.append((coll, float(sim)))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:top_k]

    def top_candidates(self, source_field: SourceField, collection: str,
                        top_k: int = TOP_K_FIELDS) -> list[Candidate]:
        idx = SOURCE_FIELDS.index(source_field)
        src_vec = self.source_matrix[idx]
        if src_vec.ndim == 1:
            src_vec = src_vec.reshape(1, -1)

        dest_indices = [i for i, f in enumerate(DEST_FIELDS) if f.collection == collection]
        if not dest_indices:
            return []

        sub_matrix = self.dest_matrix[dest_indices]
        sims = cosine_similarity(src_vec, sub_matrix)[0]

        order = np.argsort(sims)[::-1][:top_k]
        return [Candidate(DEST_FIELDS[dest_indices[i]], float(sims[i])) for i in order]
