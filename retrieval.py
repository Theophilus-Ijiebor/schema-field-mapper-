"""
Candidate retrieval stage.

This is the mechanism that lets the pipeline respect the assignment's
CRITICAL CONSTRAINT: no single LLM call ever sees both schemas in full.

Instead:
  1. Source fields and destination fields are each turned into a normalized
     bag-of-words string (normalize.py) -- independently, per side.
  2. A single TF-IDF vectorizer is *fit* across both vocabularies so the two
     sides land in the same vector space (this is a classic IR technique,
     not an LLM call -- no semantic "understanding" happens here, just
     token-overlap statistics).
  3. Cosine similarity gives a table-level match (source table -> best
     destination collection) and, within that collection, a top-k shortlist
     of candidate destination fields for every individual source field.

The LLM reasoning stage (llm_reasoner.py) then only ever receives ONE source
field plus its top-k shortlist (k=3) -- never the full schema on either side.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import (
    SourceField, DestField, SOURCE_FIELDS, DEST_FIELDS,
    SOURCE_TABLES, DEST_COLLECTIONS,
    source_fields_for_table, dest_fields_for_collection,
)
from normalize import normalize_field_text

TOP_K_FIELDS = 3
TOP_K_TABLES = 2


@dataclass
class Candidate:
    dest_field: DestField
    score: float


def _source_text(f: SourceField) -> str:
    return normalize_field_text(
        name_or_path=f.name, type_str=f.sql_type,
        comment=f.comment, constraints=f.constraints,
    )


def _dest_text(f: DestField) -> str:
    return normalize_field_text(
        name_or_path=f.path, type_str=f.bson_type, comment=f.comment,
    )


class SchemaRetriever:
    """Builds the shared vector space once and serves similarity queries."""

    def __init__(self):
        self.source_texts = [_source_text(f) for f in SOURCE_FIELDS]
        self.dest_texts = [_dest_text(f) for f in DEST_FIELDS]

        # Fit on the union of vocabularies -- this only shares *token
        # statistics*, not schema content, and involves no LLM call.
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.vectorizer.fit(self.source_texts + self.dest_texts)

        self.source_matrix = self.vectorizer.transform(self.source_texts)
        self.dest_matrix = self.vectorizer.transform(self.dest_texts)

        self._table_centroids = self._build_centroids(
            SOURCE_TABLES, source_fields_for_table, self.vectorizer, _source_text
        )
        self._collection_centroids = self._build_centroids(
            DEST_COLLECTIONS, dest_fields_for_collection, self.vectorizer, _dest_text
        )

    @staticmethod
    def _build_centroids(names, fields_for, vectorizer, text_fn):
        centroids = {}
        for name in names:
            texts = [text_fn(f) for f in fields_for(name)]
            matrix = vectorizer.transform(texts)
            centroids[name] = np.asarray(matrix.mean(axis=0))
        return centroids

    def best_tables(self, source_table: str, top_k: int = TOP_K_TABLES):
        """Rank destination collections against one source table's centroid."""
        src_centroid = self._table_centroids[source_table]
        rows = []
        for coll in DEST_COLLECTIONS:
            sim = cosine_similarity(src_centroid, self._collection_centroids[coll])[0][0]
            rows.append((coll, float(sim)))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:top_k]

    def top_candidates(self, source_field: SourceField, collection: str,
                        top_k: int = TOP_K_FIELDS) -> list[Candidate]:
        """Top-k destination field candidates within one collection."""
        idx = SOURCE_FIELDS.index(source_field)
        src_vec = self.source_matrix[idx]

        dest_indices = [i for i, f in enumerate(DEST_FIELDS) if f.collection == collection]
        if not dest_indices:
            return []

        sub_matrix = self.dest_matrix[dest_indices]
        sims = cosine_similarity(src_vec, sub_matrix)[0]

        order = np.argsort(sims)[::-1][:top_k]
        return [Candidate(DEST_FIELDS[dest_indices[i]], float(sims[i])) for i in order]
