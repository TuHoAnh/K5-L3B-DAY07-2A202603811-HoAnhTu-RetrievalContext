"""Real retrievers for Day 7: Dense (pgvector), Lexical (TF-IDF), and Hybrid (RRF)."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
import requests

STARTER_DIR = Path(__file__).resolve().parent.parent / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from baseline import TfidfIndex, tokenize  # noqa: E402


class OpenAIQueryEmbedder:
    """Embed queries using OpenAI text-embedding-3-small with local JSON cache."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        cache_path: Path = Path("starter/results/openai_query_vec_cache.json"),
    ) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.api_requests = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def embed_queries(self, queries: list[str]) -> dict[str, list[float]]:
        missing = [q for q in dict.fromkeys(queries) if q not in self.cache]
        if missing:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            for i in range(0, len(missing), 64):
                batch = missing[i : i + 64]
                last_err: Exception | None = None
                for attempt in range(4):
                    try:
                        resp = requests.post(
                            "https://api.openai.com/v1/embeddings",
                            headers=headers,
                            json={"model": self.model_name, "input": batch},
                            timeout=60,
                        )
                        if resp.status_code == 429:
                            last_err = RuntimeError(f"429: {resp.text[:200]}")
                            time.sleep(3 * (attempt + 1))
                            continue
                        if resp.status_code >= 400:
                            raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
                        data = resp.json()
                        self.api_requests += 1
                        toks = int((data.get("usage") or {}).get("total_tokens", 0) or 0)
                        self.total_tokens += toks
                        self.total_cost_usd += toks * 0.02 / 1_000_000.0
                        items = sorted(data["data"], key=lambda x: x["index"])
                        for q_text, item in zip(batch, items):
                            vec = item["embedding"]
                            if len(vec) != self.dimensions:
                                raise RuntimeError(f"Dim mismatch: expected {self.dimensions}, got {len(vec)}")
                            self.cache[q_text] = vec
                        self._save_cache()
                        break
                    except Exception as exc:
                        last_err = exc
                        if attempt < 3:
                            time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"Query embedding failed: {last_err}")
        return {q: self.cache[q] for q in queries}

    def embed_query(self, query: str) -> list[float]:
        return self.embed_queries([query])[query]


class DensePgVectorRetriever:
    """Query PostgreSQL + pgvector index (exact cosine similarity)."""

    ALLOWED_FILTERS = {"chapter", "book", "document_id", "language", "page"}
    ALLOWED_TABLES = {"chunks", "chunks_openai"}

    def __init__(
        self,
        database_url: str,
        embedder: Any,
        table_name: str = "chunks_openai",
        pg_strategy: str = "C_openai_1536",
        use_filter: bool = False,
    ) -> None:
        if table_name not in self.ALLOWED_TABLES:
            raise ValueError(f"Unsupported table: {table_name}")
        self.database_url = database_url
        self.embedder = embedder
        self.table_name = table_name
        self.pg_strategy = pg_strategy
        self.use_filter = use_filter

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self.embedder.embed_query(query)
        q_lit = "[" + ",".join(f"{x:.8f}" for x in q_vec) + "]"

        where = ["strategy = %s"]
        params: list[Any] = [q_lit, self.pg_strategy]
        if self.use_filter and filters:
            for k, v in filters.items():
                if k not in self.ALLOWED_FILTERS:
                    raise ValueError(f"Unsupported filter: {k}")
                where.append(f"{k} = %s")
                params.append(v)

        params.extend([q_lit, top_k])
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        out: list[dict[str, Any]] = []
        for cid, doc_id, source, page, text, meta, score in rows:
            out.append(
                {
                    "id": cid,
                    "source_id": doc_id,
                    "source": source,
                    "page": int(page),
                    "text": text,
                    "metadata": meta if isinstance(meta, dict) else json.loads(meta),
                    "score": float(score),
                    "strategy": "dense_pgvector",
                }
            )
        return out


class LexicalTfidfRetriever:
    """TF-IDF retriever with optional metadata filtering."""

    def __init__(self, chunks: list[dict[str, Any]], use_filter: bool = False) -> None:
        self.index = TfidfIndex(chunks)
        self.use_filter = use_filter

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self.index._vector(Counter(tokenize(query)))
        results: list[dict[str, Any]] = []
        for chunk, vec in zip(self.index.chunks, self.index.vectors):
            if self.use_filter and filters:
                meta = chunk.get("metadata") or {}
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            score = sum(val * vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, "score": float(score), "strategy": "tfidf"})
        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class HybridRrfRetriever:
    """Reciprocal Rank Fusion (RRF) combining Dense pgvector and Lexical TF-IDF."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        lexical: LexicalTfidfRetriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        d_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        l_hits = self.lexical.search(query, filters=filters, top_k=self.candidate_k)

        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(d_hits, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, item in enumerate(l_hits, start=1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        out: list[dict[str, Any]] = []
        for cid, rrf_score in ranked:
            item = dict(chunk_map[cid])
            item["score"] = round(rrf_score, 6)
            item["strategy"] = "hybrid_rrf"
            out.append(item)
        return out