"""Build a dedicated PostgreSQL + pgvector (1536-dim) index using OpenAI text-embedding-3-small."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    db_url = os.environ["DATABASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"].strip()
    model = "text-embedding-3-small"
    dims = 1536

    chunks = json.loads(Path("starter/data/converted/chunks.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from Day 6 export.")

    t0 = time.perf_counter()
    vectors: list[list[float]] = []
    total_tokens = 0
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": [c["text"] for c in batch]},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI embeddings HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        total_tokens += int((data.get("usage") or {}).get("total_tokens", 0) or 0)
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors.extend(item["embedding"] for item in items)

    cost_usd = total_tokens * 0.02 / 1_000_000.0

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks_openai (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    chapter INTEGER,
                    chapter_title TEXT,
                    book TEXT,
                    language TEXT,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({dims}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (strategy, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_openai_chapter ON chunks_openai (strategy, chapter);
                """
            )
            sql = """
                INSERT INTO chunks_openai (
                    strategy, chunk_id, document_id, source, page,
                    chapter, chapter_title, book, language,
                    text, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW();
            """
            rows = [
                (
                    "C_openai_1536",
                    c["id"],
                    c["source_id"],
                    c["source"],
                    c["page"],
                    c["metadata"].get("chapter"),
                    c["metadata"].get("chapter_title"),
                    c["metadata"].get("book"),
                    c["metadata"].get("language"),
                    c["text"],
                    json.dumps(c["metadata"], ensure_ascii=False),
                    "[" + ",".join(f"{x:.8f}" for x in vec) + "]",
                )
                for c, vec in zip(chunks, vectors)
            ]
            cur.executemany(sql, rows)
        conn.commit()

    elapsed = round(time.perf_counter() - t0, 3)
    meta = {
        "table": "chunks_openai",
        "strategy": "C_openai_1536",
        "model": model,
        "dimensions": dims,
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 6),
        "elapsed_sec": elapsed,
    }
    Path("starter/results/openai_index_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("OpenAI index built:", json.dumps(meta))


if __name__ == "__main__":
    main()