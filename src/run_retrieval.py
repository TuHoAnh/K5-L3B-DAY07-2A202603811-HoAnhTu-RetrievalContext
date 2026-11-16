"""Run and compare retrieval configurations on dev or test (answerable questions only)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

STARTER_DIR = Path(__file__).resolve().parent.parent / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from baseline import hit, pack_context  # noqa: E402
from src.retrievers import (  # noqa: E402
    OpenAIQueryEmbedder,
    DensePgVectorRetriever,
    HybridRrfRetriever,
    LexicalTfidfRetriever,
)


def evaluate_config(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int,
    budget: int,
    use_filter: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        active_filter = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=active_filter, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        top3 = marked[:3]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "strategy": name,
                "top_k": top_k,
                "budget_chars": budget,
                "filter": active_filter,
                "results_top3": top3,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": bool(top3 and top3[0]["hit"]),
                "hit3": any(c["hit"] for c in top3),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
            }
        )

    n = len(traces)
    summary = {
        "strategy": name,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "use_filter": use_filter,
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3),
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()

    load_dotenv()
    db_url = os.environ["DATABASE_URL"]
    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or "nvidia/nemotron-3.5-embed-1b:free"

    chunks = json.loads(Path("starter/data/converted/chunks.json").read_text(encoding="utf-8"))
    all_questions = json.loads(Path("starter/data/converted/questions.json").read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q["split"] == args.split]

    embedder = OpenAIQueryEmbedder()
    # Pre-warm cache in 1 single batch call
    embedder.embed_queries([q["question"] for q in questions])

    configs = {
        "tfidf_baseline": (LexicalTfidfRetriever(chunks, use_filter=False), 3, 800, False),
        "dense_pgvector": (DensePgVectorRetriever(db_url, embedder, use_filter=False), 3, 800, False),
        "hybrid_rrf": (
            HybridRrfRetriever(
                DensePgVectorRetriever(db_url, embedder, use_filter=False),
                LexicalTfidfRetriever(chunks, use_filter=False),
            ),
            3,
            800,
            False,
        ),
        "hybrid_rrf_filter": (
            HybridRrfRetriever(
                DensePgVectorRetriever(db_url, embedder, use_filter=True),
                LexicalTfidfRetriever(chunks, use_filter=True),
            ),
            3,
            800,
            True,
        ),
        "hybrid_rrf_filter_k5_b1400": (
            HybridRrfRetriever(
                DensePgVectorRetriever(db_url, embedder, use_filter=True),
                LexicalTfidfRetriever(chunks, use_filter=True),
            ),
            5,
            1400,
            True,
        ),
        "dense_filter_k5_b1600": (
            DensePgVectorRetriever(db_url, embedder, use_filter=True),
            5,
            1600,
            True,
        ),
    }

    selected_names = args.configs or list(configs.keys())
    out_dir = Path(f"starter/results/retrieval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_traces = {}
    for name in selected_names:
        retriever, top_k, budget, use_filter = configs[name]
        summ, tr = evaluate_config(
            name, retriever, questions, top_k=top_k, budget=budget, use_filter=use_filter
        )
        summaries.append(summ)
        all_traces[name] = tr
        print(
            f"{name:28s} | Hit@1={summ['hit1']}/{summ['n']} "
            f"| Hit@3={summ['hit3']}/{summ['n']} "
            f"| CtxHit={summ['context_hit']}/{summ['n']} "
            f"| AvgCtx={summ['avg_context_chars']}c | Med={summ['median_ms']}ms"
        )

    meta = {
        "split": args.split,
        "embedding_api_requests": embedder.api_requests,
        "embedding_tokens": embedder.total_tokens,
        "embedding_cost_usd": embedder.total_cost_usd,
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "traces.json").write_text(json.dumps(all_traces, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()