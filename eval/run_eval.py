"""Retrieval evaluation against eval/gold_set.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import ensure_src_on_path

ensure_src_on_path()

import config
from src.retrieval import retrieve


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & set(gold))


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def gold_ranks(retrieved: list[str], gold: list[str]) -> dict[str, int | None]:
    rank_by_id: dict[str, int | None] = {}
    for gid in gold:
        try:
            rank_by_id[gid] = retrieved.index(gid) + 1
        except ValueError:
            rank_by_id[gid] = None
    return rank_by_id

from src.utils import read_jsonl, resolve_path, write_jsonl
from tqdm import tqdm


def validate_gold_chunk_ids(
    gold_records: list[dict], corpus_chunk_ids: set[str]
) -> list[str]:
    """Return gold chunk IDs that are missing from the processed corpus."""
    missing: list[str] = []
    for record in gold_records:
        for chunk_id in record.get("must_cite_chunk_ids", []):
            if chunk_id not in corpus_chunk_ids and chunk_id not in missing:
                missing.append(chunk_id)
    return missing


def evaluate_example(record: dict, k: int) -> dict:
    question = record["question"]
    gold_ids = record.get("must_cite_chunk_ids", [])
    chunks = retrieve(question, k=k)
    retrieved_ids = [c["chunk_id"] for c in chunks]

    return {
        "question": question,
        "category": record.get("category", "unknown"),
        "must_cite_chunk_ids": gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "hit": hit_at_k(retrieved_ids, gold_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, gold_ids),
        "recall": recall_at_k(retrieved_ids, gold_ids, k),
        "gold_ranks": gold_ranks(retrieved_ids, gold_ids),
    }


def aggregate_results(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"count": 0, "hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)

    category_stats = {}
    for category, rows in sorted(by_category.items()):
        c = len(rows)
        category_stats[category] = {
            "count": c,
            "hit_at_k": sum(r["hit"] for r in rows) / c,
            "mrr": sum(r["mrr"] for r in rows) / c,
            "recall_at_k": sum(r["recall"] for r in rows) / c,
        }

    return {
        "count": n,
        "hit_at_k": sum(r["hit"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "recall_at_k": sum(r["recall"] for r in results) / n,
        "by_category": category_stats,
    }


DEFAULT_RESULTS_PATH = "eval/results/retrieval_results.json"


def build_results_payload(
    *,
    k: int,
    gold_path: Path,
    limit: int | None,
    results: list[dict],
    summary: dict,
) -> dict:
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "gold_path": str(gold_path),
        "limit": limit,
        "config": {
            "chunking_strategy": config.CHUNKING_STRATEGY,
            "chunk_size_words": config.CHUNK_SIZE_WORDS,
            "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
            "embedding_model": config.EMBEDDING_MODEL,
            "top_k": config.TOP_K,
        },
        "summary": summary,
        "failures_count": sum(1 for r in results if not r["hit"]),
        "examples": results,
    }


def save_results(path: str | Path, payload: dict) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def print_summary(summary: dict, k: int, failures: list[dict]) -> None:
    print(f"\nRetrieval evaluation (K={k}, n={summary['count']})")
    print("-" * 50)
    print(f"Hit@{k}:      {summary['hit_at_k']:.1%} ({summary['hit_at_k'] * summary['count']:.0f}/{summary['count']})")
    print(f"MRR:          {summary['mrr']:.4f}")
    print(f"Recall@{k}:   {summary['recall_at_k']:.1%}")

    print("\nBy category:")
    for category, stats in summary["by_category"].items():
        print(
            f"  {category:12} n={stats['count']:3}  "
            f"hit={stats['hit_at_k']:.1%}  mrr={stats['mrr']:.4f}  "
            f"recall={stats['recall_at_k']:.1%}"
        )

    if failures:
        print(f"\nMisses (showing up to 10 of {len(failures)}):")
        for row in failures[:10]:
            print(f"  Q: {row['question'][:80]}{'...' if len(row['question']) > 80 else ''}")
            print(f"     expected: {row['must_cite_chunk_ids']}")
            print(f"     retrieved: {row['retrieved_chunk_ids']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval against eval/gold_set.jsonl"
    )
    parser.add_argument(
        "--gold",
        default="eval/gold_set.jsonl",
        help="Path to gold evaluation set (JSONL)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (default: config.TOP_K={config.TOP_K})",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to write full results JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to --results path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write per-example results only (JSONL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples (smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    k = args.k if args.k is not None else config.TOP_K

    gold_path = resolve_path(args.gold)
    gold_records = read_jsonl(gold_path)
    if not gold_records:
        print(f"No records in {gold_path}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        gold_records = gold_records[: args.limit]

    chunks_path = resolve_path(config.PROCESSED_CHUNKS_PATH)
    corpus_chunks = read_jsonl(chunks_path)
    corpus_chunk_ids = {c["chunk_id"] for c in corpus_chunks}
    missing_ids = validate_gold_chunk_ids(gold_records, corpus_chunk_ids)
    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} gold chunk ID(s) not in {chunks_path}:",
            file=sys.stderr,
        )
        for chunk_id in missing_ids:
            print(f"  - {chunk_id}", file=sys.stderr)

    results: list[dict] = []
    for record in tqdm(gold_records, desc="Evaluating retrieval"):
        results.append(evaluate_example(record, k=k))

    summary = aggregate_results(results)
    failures = [r for r in results if not r["hit"]]
    print_summary(summary, k=k, failures=failures)

    if not args.no_save:
        payload = build_results_payload(
            k=k,
            gold_path=gold_path,
            limit=args.limit,
            results=results,
            summary=summary,
        )
        results_path = save_results(args.results, payload)
        print(f"\nWrote results to {results_path}")

    if args.output:
        output_path = resolve_path(args.output)
        write_jsonl(output_path, results)
        print(f"Wrote per-example results to {output_path}")


if __name__ == "__main__":
    main()
