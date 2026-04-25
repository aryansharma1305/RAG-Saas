import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.dependencies import get_rag_chain


def load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question file must contain a non-empty JSON list")
    for item in questions:
        if not item.get("question"):
            raise ValueError("Each question item must include a question")
        item.setdefault("expected_keywords", [])
    return questions


def latest_kb(workspace_id: str) -> str:
    settings = get_settings()
    db_path = Path(settings.app_database_url.removeprefix("sqlite:///"))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT kb_id
            FROM knowledge_bases
            WHERE workspace_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (workspace_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"No knowledge base found for workspace_id={workspace_id!r}")
    return row["kb_id"]


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return hits / len(expected_keywords)


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    settings = get_settings()
    model_keys = args.model_key or list(settings.local_models)
    unknown = sorted(set(model_keys) - set(settings.local_models))
    if unknown:
        raise ValueError(f"Unknown model keys: {', '.join(unknown)}")

    kb_ids = args.kb_id or [latest_kb(args.workspace_id)]
    questions = load_questions(args.questions)
    rag = get_rag_chain()
    results = []

    for model_key in model_keys:
        model_runs = []
        if not args.no_warmup:
            rag.answer(
                workspace_id=args.workspace_id,
                kb_ids=kb_ids,
                question="Reply with OK.",
                model_key=model_key,
                top_k=args.top_k,
                min_score=args.min_score,
            )

        for item in questions:
            started = time.perf_counter()
            response = rag.answer(
                workspace_id=args.workspace_id,
                kb_ids=kb_ids,
                question=item["question"],
                model_key=model_key,
                top_k=args.top_k,
                min_score=args.min_score,
            )
            elapsed = time.perf_counter() - started
            model_runs.append(
                {
                    "question": item["question"],
                    "answer": response["answer"],
                    "latency_seconds": round(elapsed, 2),
                    "sources": len(response["sources"]),
                    "keyword_score": round(keyword_score(response["answer"], item["expected_keywords"]), 2),
                }
            )

        latencies = [run["latency_seconds"] for run in model_runs]
        scores = [run["keyword_score"] for run in model_runs]
        results.append(
            {
                "model_key": model_key,
                "model_name": settings.local_models[model_key],
                "avg_latency_seconds": round(statistics.mean(latencies), 2),
                "min_latency_seconds": round(min(latencies), 2),
                "max_latency_seconds": round(max(latencies), 2),
                "avg_keyword_score": round(statistics.mean(scores), 2),
                "runs": model_runs,
            }
        )
    return results


def print_summary(results: list[dict]) -> None:
    print("model_key | model_name | avg_latency_seconds | avg_keyword_score")
    print("--- | --- | ---: | ---:")
    for result in results:
        print(
            f"{result['model_key']} | {result['model_name']} | "
            f"{result['avg_latency_seconds']} | {result['avg_keyword_score']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark configured RAG models against an evaluation set.")
    parser.add_argument("--workspace-id", default="workspace_acme")
    parser.add_argument("--kb-id", action="append", default=None)
    parser.add_argument("--questions", type=Path, default=Path("data/sample_eval_questions.json"))
    parser.add_argument("--model-key", action="append", choices=["qwen", "gemma", "glm"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = run_benchmark(args)
    print_summary(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
