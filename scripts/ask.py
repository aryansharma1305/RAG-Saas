import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.dependencies import get_rag_chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--kb-id", action="append", required=True)
    parser.add_argument("--model-key", default=None, choices=["qwen", "gemma", "glm"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    response = get_rag_chain().answer(
        workspace_id=args.workspace_id,
        kb_ids=args.kb_id,
        question=args.question,
        model_key=args.model_key or settings.default_model_key,
        top_k=args.top_k,
        min_score=0.0,
    )
    print(response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source['source']} score={source['score']:.4f}")


if __name__ == "__main__":
    main()
