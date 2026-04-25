import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.pinecone_store import PineconeStore


def main() -> None:
    settings = get_settings()
    store = PineconeStore(settings)
    store.ensure_index()
    print(f"ready: {settings.pinecone_index_name}")


if __name__ == "__main__":
    main()
