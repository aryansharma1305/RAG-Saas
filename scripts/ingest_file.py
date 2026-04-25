import argparse
import asyncio
import sys
from pathlib import Path

from fastapi import UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.dependencies import get_ingestion_service


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--kb-id", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    with path.open("rb") as handle:
        upload = UploadFile(file=handle, filename=path.name)
        result = await get_ingestion_service().ingest_upload(
            file=upload,
            workspace_id=args.workspace_id,
            kb_id=args.kb_id,
        )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
