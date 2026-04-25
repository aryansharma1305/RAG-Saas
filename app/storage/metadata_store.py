import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.config import Settings


class MetadataStore:
    def __init__(self, settings: Settings):
        self.db_path = self._parse_sqlite_path(settings.app_database_url)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @staticmethod
    def _parse_sqlite_path(database_url: str) -> Path:
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// database URLs are supported in this MVP")
        return Path(database_url.removeprefix(prefix))

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    kb_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunks_indexed INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_kb(self, workspace_id: str, name: str) -> dict:
        kb_id = f"kb_{uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (kb_id, workspace_id, name) VALUES (?, ?, ?)",
                (kb_id, workspace_id, name),
            )
        return {"kb_id": kb_id, "workspace_id": workspace_id, "name": name}

    def get_kb(self, workspace_id: str, kb_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT kb_id, workspace_id, name FROM knowledge_bases WHERE workspace_id = ? AND kb_id = ?",
                (workspace_id, kb_id),
            ).fetchone()
        return dict(row) if row else None

    def add_document(self, workspace_id: str, kb_id: str, source: str, chunks_indexed: int) -> str:
        document_id = f"doc_{uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, workspace_id, kb_id, source, chunks_indexed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, workspace_id, kb_id, source, chunks_indexed),
            )
        return document_id

    def list_kbs(self, workspace_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT kb_id, workspace_id, name FROM knowledge_bases WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]
