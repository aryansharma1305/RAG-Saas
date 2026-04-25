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
            raise ValueError("Only sqlite:/// database URLs are supported")
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
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
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
                    file_path TEXT,
                    chunks_indexed INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "file_path" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN file_path TEXT")

    def ensure_workspace(self, workspace_id: str, owner_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT workspace_id, owner_id FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if row:
                return dict(row)
            conn.execute(
                "INSERT INTO workspaces (workspace_id, owner_id) VALUES (?, ?)",
                (workspace_id, owner_id),
            )
        return {"workspace_id": workspace_id, "owner_id": owner_id}

    def assert_workspace_access(self, workspace_id: str, owner_id: str) -> None:
        row = self.ensure_workspace(workspace_id, owner_id)
        if row["owner_id"] != owner_id:
            raise PermissionError("Workspace access denied")

    def create_kb(self, workspace_id: str, name: str, owner_id: str) -> dict:
        self.assert_workspace_access(workspace_id, owner_id)
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

    def add_document(
        self,
        workspace_id: str,
        kb_id: str,
        source: str,
        chunks_indexed: int,
        file_path: str | None = None,
    ) -> str:
        document_id = f"doc_{uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, workspace_id, kb_id, source, file_path, chunks_indexed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, workspace_id, kb_id, source, file_path, chunks_indexed),
            )
        return document_id

    def update_document_chunks(self, document_id: str, chunks_indexed: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE documents SET chunks_indexed = ? WHERE document_id = ?",
                (chunks_indexed, document_id),
            )

    def list_kbs(self, workspace_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT kb_id, workspace_id, name FROM knowledge_bases WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_documents(self, workspace_id: str, kb_id: str | None = None) -> list[dict]:
        query = """
            SELECT document_id, workspace_id, kb_id, source, chunks_indexed, created_at
            FROM documents
            WHERE workspace_id = ?
        """
        params: list[str] = [workspace_id]
        if kb_id:
            query += " AND kb_id = ?"
            params.append(kb_id)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, workspace_id: str, document_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, workspace_id, kb_id, source, file_path, chunks_indexed, created_at
                FROM documents
                WHERE workspace_id = ? AND document_id = ?
                """,
                (workspace_id, document_id),
            ).fetchone()
        return dict(row) if row else None

    def remove_document(self, workspace_id: str, document_id: str) -> dict | None:
        document = self.get_document(workspace_id, document_id)
        if not document:
            return None
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM documents WHERE workspace_id = ? AND document_id = ?",
                (workspace_id, document_id),
            )
        return document
