from pathlib import Path

import pytest

from app.config import Settings
from app.storage.metadata_store import MetadataStore


def store_for(tmp_path: Path) -> MetadataStore:
    return MetadataStore(Settings(APP_DATABASE_URL=f"sqlite:///{tmp_path / 'rag.db'}"))


def test_workspace_owner_is_enforced(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    store.create_kb(workspace_id="workspace_acme", name="Policies", owner_id="user_1")

    with pytest.raises(PermissionError):
        store.assert_workspace_access("workspace_acme", "user_2")


def test_document_lifecycle(tmp_path: Path) -> None:
    store = store_for(tmp_path)
    kb = store.create_kb(workspace_id="workspace_acme", name="Policies", owner_id="user_1")
    document_id = store.add_document(
        workspace_id="workspace_acme",
        kb_id=kb["kb_id"],
        source="policy.pdf",
        chunks_indexed=3,
        file_path="data/uploads/policy.pdf",
    )

    documents = store.list_documents("workspace_acme", kb["kb_id"])
    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id

    removed = store.remove_document("workspace_acme", document_id)
    assert removed is not None
    assert store.get_document("workspace_acme", document_id) is None
