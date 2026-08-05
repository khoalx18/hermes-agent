"""Deleting a compression tip must remove the whole logical conversation.

Regression for the desktop bug where clicking delete on a session that is the
live tip of a compression chain appeared to succeed, then the sidebar
re-listings (e.g. after switching profiles) resurrected the conversation via
its still-present root — mirroring the whole-chain flip that
``set_session_archived`` already performs.
"""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def _delegate_child(db: SessionDB, parent: str, child: str):
    db.create_session(child, source="cli", parent_session_id=parent)
    db._conn.execute(
        "UPDATE sessions SET model_config = json_object('_delegate_from', ?) WHERE id = ?",
        (parent, child),
    )
    db._conn.commit()


def _branch_child(db: SessionDB, parent: str, child: str):
    db.create_session(child, source="cli", parent_session_id=parent)
    db._conn.execute(
        "UPDATE sessions SET model_config = json_object('_branched_from', ?) WHERE id = ?",
        (parent, child),
    )
    db._conn.commit()


def test_delete_compression_tip_cascades_whole_chain(db):
    _compression_pair(db)

    assert db.delete_session("tip") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_root_cascades_descendants(db):
    _compression_pair(db)

    assert db.delete_session("root") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None


def test_delete_single_keeps_compression_children(db):
    """CLI --lineage single semantics: delete one row, keep the rest."""
    _compression_pair(db)

    assert db.delete_session("tip", include_compression_lineage=False) is True

    assert db.get_session("root") is not None
    assert db.get_session("tip") is None
    # Root now projects forward to itself (no live continuation).
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["root"]


def test_delete_mid_chain_cascades_ancestors_and_descendants(db):
    _compression_pair(db)
    db.create_session("tip2", source="cli", parent_session_id="tip")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip2'",
        (time.time() - 10,),
    )
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'compression' WHERE id = 'tip'"
    )
    db._conn.commit()

    assert db.delete_session("tip") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert db.get_session("tip2") is None


def test_delete_cascades_delegate_children_of_lineage(db):
    _compression_pair(db)
    _delegate_child(db, "tip", "delegate-run")

    assert db.delete_session("tip") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert db.get_session("delegate-run") is None


def test_delete_orphans_branch_children_not_cascades(db):
    _compression_pair(db)
    _branch_child(db, "root", "branch-run")

    assert db.delete_session("tip") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert db.get_session("branch-run") is not None
    assert db.get_session("branch-run")["parent_session_id"] is None
    # The orphaned branch surfaces as its own conversation.
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["branch-run"]


def test_get_session_delete_targets_includes_lineage(db):
    _compression_pair(db)

    # Requested session is first, then the rest of the lineage (sorted).
    assert db.get_session_delete_targets("tip") == ["tip", "root"]

    assert db.get_session_delete_targets("tip", include_compression_lineage=False) == ["tip"]


def test_expected_delete_ids_matches_lineage_targets(db):
    """Export-before-delete (CLI --lineage logical) passes targets as expected."""
    _compression_pair(db)
    targets = db.get_session_delete_targets("tip")

    assert db.delete_session("tip", expected_delete_ids=targets) is True
    assert db.get_session("root") is None
    assert db.get_session("tip") is None


def test_expected_delete_ids_fails_closed_on_mismatch(db):
    _compression_pair(db)

    assert db.delete_session("tip", expected_delete_ids=["tip"]) is False
    assert db.get_session("root") is not None
    assert db.get_session("tip") is not None


def test_bulk_delete_cascades_lineage(db):
    _compression_pair(db)

    assert db.delete_sessions(["tip"]) == 2
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
