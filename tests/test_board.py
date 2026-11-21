import uuid
from datetime import datetime, timedelta, timezone

from app.board import BoardStore, ScheduleItem, resolve_column


def test_resolve_column_done():
    item = ScheduleItem("1", "task", "x", "done")
    assert resolve_column(item, datetime.now(timezone.utc)) == "done"


def test_resolve_column_due_soon():
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    item = ScheduleItem("1", "reminder", "开会", "pending", due_at=soon)
    assert resolve_column(item, datetime.now(timezone.utc)) == "upcoming"


def test_resolve_column_no_due():
    item = ScheduleItem("1", "task", "整理", "pending")
    assert resolve_column(item, datetime.now(timezone.utc)) == "todo"


def test_apply_snapshot_and_columns():
    store = BoardStore()
    store.apply_snapshot({
        "version": 2,
        "items": [
            {"id": str(uuid.uuid4()), "kind": "task", "title": "A", "status": "pending"},
            {"id": str(uuid.uuid4()), "kind": "reminder", "title": "B", "status": "done"},
        ],
    })
    cols = store.columns()
    assert store.version == 2
    assert len(cols["todo"]) == 1
    assert len(cols["done"]) == 1


def test_highlight_notify_by_title():
    store = BoardStore()
    item_id = "00112233-4455-6677-8899-aabbccddeeff"
    store.apply_snapshot({
        "version": 1,
        "items": [{"id": item_id, "kind": "reminder", "title": "吃药", "status": "pending"}],
    })
    store.highlight_notify("提醒你：吃药")
    assert item_id in store.highlighted_ids