"""日程看板：解析 schedule.snapshot，按列分组供 UI 展示。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

ColumnId = Literal["upcoming", "todo", "done"]

UPCOMING_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class ScheduleItem:
    id: str
    kind: str
    title: str
    status: str
    due_at: str | None = None
    countdown_sec: int | None = None
    column: ColumnId | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScheduleItem":
        col = raw.get("column")
        if col is not None and col not in ("upcoming", "todo", "done"):
            col = None
        return cls(
            id=raw["id"],
            kind=raw["kind"],
            title=raw["title"],
            status=raw["status"],
            due_at=raw.get("due_at"),
            countdown_sec=raw.get("countdown_sec"),
            column=col,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
        }
        if self.due_at is not None:
            out["due_at"] = self.due_at
        if self.countdown_sec is not None:
            out["countdown_sec"] = self.countdown_sec
        if self.column is not None:
            out["column"] = self.column
        return out


@dataclass
class BoardStore:
    """内存中的看板状态；由 schedule 事件更新。"""

    version: int = 0
    items: list[ScheduleItem] = field(default_factory=list)
    highlighted_ids: set[str] = field(default_factory=set)
    last_notify_text: str | None = None

    def apply_snapshot(self, data: dict[str, Any], now: datetime | None = None) -> None:
        self.version = int(data.get("version", 0))
        self.items = [ScheduleItem.from_dict(item) for item in data.get("items", [])]
        self.highlighted_ids.clear()

    def apply_updated(self, data: dict[str, Any]) -> None:
        """与 snapshot 相同语义的全量替换（schedule.updated）。"""
        self.apply_snapshot(data)

    def highlight_notify(self, text: str) -> None:
        self.last_notify_text = text
        self.highlighted_ids.clear()
        needle = text.strip()
        for item in self.items:
            if needle in item.title or item.title in needle:
                self.highlighted_ids.add(item.id)

    def columns(self, now: datetime | None = None) -> dict[ColumnId, list[dict[str, Any]]]:
        now = now or datetime.now(timezone.utc)
        grouped: dict[ColumnId, list[dict[str, Any]]] = {"upcoming": [], "todo": [], "done": []}
        for item in self.items:
            col = item.column or resolve_column(item, now)
            card = item.to_dict()
            card["board_column"] = col
            card["highlighted"] = item.id in self.highlighted_ids
            grouped[col].append(card)
        for col in grouped:
            grouped[col].sort(key=lambda c: (c.get("due_at") or "", c["title"]))
        return grouped

    def to_json(self, now: datetime | None = None) -> dict[str, Any]:
        return {
            "version": self.version,
            "last_notify_text": self.last_notify_text,
            "columns": self.columns(now),
        }


def resolve_column(item: ScheduleItem, now: datetime) -> ColumnId:
    if item.status == "done":
        return "done"
    if item.column in ("upcoming", "todo", "done"):
        return item.column
    due = parse_due_at(item.due_at)
    if due is not None:
        if due <= now + UPCOMING_WINDOW:
            return "upcoming"
        return "todo"
    if item.countdown_sec is not None:
        return "upcoming"
    return "todo"


def parse_due_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)