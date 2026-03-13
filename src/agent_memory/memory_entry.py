from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry stored by an agent."""

    key: str
    content: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    access_count: int = 0

    def touch(self) -> None:
        """Record an access to this memory entry."""
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
