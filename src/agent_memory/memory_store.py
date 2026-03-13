import json
from pathlib import Path
from typing import Optional

from agent_memory.memory_entry import MemoryEntry


class MemoryStore:
    """Key-value memory store for agent state with optional persistence."""

    def __init__(self, persist_path: Optional[str] = None, max_entries: int = 10000):
        self._store: dict[str, MemoryEntry] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_entries = max_entries

        if self._persist_path and self._persist_path.exists():
            self._load()

    def put(self, key: str, content, metadata: Optional[dict] = None) -> MemoryEntry:
        """Store a memory entry. Overwrites if key exists."""
        if len(self._store) >= self._max_entries and key not in self._store:
            self._evict_lru()

        entry = MemoryEntry(key=key, content=content, metadata=metadata or {})
        self._store[key] = entry
        return entry

    def get(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key. Returns None if not found."""
        entry = self._store.get(key)
        if entry:
            entry.touch()
        return entry

    def delete(self, key: str) -> bool:
        """Delete a memory entry. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def search(self, prefix: str) -> list[MemoryEntry]:
        """Return all entries whose keys start with the given prefix."""
        return [e for k, e in self._store.items() if k.startswith(prefix)]

    def list_keys(self) -> list[str]:
        """Return all keys in the store."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def save(self) -> None:
        """Persist the store to disk."""
        if not self._persist_path:
            raise ValueError("No persist_path configured")
        data = {k: v.to_dict() for k, v in self._store.items()}
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load the store from disk."""
        raw = json.loads(self._persist_path.read_text())
        self._store = {k: MemoryEntry.from_dict(v) for k, v in raw.items()}

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry (lowest access_count)."""
        if not self._store:
            return
        lru_key = min(self._store, key=lambda k: self._store[k].access_count)
        del self._store[lru_key]
