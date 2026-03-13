import json
import tempfile
from pathlib import Path

import pytest

from agent_memory.memory_entry import MemoryEntry
from agent_memory.memory_store import MemoryStore


class TestMemoryEntry:
    def test_create_entry(self):
        entry = MemoryEntry(key="test", content="hello")
        assert entry.key == "test"
        assert entry.content == "hello"
        assert entry.access_count == 0

    def test_touch_increments_count(self):
        entry = MemoryEntry(key="test", content="hello")
        entry.touch()
        entry.touch()
        assert entry.access_count == 2

    def test_round_trip_dict(self):
        entry = MemoryEntry(key="k", content={"nested": True}, metadata={"tag": "a"})
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored.key == entry.key
        assert restored.content == entry.content
        assert restored.metadata == entry.metadata


class TestMemoryStore:
    def test_put_and_get(self):
        store = MemoryStore()
        store.put("greeting", "hello world")
        entry = store.get("greeting")
        assert entry is not None
        assert entry.content == "hello world"

    def test_get_missing_key_returns_none(self):
        store = MemoryStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = MemoryStore()
        store.put("k", "v")
        assert store.delete("k") is True
        assert store.delete("k") is False
        assert "k" not in store

    def test_search_by_prefix(self):
        store = MemoryStore()
        store.put("user:1", "alice")
        store.put("user:2", "bob")
        store.put("system:config", "cfg")
        results = store.search("user:")
        assert len(results) == 2

    def test_list_keys(self):
        store = MemoryStore()
        store.put("a", 1)
        store.put("b", 2)
        assert sorted(store.list_keys()) == ["a", "b"]

    def test_clear(self):
        store = MemoryStore()
        store.put("a", 1)
        store.clear()
        assert len(store) == 0

    def test_max_entries_evicts_lru(self):
        store = MemoryStore(max_entries=2)
        store.put("a", 1)
        store.put("b", 2)
        store.get("a")  # access a so b is least used
        store.put("c", 3)  # should evict b
        assert "a" in store
        assert "c" in store
        assert "b" not in store

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "memory.json")
            store = MemoryStore(persist_path=path)
            store.put("k", "v", metadata={"tag": "test"})
            store.save()

            store2 = MemoryStore(persist_path=path)
            entry = store2.get("k")
            assert entry is not None
            assert entry.content == "v"
            assert entry.metadata == {"tag": "test"}
