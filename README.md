# Agent Memory Management

A lightweight memory management system for AI agents, providing key-value storage with persistence, prefix-based search, and LRU eviction.

## Features

- **Key-value memory store** with typed entries
- **Persistence** to JSON files
- **Prefix search** for organized memory namespaces
- **LRU eviction** when max capacity is reached
- **Serialization** with `to_dict` / `from_dict` round-tripping

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from agent_memory import MemoryStore

store = MemoryStore(persist_path="agent_memory.json", max_entries=5000)

# Store memories
store.put("user:name", "Alice", metadata={"source": "onboarding"})
store.put("task:current", {"goal": "summarize document", "status": "in_progress"})

# Retrieve
entry = store.get("user:name")
print(entry.content)  # "Alice"

# Search by prefix
user_memories = store.search("user:")

# Persist to disk
store.save()
```

## Running Tests

```bash
pytest
```
