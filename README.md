# TreeDB

Persistent tree database for hierarchical metadata and file-backed datasets.

## What it is

TreeDB stores data in `StorageTree` nodes:
- each node can contain child nodes
- each node can store arbitrary attributes
- the full tree can be persisted with ZODB

This is especially useful for dataset pipelines where folder structure, file references, and metadata all belong together.

## Installation

Local editable install:

`pip install -e .`

With optional IO helpers:

`pip install -e .[io]`

## Basic usage

```python
from TreeDB import StorageTree

tree = StorageTree(key="dataset")
node = tree.get_node_path(["speaker_01", "session_a", "clip_001"])
node.set_attribute("label", "hello")

print(node.get_attribute("label"))
```

## Persistent database usage

```python
from TreeDB import StorageTreeDatabase

db = StorageTreeDatabase(path="./data", name="treedb.fs", new=True)
tree = db.open_connection()

tree.get_node_path(["speaker_01", "clip_001"]).set_attribute("duration", 1.23)
db.commit()
db.close_connection()
db.close()
```

## Backward compatibility

The old short method names still work:
- `ga()`
- `gn()`
- `gns()`
- `gna()`

Readable wrappers were added so the API is easier to maintain and easier for weaker coding models to follow.

## Development

- source package lives in `src/TreeDB`
- tests live in `tests/`
- API notes live in `docs/API.md`
