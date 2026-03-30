# LOTDB

Persistent tree database for hierarchical metadata and file-backed datasets.

LOTDB stands for Light Object Tree DB.

## What it is

LOTDB stores data in `StorageTree` nodes:
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
from lotdb import StorageTree

tree = StorageTree(key="dataset")
node = tree.get_node_path(["speaker_01", "session_a", "clip_001"])
node.set_attribute("label", "hello")

print(node.get_attribute("label"))
```

## Persistent database usage

```python
from lotdb import LOTDB

db = LOTDB(path="./data", name="lotdb.fs", new=True)
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

Import compatibility is also preserved for now:
- new import: `import lotdb`
- legacy imports still work: `import TreeDB`, `import StorageTree`
- new database class: `LOTDB`
- legacy alias still works: `StorageTreeDatabase`

## Development

- source packages live in `src/lotdb` and `src/TreeDB`
- tests live in `tests/`
- API notes live in `docs/API.md`

## Publishing

PyPI publishing is configured with GitHub Actions via trusted publishing.

Recommended setup:
- create a PyPI project named `lotdb`
- add a trusted publisher on PyPI for this GitHub repository
- publish by creating a GitHub release

TestPyPI is also configured.

Recommended rollout:
- first connect this repo to TestPyPI trusted publishing
- trigger the `Publish to TestPyPI` workflow manually or with a tag like `test-v1.1.0`
- verify install from TestPyPI
- then connect the same repo/workflow to real PyPI
- publish a GitHub release for the real upload

Example test install:

`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lotdb`
