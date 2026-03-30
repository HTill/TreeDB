# LOTDB

Light Object Tree DB.

LOTDB is a persistent tree for organizing computation results, metadata, and large data payloads with very little boilerplate.

## Core idea

LOTDB is built around two node types:
- `BaseNode` for generic hierarchy and metadata
- `DataNode` for hierarchy plus data-oriented read/write behavior

Each node can:
- contain child nodes
- store arbitrary attributes
- be persisted with ZODB

Large payloads can be stored through pluggable backends such as:
- `blob`
- `zarr`

This makes LOTDB useful for pipelines where you want to:
- branch variants of computations
- cache intermediate results on disk
- keep metadata close to the computation tree
- avoid manually managing lots of folder/path boilerplate

## Installation

Install from PyPI:

`pip install lotdb`

With optional extras:

`pip install "lotdb[io,measurements]"`

## Mental model

- `BaseNode` = generic tree node
- `DataNode` = specialized node for data payloads
- `LOTDB` = persistent container for the root tree

The recommended API is node-centered.

## Quick start

```python
from lotdb import BaseNode

root = BaseNode(key="dataset")
node = root.get_node_path(["speaker_01", "session_a", "clip_001"])
node.set_attribute("label", "hello")

print(node.get_attribute("label"))
```

## Persistent usage

```python
from lotdb import LOTDB

db = LOTDB(path="./data", name="lotdb.fs", new=True)
root = db.open_connection()

root.get_node_path(["speaker_01", "clip_001"]).set_attribute("duration", 1.23)

db.commit()
db.close_connection()
db.close()
```

## Recommended data workflow

Use `get_data_node(...)` when the final node should own data behavior.

```python
import numpy as np
from lotdb import LOTDB

db = LOTDB(path="./data", name="lotdb.fs", new=True)
root = db.open_connection()

capture_1 = root.get_data_node(
    ["sensor_01", "capture_0001"],
    samplerate_hz=1000,
    backend="zarr",   # or "blob"
    data_attribute="imu",
)

capture_2 = root.get_data_node(
    ["sensor_01", "capture_0002"],
    samplerate_hz=1000,
    backend="zarr",
    data_attribute="imu",
)

data = np.random.randn(2000, 6).astype("float32")
capture_1.write_data(data, database=db)
capture_2.write_data(data * 0.5, database=db)

# same node, second payload
capture_1.write_data(np.random.randn(2000, 2).astype("float32"), database=db, data_attribute="control")

window = capture_1.read_seconds(1.0, 2.0)

for block in capture_1.iter_data_blocks(0.5, block_unit="seconds"):
    process(block)

# first iteration layer under the root
for node in root.iterate_tree_level(1):
    print(node.key)

# first iteration layer under sensor_01
sensor_node = root.get_node_path(["sensor_01"])
for node in sensor_node.iterate_tree_level(1):
    print(node.key)

# all leaves below sensor_01
for leaf in sensor_node.iterate_tree_leaves():
    print("leaf", leaf.key)

# buffered iteration over leaves
for batch in sensor_node.iterate_tree_crone_buffered(buffer_size=2):
    print([node.key for node in batch])

db.commit()
db.close_connection()
db.close()
```

This is the main API LOTDB is optimized for.

## Generic nodes vs data nodes

Use `BaseNode` when you only need:
- hierarchy
- metadata
- relationships

Use `DataNode` when you want the node itself to own:
- `write_data(...)`
- `replace_data(...)`
- `append_data(...)`
- `has_data()`
- `delete_data()`
- `read_data(...)`
- `read_seconds(...)`
- `iter_data_blocks(...)`

`get_node(...)` remains the generic retriever.
`get_data_node(...)` ensures the final node is a `DataNode`.

## Lower-level data API

The lower-level API still exists when you want direct control:

```python
import numpy as np
from lotdb import DataReader, DataWriter

DataWriter.write_array(
    node,
    np.random.randn(1000, 4).astype("float32"),
    samplerate_hz=500,
    backend="blob",
    data_attribute="signal",
)

signal = DataReader.read_interval(node, "signal", 100, 200)
```

`write_data(...)` supports explicit payload policies:
- `if_exists="replace"` (default)
- `if_exists="error"`
- `if_exists="append"`

This is especially useful when one `DataNode` stores multiple payloads such as:
- `audio`
- `control`

Example:

```python
capture.write_data(audio, database=db, data_attribute="audio", if_exists="replace")
capture.write_data(control, database=db, data_attribute="control", if_exists="replace")

capture.append_data(more_audio, database=db, data_attribute="audio")

if capture.has_data("control"):
    capture.delete_data("control")
```

## File ingestion

External files are now treated as ingestion inputs.

`DataWriter.attach_file(...)` or `DataNode.attach_file(...)` loads the file, converts it into the configured backend representation, and stores source metadata on the node attributes.

After that, reads happen only through the backend:

```python
data_node.attach_file("./capture.npy", database=db, data_attribute="imu")
payload = data_node.read_data()
```

Typical formats currently supported:
- `wav`
- `npy`
- `csv`
- `txt`
- `png` / `jpg` / `jpeg`
- raw bytes for unknown formats

Typical source attributes written onto the node:
- `_source_filepath`
- `_source_format`
- `_source_filename`
- `_source_samplerate_hz` for wav files

The older direct file-writing helper methods were removed to keep the public API focused on backend-managed data and ingestion.

## Why LOTDB instead of only HDF5/Zarr?

LOTDB is not just a container for arrays.

It is useful when you want:
- a persistent computation tree
- easy branching of variants
- metadata attached directly to pipeline nodes
- cached intermediate states across runs
- backend flexibility for how payloads are stored

## Development

- source package: `src/lotdb`
- tests: `tests/`
- API notes: `docs/API.md`

Development install:

`pip install -e .`

With extras:

`pip install -e .[io,measurements]`

## Publishing

PyPI publishing is configured through GitHub Actions trusted publishing.

Typical release flow:
1. bump version in `pyproject.toml`
2. push to `main`
3. create a GitHub release
4. GitHub publishes to PyPI
