# LOTDB

Persistent tree database for hierarchical metadata and file-backed datasets.

LOTDB stands for Light Object Tree DB.

## What it is

LOTDB stores data in `BaseNode` nodes:
- each node can contain child nodes
- each node can store arbitrary attributes
- the full tree can be persisted with ZODB

This is especially useful for dataset pipelines where folder structure, file references, and metadata all belong together.

LOTDB now supports both:
- file references for external assets
- native ZODB blob storage for large sensor measurements and array payloads
- backend-linked measurements via blob or Zarr

## Installation

Local editable install:

`pip install -e .`

With optional IO helpers:

`pip install -e .[io]`

## Basic usage

```python
from lotdb import BaseNode

tree = BaseNode(key="dataset")
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

## Sensor/blob storage usage

```python
import numpy as np
from lotdb import BlobReader, BlobWriter, LOTDB

db = LOTDB(path="./data", name="lotdb.fs", new=True)
tree = db.open_connection()

node = tree.get_node_path(["sensor_01", "capture_0001"])
BlobWriter.write_array(
    node,
    np.random.randn(1000, 6).astype("float32"),
    blob_attribute="measurement",
    metadata={"samplerate_hz": 1000, "sensor_type": "imu"},
)

db.commit()
measurement = BlobReader.read_array(node, "measurement")
db.close_connection()
db.close()
```

File-path based helpers are now folded into `DataReader` / `DataWriter` instead of living as separate reader/writer utility classes.

## Data backend usage

```python
import numpy as np
from lotdb import DataReader, DataWriter, LOTDB

db = LOTDB(path="./data", name="lotdb.fs", new=True)
tree = db.open_connection()
node = tree.get_node_path(["sensor_01", "capture_0002"])

DataWriter.write_array(
    node,
    np.random.randn(2000, 6).astype("float32"),
    samplerate_hz=1000,
    backend="zarr",  # or "blob"
    data_attribute="imu",
    database=db,
    metadata={"sensor_type": "imu", "unit": "m/s^2"},
)

for block in DataReader.iter_blocks(
    node,
    "imu",
    block_size=0.5,
    block_unit="seconds",
):
    process(block)

second_one_to_two = DataReader.read_seconds(node, "imu", 1.0, 2.0)

db.commit()
db.close_connection()
db.close()
```

## Data node API

```python
data_node = tree.get_data_node(
    ["sensor_01", "capture_0003"],
    samplerate_hz=1000,
    backend="zarr",
    data_attribute="imu",
)

data_node.write_data(data, database=db)

for block in data_node.iter_data_blocks(0.25, block_unit="seconds"):
    process(block)

window = data_node.read_seconds(1.0, 2.0)
```

This is the recommended high-level API for large binary/data payloads.

## Development

- source package lives in `src/lotdb`
- tests live in `tests/`
- API notes live in `docs/API.md`

## Publishing

PyPI publishing is configured with GitHub Actions via trusted publishing.

Recommended setup:
- create a PyPI project named `lotdb`
- add a trusted publisher on PyPI for this GitHub repository
- publish by creating a GitHub release
