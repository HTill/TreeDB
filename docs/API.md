# LOTDB API

## Core classes

### `BaseNode`
- Hierarchical persistent node.
- Stores child nodes in `node_storage`.
- Stores metadata/attributes in `attribute_storage`.

Readable methods:
- `get_node(key, create=True)`
- `get_node_path(keys, create=True)`
- `get_attribute(key, default=None)`
- `set_attribute(key, value)`

Compatibility aliases:
- `gn()`
- `gns()`
- `ga()`
- `gna()`
- `gps()`

Important behaviors:
- Missing nodes are created by default.
- `ga(key)` preserves legacy behavior and creates the attribute with value `None` if missing.
- `copy_tree()` returns a detached deep copy.
- `merge_tree()` merges child trees recursively.
- `delete_node(..., only_node=True)` removes a node but promotes its children.

### `LOTDB`
- ZODB-backed storage for a root `BaseNode`.
- `open_connection()` opens a transaction-scoped connection.
- `commit()` must be called explicitly for writes.
- `connection()` provides a context manager that closes the connection automatically.

### `PathFileObj`
- Persistent file path object storing root and filename separately.
- Useful for keeping file references inside the tree.

## File helpers

Files are treated as ingestion sources.

Use:
- `DataWriter.attach_file(...)`

This loads the external file, converts it into the configured backend representation, and stores source metadata on the node attributes.

The older direct file-writing helper methods were removed in favor of backend-first ingestion.

## Blob helpers

### `BlobObject`
- ZODB-managed binary payload stored inside the database/blob directory.
- Can store raw bytes or NumPy arrays.
- Supports metadata like sensor type, sample rate, units, and shape.

### `BlobReader` / `BlobWriter`
- `BlobWriter.write_bytes()` stores arbitrary binary payloads on a node.
- `BlobWriter.write_array()` stores NumPy arrays in `.npy` format inside a blob.
- `BlobReader.read_bytes()` and `BlobReader.read_array()` restore the payload.

Typical use case:
- sensor measurement sequences
- multichannel waveform data
- binary captures that should be managed by LOTDB instead of external file paths

## Data helpers

### `DataWriter`
- Stores sensor arrays using a selected backend: `blob` or `zarr`.
- Attaches a data reference object to a node.
- Records metadata like sample rate, dtype, shape, dataset path, and custom sensor metadata.
- Also supports file ingestion through `attach_file(...)`.

### `DataReader`
- Reads data ranges in samples or seconds.
- Provides `iter_blocks(...)` for streaming/block-based consumption.
- Converts second-based ranges into sample indices using `samplerate_hz`.
- Reads from the configured backend; it does not need the original source file anymore.

### `DataObject`
- Persistent data reference stored on a node.
- For `blob`, the payload lives in a ZODB blob.
- For `zarr`, the node stores a link to a Zarr dataset path on disk.

### `DataNode`
- Specialized data-oriented node that inherits from `BaseNode`.
- Stores default data configuration like backend, sample rate, and data attribute name.
- Convenience methods:
  - `write_data(...)`
  - `replace_data(...)`
  - `append_data(...)`
  - `has_data()`
  - `delete_data()`
  - `read_data(...)`
  - `read_seconds(...)`
  - `iter_data_blocks(...)`

`write_data(...)` supports explicit payload policies:
- `if_exists="replace"`
- `if_exists="error"`
- `if_exists="append"`

### `BaseNode.get_data_node()`
- Creates or upgrades the final node in a path to a `DataNode`.
- This is the easiest way to create sensor capture nodes with automatic backend setup.

## Bulk loading helpers

These file-indexing helpers live in `lotdb.utils`.

### `lotdb.utils.load_files_folder()`
Creates nodes from filenames split by a separator.

### `lotdb.utils.load_files_directory()`
Creates nodes from folder hierarchy and filenames.

## Data/export helpers

- `filter_node_list()`
- `tree_to_dataframe()`
- `node_list_to_dataframe()`
- `node_process_cruncher()`
