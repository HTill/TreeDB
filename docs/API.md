# LOTDB API

## Core classes

### `StorageTree`
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
- ZODB-backed storage for a root `StorageTree`.
- `open_connection()` opens a transaction-scoped connection.
- `commit()` must be called explicitly for writes.
- `connection()` provides a context manager that closes the connection automatically.

Compatibility alias:
- `StorageTreeDatabase`

### `PathFileObj`
- Persistent file path object storing root and filename separately.
- Useful for keeping file references inside the tree.

## File helpers

### `add_file(node, ...)`
Attach a `PathFileObj` to a node attribute.

### `FileReader` / `FileWriter`
Helpers for wav/txt/npy oriented workflows.

## Bulk loading helpers

### `LOTDB.load_files_folder()`
Creates nodes from filenames split by a separator.

### `LOTDB.load_files_directory()`
Creates nodes from folder hierarchy and filenames.

## Data/export helpers

- `filter_node_list()`
- `tree_to_dataframe()`
- `node_list_to_dataframe()`
- `node_process_cruncher()`
