from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Union

import persistent
from BTrees.OOBTree import OOBTree


_MISSING = object()


def _copy_attribute_value(value: Any) -> Any:
    if hasattr(value, "copy_for_tree"):
        return value.copy_for_tree()
    return value


class BaseNode(persistent.Persistent):
    def __init__(self, key: str = "head", parent: Optional["BaseNode"] = None) -> None:
        self._parent = parent
        self._key = str(key)
        self.node_storage = OOBTree()
        self.attribute_storage = OOBTree()

    def __repr__(self) -> str:
        return (
            "BaseNode("
            f"key={self.key!r}, "
            f"nodes={len(self.node_storage)}, "
            f"attributes={len(self.attribute_storage)}"
            ")"
        )

    @property
    def key(self) -> str:
        return self._key

    @key.setter
    def key(self, key: str) -> None:
        new_key = str(key)
        old_key = self._key

        if new_key == old_key:
            return

        parent = self.parent
        if parent is None:
            self._key = new_key
            return

        parent_storage = parent.node_storage
        if new_key in parent_storage and parent_storage[new_key] is not self:
            raise KeyError(f"Node key {new_key!r} already exists under parent {parent.key!r}.")

        if old_key in parent_storage and parent_storage[old_key] is self:
            del parent_storage[old_key]

        self._key = new_key
        parent_storage[self._key] = self

    @property
    def parent(self) -> Optional["BaseNode"]:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional["BaseNode"]) -> None:
        self._parent = parent

    def get_attribute(self, attribute_key: str, default: Any = None) -> Any:
        key = str(attribute_key)
        if key in self.attribute_storage:
            return self.attribute_storage[key]
        return default

    def set_attribute(self, attribute_key: str, value: Any) -> Any:
        self.attribute_storage[str(attribute_key)] = value
        return value

    def _clone_empty(self) -> "BaseNode":
        return BaseNode(key=self.key)

    def ga(self, attribute_key: str, attribute: Any = _MISSING) -> Any:
        key = str(attribute_key)

        if attribute is _MISSING:
            if key not in self.attribute_storage:
                self.attribute_storage[key] = None
            return self.attribute_storage[key]

        self.attribute_storage[key] = attribute
        return self.attribute_storage[key]

    def get_node_attribute(self, node_keys: Iterable[str], attribute_key: str, attribute: Any = _MISSING) -> Any:
        cur_node = self.get_node_path(node_keys)
        return cur_node.ga(attribute_key, attribute)

    def gna(self, node_keys: Iterable[str], attribute_key: str, attribute: Any = _MISSING) -> Any:
        return self.get_node_attribute(node_keys, attribute_key, attribute)

    def get_node(self, node_key: str, create: bool = True) -> "BaseNode":
        key = str(node_key)

        if key not in self.node_storage:
            if not create:
                raise KeyError(f"Node {key!r} does not exist under {self.key!r}.")
            self.node_storage[key] = BaseNode(key=key, parent=self)

        return self.node_storage[key]

    def get_data_node(
        self,
        node_keys: Iterable[str],
        samplerate_hz: float | None = None,
        backend: str | None = None,
        data_attribute: str | None = None,
        create: bool = True,
    ) -> "DataNode":
        node_keys = [str(node_key) for node_key in node_keys]
        if not node_keys:
            raise ValueError("node_keys must contain at least one key.")

        cur_node: BaseNode = self
        for key in node_keys[:-1]:
            cur_node = cur_node.get_node(key, create=create)

        final_key = node_keys[-1]
        if final_key not in cur_node.node_storage:
            if not create:
                raise KeyError(f"Node {final_key!r} does not exist under {cur_node.key!r}.")
            data_node = DataNode(
                key=final_key,
                parent=cur_node,
                samplerate_hz=samplerate_hz,
                data_backend=backend or "zarr",
                data_attribute=data_attribute or "data",
            )
            cur_node.node_storage[final_key] = data_node
            return data_node

        existing = cur_node.node_storage[final_key]
        if isinstance(existing, DataNode):
            existing.configure_data_storage(
                samplerate_hz=samplerate_hz,
                backend=backend,
                data_attribute=data_attribute,
            )
            return existing

        if samplerate_hz is None:
            raise ValueError("samplerate_hz is required when upgrading a BaseNode to a DataNode.")

        data_node = DataNode.from_tree(
            existing,
            samplerate_hz=samplerate_hz,
            data_backend=backend or "zarr",
            data_attribute=data_attribute or "data",
        )
        cur_node.node_storage[final_key] = data_node
        return data_node

    def gn(self, node_key: str) -> "BaseNode":
        return self.get_node(node_key=node_key, create=True)

    def get_node_path(self, node_keys: Iterable[str], create: bool = True) -> "BaseNode":
        cur_node = self
        for key in node_keys:
            cur_node = cur_node.get_node(key, create=create)
        return cur_node

    def gns(self, node_keys: Iterable[str]) -> "BaseNode":
        return self.get_node_path(node_keys=node_keys, create=True)

    def gps(self) -> List[str]:
        if self.parent is None:
            return []
        parents = self.parent.gps()
        parents.append(self.key)
        return parents

    def get_path_keys(self) -> List[str]:
        return self.gps()

    def get_nodes(self, node_keys: Iterable[str], including: bool = True) -> List["BaseNode"]:
        keys = [str(node_key) for node_key in node_keys]

        if including:
            return [self.gn(node_key) for node_key in keys]

        return [self.gn(node_key) for node_key in self.all_node_keys() if node_key not in keys]

    def add_tree(self, tree: "BaseNode", copy: bool = False, unique_key: bool = False) -> "BaseNode":
        tree_to_add = tree.copy_tree() if copy else tree

        if tree_to_add.parent is not None and tree_to_add.parent is not self:
            old_parent = tree_to_add.parent
            old_key = tree_to_add.key
            if old_key in old_parent.node_storage and old_parent.node_storage[old_key] is tree_to_add:
                del old_parent.node_storage[old_key]
            tree_to_add.parent = None

        if unique_key:
            new_key_list = tree.gps()
            tree_key = "_~_".join(new_key_list) if new_key_list else tree_to_add.key
            tree_to_add.key = tree_key

        tree_to_add.parent = self
        self.node_storage[tree_to_add.key] = tree_to_add
        return tree_to_add

    def add_trees(self, tree_list: Iterable["BaseNode"], copy: bool = False, unique_key: bool = False) -> None:
        for tree in tree_list:
            self.add_tree(tree=tree, copy=copy, unique_key=unique_key)

    def merge_tree(self, tree: "BaseNode", overwrite: bool = False, copy: bool = False) -> "BaseNode":
        tree_to_merge = tree.copy_tree() if copy else tree

        for att_key in tree_to_merge.all_attribute_keys():
            if overwrite or att_key not in self.attribute_storage:
                self.attribute_storage[att_key] = tree_to_merge.get_attribute(att_key)

        for node in tree_to_merge.all_nodes():
            if node.key in self.node_storage:
                self.node_storage[node.key].merge_tree(node, overwrite=overwrite, copy=False)
            else:
                self.add_tree(node, copy=False, unique_key=False)

        return self

    def delete_tree(self, only_node: bool = False) -> None:
        if self.parent is None:
            raise ValueError("Cannot delete the root tree from itself.")
        self.parent.delete_node(self.key, only_node=only_node)

    def delete_node(self, node_key: str, only_node: bool = False) -> None:
        key = str(node_key)
        if key not in self.node_storage:
            return

        node = self.node_storage[key]

        if only_node:
            for child_node in list(node.all_nodes()):
                self.add_tree(child_node, copy=False, unique_key=False)

        del self.node_storage[key]

    def delete_attribute(self, attribute_key: str) -> None:
        key = str(attribute_key)
        if key in self.attribute_storage:
            del self.attribute_storage[key]

    def all_attribute_keys(self) -> List[str]:
        return list(self.attribute_storage.keys())

    def all_attributes(self) -> List[Any]:
        return list(self.attribute_storage.values())

    def all_node_keys(self) -> List[str]:
        return list(self.node_storage.keys())

    def all_nodes(self) -> List["BaseNode"]:
        return list(self.node_storage.values())

    def copy_tree(self) -> "BaseNode":
        tree_copy = self._clone_empty()

        for attribute_key in self.all_attribute_keys():
            tree_copy.set_attribute(attribute_key, _copy_attribute_value(self.get_attribute(attribute_key)))

        for node in self.all_nodes():
            tree_copy.add_tree(node, copy=True)

        return tree_copy

    def iterate_tree_leaves(self) -> Iterator["BaseNode"]:
        nodes = self.all_nodes()
        if not nodes:
            yield self
            return

        for node in nodes:
            for leaf in node.iterate_tree_leaves():
                yield leaf

    def iterate_tree_crone(self) -> Iterator["BaseNode"]:
        for node in self.iterate_tree_leaves():
            yield node

    def iterate_tree_level(self, level: Union[int, str] = "deepest") -> Iterator["BaseNode"]:
        if level == "deepest":
            level = self.get_max_depth()

        if not isinstance(level, int) or level < 0:
            raise ValueError("level must be a non-negative integer or 'deepest'.")

        if level == 0:
            yield self
            return

        for node in self.all_nodes():
            for level_node in node.iterate_tree_level(level - 1):
                yield level_node

    def _buffered_iteration(self, iter_obj: Iterator["BaseNode"], buffer_size: int) -> Iterator[List["BaseNode"]]:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0.")

        node_list = []
        for next_node in iter_obj:
            node_list.append(next_node)
            if len(node_list) == buffer_size:
                yield node_list
                node_list = []

        if node_list:
            yield node_list

    def iterate_tree_level_buffered(self, buffer_size: int = 100_000, level: Union[int, str] = "deepest") -> Iterator[List["BaseNode"]]:
        iter_obj = self.iterate_tree_level(level=level)
        return self._buffered_iteration(iter_obj=iter_obj, buffer_size=buffer_size)

    def iterate_tree_crone_buffered(self, buffer_size: int = 100_000) -> Iterator[List["BaseNode"]]:
        iter_obj = self.iterate_tree_crone()
        return self._buffered_iteration(iter_obj=iter_obj, buffer_size=buffer_size)

    def get_main_tree(self) -> "BaseNode":
        if self.parent is None:
            return self
        return self.parent.get_main_tree()

    def get_max_depth(self, counter: int = 0) -> int:
        nodes = self.all_nodes()
        if not nodes:
            return counter

        max_depth = counter
        for node in nodes:
            cur_max_depth = node.get_max_depth(counter + 1)
            max_depth = max(max_depth, cur_max_depth)

        return max_depth

    def display(self, counter: int = 1) -> None:
        nodes = self.all_nodes()
        if nodes:
            print("n-" * counter, self.key)
            if self.parent is not None:
                print("p-" * counter, self.parent.key)

            for node in nodes:
                print("~n" * counter, counter, node.key)
                print("~a" * counter, node.all_attributes())
                node.display(counter + 1)

    def unconnect(self) -> "BaseNodeUnconnected":
        return BaseNodeUnconnected(self)


class BaseNodeUnconnected:
    def __init__(self, tree: BaseNode) -> None:
        self.parents = tree.gps()


class DataNode(BaseNode):
    def __init__(
        self,
        key: str = "data",
        parent: Optional[BaseNode] = None,
        samplerate_hz: float | None = None,
        data_backend: str = "zarr",
        data_attribute: str = "data",
    ) -> None:
        super().__init__(key=key, parent=parent)
        self.samplerate_hz = samplerate_hz
        self.data_backend = str(data_backend)
        self.data_attribute = str(data_attribute)

    @classmethod
    def from_tree(
        cls,
        tree: BaseNode,
        samplerate_hz: float,
        data_backend: str = "zarr",
        data_attribute: str = "data",
    ) -> "DataNode":
        data_node = cls(
            key=tree.key,
            parent=tree.parent,
            samplerate_hz=samplerate_hz,
            data_backend=data_backend,
            data_attribute=data_attribute,
        )
        data_node.attribute_storage = tree.attribute_storage
        data_node.node_storage = tree.node_storage
        for child_node in data_node.all_nodes():
            child_node.parent = data_node
        return data_node

    def _clone_empty(self) -> "DataNode":
        return DataNode(
            key=self.key,
            samplerate_hz=self.samplerate_hz,
            data_backend=self.data_backend,
            data_attribute=self.data_attribute,
        )

    def configure_data_storage(
        self,
        samplerate_hz: float | None = None,
        backend: str | None = None,
        data_attribute: str | None = None,
    ) -> None:
        if samplerate_hz is not None:
            if samplerate_hz <= 0:
                raise ValueError("samplerate_hz must be greater than 0.")
            self.samplerate_hz = float(samplerate_hz)
        if backend is not None:
            self.data_backend = str(backend)
        if data_attribute is not None:
            self.data_attribute = str(data_attribute)

    def write_data(
        self,
        array: Any,
        database=None,
        data_attribute: str | None = None,
        backend: str | None = None,
        samplerate_hz: float | None = None,
        metadata: Optional[dict[str, Any]] = None,
        chunks: Optional[tuple[int, ...]] = None,
        zarr_store_path: str | None = None,
        content_type: str = "application/x-npy",
    ):
        from .measurements import DataWriter

        if samplerate_hz is None:
            samplerate_hz = self.samplerate_hz
        if samplerate_hz is None:
            raise ValueError("samplerate_hz must be configured on the DataNode or passed explicitly.")

        data_attribute = data_attribute or self.data_attribute
        backend = backend or self.data_backend
        self.configure_data_storage(
            samplerate_hz=samplerate_hz,
            backend=backend,
            data_attribute=data_attribute,
        )

        return DataWriter.write_array(
            self,
            array,
            samplerate_hz=samplerate_hz,
            data_attribute=data_attribute,
            backend=backend,
            metadata=metadata,
            database=database,
            zarr_store_path=zarr_store_path,
            chunks=chunks,
            content_type=content_type,
        )

    def read_data(self, start: Any = None, stop: Any = None, unit: str = "samples", data_attribute: str | None = None):
        from .measurements import DataReader

        return DataReader.read_interval(
            self,
            data_attribute=data_attribute or self.data_attribute,
            start=start,
            stop=stop,
            unit=unit,
        )

    def read_seconds(self, start_second: float | None = None, stop_second: float | None = None, data_attribute: str | None = None):
        from .measurements import DataReader

        return DataReader.read_seconds(
            self,
            data_attribute=data_attribute or self.data_attribute,
            start_second=start_second,
            stop_second=stop_second,
        )

    def iter_data_blocks(
        self,
        block_size: float | int,
        block_unit: str = "samples",
        start: Any = None,
        stop: Any = None,
        range_unit: str = "samples",
        data_attribute: str | None = None,
    ):
        from .measurements import DataReader

        return DataReader.iter_blocks(
            self,
            data_attribute=data_attribute or self.data_attribute,
            block_size=block_size,
            block_unit=block_unit,
            start=start,
            stop=stop,
            range_unit=range_unit,
        )
