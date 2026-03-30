from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Union

import persistent
from BTrees.OOBTree import OOBTree


_MISSING = object()


class StorageTree(persistent.Persistent):
    def __init__(self, key: str = "head", parent: Optional["StorageTree"] = None) -> None:
        self._parent = parent
        self._key = str(key)
        self.node_storage = OOBTree()
        self.attribute_storage = OOBTree()

    def __repr__(self) -> str:
        return (
            "StorageTree("
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
    def parent(self) -> Optional["StorageTree"]:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional["StorageTree"]) -> None:
        self._parent = parent

    def get_attribute(self, attribute_key: str, default: Any = None) -> Any:
        key = str(attribute_key)
        if key in self.attribute_storage:
            return self.attribute_storage[key]
        return default

    def set_attribute(self, attribute_key: str, value: Any) -> Any:
        self.attribute_storage[str(attribute_key)] = value
        return value

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

    def get_node(self, node_key: str, create: bool = True) -> "StorageTree":
        key = str(node_key)

        if key not in self.node_storage:
            if not create:
                raise KeyError(f"Node {key!r} does not exist under {self.key!r}.")
            self.node_storage[key] = StorageTree(key=key, parent=self)

        return self.node_storage[key]

    def gn(self, node_key: str) -> "StorageTree":
        return self.get_node(node_key=node_key, create=True)

    def get_node_path(self, node_keys: Iterable[str], create: bool = True) -> "StorageTree":
        cur_node = self
        for key in node_keys:
            cur_node = cur_node.get_node(key, create=create)
        return cur_node

    def gns(self, node_keys: Iterable[str]) -> "StorageTree":
        return self.get_node_path(node_keys=node_keys, create=True)

    def gps(self) -> List[str]:
        if self.parent is None:
            return []
        parents = self.parent.gps()
        parents.append(self.key)
        return parents

    def get_path_keys(self) -> List[str]:
        return self.gps()

    def get_nodes(self, node_keys: Iterable[str], including: bool = True) -> List["StorageTree"]:
        keys = [str(node_key) for node_key in node_keys]

        if including:
            return [self.gn(node_key) for node_key in keys]

        return [self.gn(node_key) for node_key in self.all_node_keys() if node_key not in keys]

    def add_tree(self, tree: "StorageTree", copy: bool = False, unique_key: bool = False) -> "StorageTree":
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

    def add_trees(self, tree_list: Iterable["StorageTree"], copy: bool = False, unique_key: bool = False) -> None:
        for tree in tree_list:
            self.add_tree(tree=tree, copy=copy, unique_key=unique_key)

    def merge_tree(self, tree: "StorageTree", overwrite: bool = False, copy: bool = False) -> "StorageTree":
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

    def all_nodes(self) -> List["StorageTree"]:
        return list(self.node_storage.values())

    def copy_tree(self) -> "StorageTree":
        tree_copy = StorageTree(key=self.key)

        for attribute_key in self.all_attribute_keys():
            tree_copy.set_attribute(attribute_key, self.get_attribute(attribute_key))

        for node in self.all_nodes():
            tree_copy.add_tree(node, copy=True)

        return tree_copy

    def iterate_tree_leaves(self) -> Iterator["StorageTree"]:
        nodes = self.all_nodes()
        if not nodes:
            yield self
            return

        for node in nodes:
            for leaf in node.iterate_tree_leaves():
                yield leaf

    def iterate_tree_crone(self) -> Iterator["StorageTree"]:
        for node in self.iterate_tree_leaves():
            yield node

    def iterate_tree_level(self, level: Union[int, str] = "deepest") -> Iterator["StorageTree"]:
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

    def _buffered_iteration(self, iter_obj: Iterator["StorageTree"], buffer_size: int) -> Iterator[List["StorageTree"]]:
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

    def iterate_tree_level_buffered(self, buffer_size: int = 100_000, level: Union[int, str] = "deepest") -> Iterator[List["StorageTree"]]:
        iter_obj = self.iterate_tree_level(level=level)
        return self._buffered_iteration(iter_obj=iter_obj, buffer_size=buffer_size)

    def iterate_tree_crone_buffered(self, buffer_size: int = 100_000) -> Iterator[List["StorageTree"]]:
        iter_obj = self.iterate_tree_crone()
        return self._buffered_iteration(iter_obj=iter_obj, buffer_size=buffer_size)

    def get_main_tree(self) -> "StorageTree":
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

    def unconnect(self) -> "StorageTreeUnconnected":
        return StorageTreeUnconnected(self)


class StorageTreeUnconnected:
    def __init__(self, tree: StorageTree) -> None:
        self.parents = tree.gps()
