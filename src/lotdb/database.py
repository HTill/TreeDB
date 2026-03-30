from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import ZODB
import ZODB.FileStorage
import transaction

from .measurements import DataWriter
from .paths import PathFileObj
from .tree import BaseNode, BaseNodeUnconnected


def _resolve_parameter_separator(parameter_separator: Optional[str], parameter_seperator: Optional[str]) -> Optional[str]:
    if parameter_separator is None and parameter_seperator is not None:
        return parameter_seperator
    return parameter_separator


class NodeConnection:
    def __init__(
        self,
        database: "LOTDB",
        connection_id: str = "standard",
        unconnected_tree: Optional[BaseNodeUnconnected] = None,
    ) -> None:
        self.database = database
        self.connection_id = connection_id
        self.unconnected_tree = unconnected_tree

    def __enter__(self) -> BaseNode:
        return self.database.open_connection(
            unconnected_tree=self.unconnected_tree,
            connection_id=self.connection_id,
        )

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.database.abort(self.connection_id)
        self.database.close_connection(self.connection_id)
        return False


class LOTDB:
    def __init__(
        self,
        path: str = "",
        name: str = "LOTDB",
        new: bool = False,
        read_only: bool = False,
        cache_size: int | None = None,
        auto_minimize_cache: bool | None = None,
    ) -> None:
        self.path = path
        self.name = name
        self.read_only = read_only
        self.cache_size = 1 if cache_size is None and read_only else (cache_size or 400)
        self.auto_minimize_cache = read_only if auto_minimize_cache is None else auto_minimize_cache
        self.db = self.setup_storage_tree_db(new=new, read_only=read_only)
        self.conn_dict: Dict[str, Tuple[Any, Any]] = {}

    def _storage_filepath(self) -> str:
        if self.path:
            return PathFileObj(root=self.path, file=self.name).filepath
        return self.name

    def _blob_directory(self) -> str:
        blob_folder = f"{self.name}.blobs"
        if self.path:
            return PathFileObj(root=self.path, file=blob_folder).filepath
        return blob_folder

    def data_store_path(self) -> str:
        data_folder = f"{self.name}.data.zarr"
        if self.path:
            return PathFileObj(root=self.path, file=data_folder).filepath
        return data_folder

    def setup_storage_tree_db(self, new: bool = False, read_only: bool = False):
        if self.path:
            os.makedirs(self.path, exist_ok=True)

        storage = ZODB.FileStorage.FileStorage(
            self._storage_filepath(),
            create=new,
            read_only=read_only,
            blob_dir=self._blob_directory(),
        )
        db = ZODB.DB(storage, cache_size=self.cache_size)

        transaction_manager = transaction.TransactionManager()
        conn = db.open(transaction_manager=transaction_manager)
        if not hasattr(conn.root, "stt") and read_only:
            conn.close()
            db.close()
            raise FileNotFoundError("Cannot open LOTDB in read-only mode before it has been initialized.")

        if not hasattr(conn.root, "stt"):
            conn.root.stt = BaseNode(key=self.name)
            transaction_manager.commit()
        if self.auto_minimize_cache:
            conn.cacheMinimize()
        conn.close()
        return db

    def open_connection(
        self,
        unconnected_tree: Optional[BaseNodeUnconnected] = None,
        connection_id: str = "standard",
    ) -> BaseNode:
        if connection_id in self.conn_dict:
            raise ValueError(f"Connection {connection_id!r} is already open.")

        transaction_manager = transaction.TransactionManager()
        conn = self.db.open(transaction_manager=transaction_manager)
        self.conn_dict[connection_id] = (conn, transaction_manager)

        tree = conn.root.stt
        if unconnected_tree is not None:
            tree = tree.gns(unconnected_tree.parents)

        return tree

    def connection(
        self,
        connection_id: str = "standard",
        unconnected_tree: Optional[BaseNodeUnconnected] = None,
    ) -> NodeConnection:
        return NodeConnection(self, connection_id=connection_id, unconnected_tree=unconnected_tree)

    def close_connection(self, connection_id: str = "standard") -> None:
        if connection_id not in self.conn_dict:
            return

        conn, transaction_manager = self.conn_dict[connection_id]
        if self.auto_minimize_cache:
            conn.cacheMinimize()
        transaction_manager.abort()
        conn.close()
        del self.conn_dict[connection_id]

    def commit(self, connection_id: str = "standard") -> None:
        if self.read_only:
            raise RuntimeError("Cannot commit a read-only LOTDB connection.")
        if connection_id not in self.conn_dict:
            return

        _, transaction_manager = self.conn_dict[connection_id]
        transaction_manager.commit()

    def minimize_cache(self, connection_id: str = "standard") -> None:
        if connection_id not in self.conn_dict:
            return

        conn, _ = self.conn_dict[connection_id]
        conn.cacheMinimize()

    def abort(self, connection_id: str = "standard") -> None:
        if connection_id not in self.conn_dict:
            return

        _, transaction_manager = self.conn_dict[connection_id]
        transaction_manager.abort()

    def close(self) -> None:
        for connection_id in list(self.conn_dict.keys()):
            self.close_connection(connection_id=connection_id)
        self.db.close()

    def load_files_directory(
        self,
        dir_path: str,
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
        filepath_attribute: str = "_pfo_audio_wav",
        file_extension: str = "wav",
        commit_every: Optional[int] = None,
    ) -> int:
        separator = _resolve_parameter_separator(parameter_separator, parameter_seperator)
        created_connection = "standard" not in self.conn_dict
        tree = self.open_connection() if created_connection else self.conn_dict["standard"][0].root.stt

        root_folder = os.path.basename(os.path.normpath(dir_path))
        parent_dir = os.path.dirname(os.path.normpath(dir_path))
        processed = 0

        for root, _, files in os.walk(dir_path):
            for file_name in files:
                if not file_name.endswith(f".{file_extension}"):
                    continue

                relative_root = os.path.relpath(root, start=parent_dir)
                path_parts = relative_root.split(os.path.sep) if relative_root != "." else [root_folder]
                last_node = tree.gns(path_parts)

                stem = os.path.splitext(file_name)[0]
                cur_node = last_node.gn(stem) if separator is None else last_node.gns(stem.split(separator))
                DataWriter.attach_file_reference(cur_node, filepath_attribute=filepath_attribute, root=root, file=file_name)

                processed += 1
                if commit_every and processed % commit_every == 0:
                    self.commit()

        if processed:
            self.commit()

        if created_connection:
            self.close_connection()

        return processed

    def load_files_folder(
        self,
        folder_path: str,
        parameter_separator: Optional[str] = "_~_",
        parameter_seperator: Optional[str] = None,
        filepath_attribute: str = "_pfo_audio_wav",
        file_extension: str = "wav",
        commit_every: Optional[int] = None,
    ) -> int:
        separator = _resolve_parameter_separator(parameter_separator, parameter_seperator)
        created_connection = "standard" not in self.conn_dict
        tree = self.open_connection() if created_connection else self.conn_dict["standard"][0].root.stt

        processed = 0
        for file_name in os.listdir(folder_path):
            if not file_name.endswith(f".{file_extension}"):
                continue

            stem = os.path.splitext(file_name)[0]
            file_node_names = [stem] if separator is None else stem.split(separator)
            cur_node = tree.gns(file_node_names)
            DataWriter.attach_file_reference(cur_node, filepath_attribute=filepath_attribute, root=folder_path, file=file_name)

            processed += 1
            if commit_every and processed % commit_every == 0:
                self.commit()

        if processed:
            self.commit()

        if created_connection:
            self.close_connection()

        return processed


def load_small_files_directory(
    tree: BaseNode,
    dir_path: str,
    parameter_separator: Optional[str] = None,
    parameter_seperator: Optional[str] = None,
    filepath_attribute: str = "_pfo_audio_wav",
    file_extension: str = "wav",
) -> int:
    separator = _resolve_parameter_separator(parameter_separator, parameter_seperator)
    root_folder = os.path.basename(os.path.normpath(dir_path))
    parent_dir = os.path.dirname(os.path.normpath(dir_path))
    processed = 0

    for root, _, files in os.walk(dir_path):
        for file_name in files:
            if not file_name.endswith(f".{file_extension}"):
                continue

            relative_root = os.path.relpath(root, start=parent_dir)
            path_parts = relative_root.split(os.path.sep) if relative_root != "." else [root_folder]
            last_node = tree.gns(path_parts)

            stem = os.path.splitext(file_name)[0]
            cur_node = last_node.gn(stem) if separator is None else last_node.gns(stem.split(separator))
            DataWriter.attach_file_reference(cur_node, filepath_attribute=filepath_attribute, root=root, file=file_name)
            processed += 1

    return processed


def load_small_files_folder(
    tree: BaseNode,
    folder_path: str,
    parameter_separator: Optional[str] = "_~_",
    parameter_seperator: Optional[str] = None,
    filepath_attribute: str = "_pfo_audio_wav",
    file_extension: str = "wav",
) -> int:
    separator = _resolve_parameter_separator(parameter_separator, parameter_seperator)
    processed = 0

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(f".{file_extension}"):
            continue

        stem = os.path.splitext(file_name)[0]
        file_node_names = [stem] if separator is None else stem.split(separator)
        cur_node = tree.gns(file_node_names)
        DataWriter.attach_file_reference(cur_node, filepath_attribute=filepath_attribute, root=folder_path, file=file_name)
        processed += 1

    return processed
