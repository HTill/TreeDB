from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import ZODB
import ZODB.FileStorage
import transaction

from .files import add_file
from .paths import PathFileObj
from .tree import StorageTree, StorageTreeUnconnected


def _resolve_parameter_separator(parameter_separator: Optional[str], parameter_seperator: Optional[str]) -> Optional[str]:
    if parameter_separator is None and parameter_seperator is not None:
        return parameter_seperator
    return parameter_separator


class StorageTreeConnection:
    def __init__(
        self,
        database: "StorageTreeDatabase",
        connection_id: str = "standard",
        unconnected_tree: Optional[StorageTreeUnconnected] = None,
    ) -> None:
        self.database = database
        self.connection_id = connection_id
        self.unconnected_tree = unconnected_tree

    def __enter__(self) -> StorageTree:
        return self.database.open_connection(
            unconnected_tree=self.unconnected_tree,
            connection_id=self.connection_id,
        )

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.database.abort(self.connection_id)
        self.database.close_connection(self.connection_id)
        return False


class StorageTreeDatabase:
    def __init__(
        self,
        path: str = "",
        name: str = "StorageTreeDB",
        new: bool = False,
        read_only: bool = False,
    ) -> None:
        self.path = path
        self.name = name
        self.read_only = read_only
        self.db = self.setup_storage_tree_db(new=new, read_only=read_only)
        self.conn_dict: Dict[str, Tuple[Any, Any]] = {}

    def _storage_filepath(self) -> str:
        if self.path:
            return PathFileObj(root=self.path, file=self.name).filepath
        return self.name

    def setup_storage_tree_db(self, new: bool = False, read_only: bool = False):
        if self.path:
            os.makedirs(self.path, exist_ok=True)

        storage = ZODB.FileStorage.FileStorage(
            self._storage_filepath(),
            create=new,
            read_only=read_only,
        )
        db = ZODB.DB(storage)

        transaction_manager = transaction.TransactionManager()
        conn = db.open(transaction_manager=transaction_manager)
        if not hasattr(conn.root, "stt"):
            conn.root.stt = StorageTree(key=self.name)
            transaction_manager.commit()
        conn.close()
        return db

    def setup_StorageTree_DB(self, new: bool = False, read_only: bool = False):
        return self.setup_storage_tree_db(new=new, read_only=read_only)

    def open_connection(
        self,
        unconnected_tree: Optional[StorageTreeUnconnected] = None,
        connection_id: str = "standard",
    ) -> StorageTree:
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
        unconnected_tree: Optional[StorageTreeUnconnected] = None,
    ) -> StorageTreeConnection:
        return StorageTreeConnection(self, connection_id=connection_id, unconnected_tree=unconnected_tree)

    def close_connection(self, connection_id: str = "standard") -> None:
        if connection_id not in self.conn_dict:
            return

        conn, transaction_manager = self.conn_dict[connection_id]
        transaction_manager.abort()
        conn.close()
        del self.conn_dict[connection_id]

    def commit(self, connection_id: str = "standard") -> None:
        if connection_id not in self.conn_dict:
            return

        _, transaction_manager = self.conn_dict[connection_id]
        transaction_manager.commit()

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
                add_file(cur_node, filepath_attribute=filepath_attribute, root=root, file=file_name)

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
            add_file(cur_node, filepath_attribute=filepath_attribute, root=folder_path, file=file_name)

            processed += 1
            if commit_every and processed % commit_every == 0:
                self.commit()

        if processed:
            self.commit()

        if created_connection:
            self.close_connection()

        return processed


def load_small_files_directory(
    tree: StorageTree,
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
            add_file(cur_node, filepath_attribute=filepath_attribute, root=root, file=file_name)
            processed += 1

    return processed


def load_small_files_folder(
    tree: StorageTree,
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
        add_file(cur_node, filepath_attribute=filepath_attribute, root=folder_path, file=file_name)
        processed += 1

    return processed
