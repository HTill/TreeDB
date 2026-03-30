from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import ZODB
import ZODB.FileStorage
import transaction

from .paths import PathFileObj
from .tree import BaseNode, BaseNodeUnconnected


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
    ) -> None:
        self.path = path
        self.name = name
        self.read_only = read_only
        self.cache_size = cache_size
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
        db_kwargs: dict[str, Any] = {}
        if self.cache_size is not None:
            db_kwargs["cache_size"] = self.cache_size
        db = ZODB.DB(storage, **db_kwargs)

        transaction_manager = transaction.TransactionManager()
        conn = db.open(transaction_manager=transaction_manager)
        if not hasattr(conn.root, "stt") and read_only:
            conn.close()
            db.close()
            raise FileNotFoundError("Cannot open LOTDB in read-only mode before it has been initialized.")

        if not hasattr(conn.root, "stt"):
            conn.root.stt = BaseNode(key=self.name)
            transaction_manager.commit()
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
