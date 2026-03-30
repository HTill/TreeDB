from __future__ import annotations

import shutil
from typing import Any, Mapping, Optional

import persistent
from BTrees.OOBTree import OOBTree
from ZODB.blob import Blob
from numpy import ndarray


class BlobObject(persistent.Persistent):
    def __init__(
        self,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.blob = Blob()
        self.metadata = OOBTree()
        self.metadata["content_type"] = content_type

        for key, value in (metadata or {}).items():
            self.metadata[str(key)] = value

    @property
    def content_type(self) -> str:
        return str(self.metadata.get("content_type", "application/octet-stream"))

    def set_metadata(self, key: str, value: Any) -> Any:
        self.metadata[str(key)] = value
        return value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(str(key), default)

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata.items())

    def write_bytes(self, data: bytes) -> None:
        with self.blob.open("w") as file_handle:
            file_handle.write(data)

    def read_bytes(self) -> bytes:
        with self.blob.open("r") as file_handle:
            return file_handle.read()

    def write_array(self, array: ndarray, allow_pickle: bool = False) -> None:
        from numpy import save

        with self.blob.open("w") as file_handle:
            save(file_handle, array, allow_pickle=allow_pickle)

        self.metadata["format"] = "npy"
        self.metadata["dtype"] = str(array.dtype)
        self.metadata["shape"] = tuple(array.shape)

    def read_array(self, allow_pickle: bool = False):
        from numpy import load

        with self.blob.open("r") as file_handle:
            return load(file_handle, allow_pickle=allow_pickle)

    def copy_for_tree(self) -> "BlobObject":
        clone = BlobObject(content_type=self.content_type, metadata=self.metadata_dict())
        with self.blob.open("r") as source_handle, clone.blob.open("w") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
        return clone


def add_blob(
    node,
    blob_attribute: str = "_blob",
    data: bytes = b"",
    content_type: str = "application/octet-stream",
    metadata: Optional[Mapping[str, Any]] = None,
) -> BlobObject:
    blob_object = BlobObject(content_type=content_type, metadata=metadata)
    blob_object.write_bytes(data)
    node.set_attribute(blob_attribute, blob_object)
    return blob_object


class BlobWriter:
    @staticmethod
    def write_bytes(
        node,
        data: bytes,
        blob_attribute: str = "_blob",
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> BlobObject:
        blob_object = BlobObject(content_type=content_type, metadata=metadata)
        blob_object.write_bytes(data)
        node.set_attribute(blob_attribute, blob_object)
        return blob_object

    @staticmethod
    def write_array(
        node,
        array: ndarray,
        blob_attribute: str = "_blob_array",
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: str = "application/x-npy",
        allow_pickle: bool = False,
    ) -> BlobObject:
        blob_object = BlobObject(content_type=content_type, metadata=metadata)
        blob_object.write_array(array, allow_pickle=allow_pickle)
        node.set_attribute(blob_attribute, blob_object)
        return blob_object


class BlobReader:
    @staticmethod
    def read_bytes(node, blob_attribute: str = "_blob") -> bytes:
        return node.get_attribute(blob_attribute).read_bytes()

    @staticmethod
    def read_array(node, blob_attribute: str = "_blob_array", allow_pickle: bool = False):
        return node.get_attribute(blob_attribute).read_array(allow_pickle=allow_pickle)
