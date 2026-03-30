from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, cast

import persistent
from BTrees.OOBTree import OOBTree
from numpy import asarray, ndarray, save

from .blobs import BlobObject
from .paths import PathFileObj


def _copy_mapping(mapping: Mapping[str, Any]) -> OOBTree:
    copied = OOBTree()
    for key, value in mapping.items():
        copied[str(key)] = value
    return copied


def _as_path_file_obj(value: str | PathFileObj) -> PathFileObj:
    if isinstance(value, PathFileObj):
        return value.copy_for_tree()
    return PathFileObj(filepath=str(value))


def _zarr_dataset_path(value: PathFileObj) -> str:
    return value.as_linux_path().lstrip("/")


def _store_source_attributes(node, filepath: Path, source_format: str, samplerate_hz: float | None = None) -> None:
    node.set_attribute("_source_filepath", PathFileObj(filepath=str(filepath)))
    node.set_attribute("_source_format", source_format)
    node.set_attribute("_source_filename", filepath.name)
    if samplerate_hz is not None:
        node.set_attribute("_source_samplerate_hz", samplerate_hz)


def _node_data_path(node, data_attribute: str) -> str:
    node_path = list(node.gps())
    if not node_path:
        node_path = [node.key]
    node_path.append(str(data_attribute))
    return "/".join(node_path)


class DataObject(persistent.Persistent):
    def __init__(
        self,
        backend: str,
        samplerate_hz: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: str = "application/x-npy",
        time_axis: int = 0,
    ) -> None:
        if samplerate_hz is not None and samplerate_hz <= 0:
            raise ValueError("samplerate_hz must be greater than 0.")
        if time_axis != 0:
            raise NotImplementedError("Only time_axis=0 is currently supported.")

        self.backend = str(backend)
        self.samplerate_hz = None if samplerate_hz is None else float(samplerate_hz)
        self.time_axis = int(time_axis)
        self.content_type = str(content_type)
        self.metadata = _copy_mapping(metadata or {})
        self.blob_object: BlobObject | None = None
        self.store_path: PathFileObj | None = None
        self.dataset_path: PathFileObj | None = None

    def set_metadata(self, key: str, value: Any) -> Any:
        self.metadata[str(key)] = value
        return value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(str(key), default)

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata.items())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.get_metadata("shape", ()))

    @property
    def dtype(self) -> str | None:
        dtype = self.get_metadata("dtype")
        return None if dtype is None else str(dtype)

    @property
    def sample_count(self) -> int:
        shape = self.shape
        if not shape:
            return 0
        return int(shape[0])

    def _set_array_metadata(self, array: ndarray) -> None:
        self.set_metadata("shape", tuple(array.shape))
        self.set_metadata("dtype", str(array.dtype))
        self.set_metadata("backend", self.backend)
        self.set_metadata("samplerate_hz", self.samplerate_hz)
        self.set_metadata("time_axis", self.time_axis)
        self.set_metadata("payload_kind", "array")

    def write_blob_bytes(self, data: bytes, payload_kind: str = "bytes") -> None:
        blob_object = BlobObject(content_type=self.content_type)
        blob_object.write_bytes(data)
        self.blob_object = blob_object
        self.set_metadata("backend", self.backend)
        self.set_metadata("payload_kind", payload_kind)
        self.set_metadata("byte_length", len(data))

    def read_bytes(self) -> bytes:
        if self.backend != "blob" or self.blob_object is None:
            raise ValueError("read_bytes is only supported for blob-backed byte/text/table payloads.")
        return self.blob_object.read_bytes()

    def read_text(self, encoding: Optional[str] = None) -> str:
        used_encoding = encoding or cast(str, self.get_metadata("text_encoding", "utf-8"))
        return self.read_bytes().decode(used_encoding)

    def read_table(self):
        import pandas as pd

        csv_buffer = io.StringIO(self.read_text())
        return pd.read_csv(csv_buffer)

    def read(self):
        payload_kind = self.get_metadata("payload_kind", "array")

        if payload_kind == "array":
            return self.read_interval()
        if payload_kind == "bytes":
            return self.read_bytes()
        if payload_kind == "text":
            return self.read_text()
        if payload_kind == "table":
            return self.read_table()

        raise ValueError(f"Unsupported payload kind: {payload_kind!r}")

    def write_blob_array(self, array: ndarray, allow_pickle: bool = False) -> None:
        blob_object = BlobObject(content_type=self.content_type)
        blob_object.write_array(array, allow_pickle=allow_pickle)
        self.blob_object = blob_object
        self._set_array_metadata(array)

    def write_zarr_array(
        self,
        array: ndarray,
        store_path: str | PathFileObj,
        dataset_path: str | PathFileObj,
        chunks: Optional[tuple[int, ...]] = None,
    ) -> None:
        import zarr

        store_path_obj = _as_path_file_obj(store_path)
        dataset_path_obj = _as_path_file_obj(dataset_path)

        zarr_array = cast(Any, zarr.open(
            store=store_path_obj.filepath,
            mode="w",
            path=_zarr_dataset_path(dataset_path_obj),
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks,
        ))
        zarr_array[:] = array

        self.store_path = store_path_obj
        self.dataset_path = dataset_path_obj
        self.set_metadata("zarr_store_path", self.store_path)
        self.set_metadata("zarr_dataset_path", self.dataset_path)
        self.set_metadata("zarr_chunks", tuple(chunks) if chunks is not None else tuple(zarr_array.chunks))
        self._set_array_metadata(array)

    def _seconds_to_index(self, value: float, *, is_stop: bool) -> int:
        if self.samplerate_hz is None:
            raise ValueError("samplerate_hz is required for second-based reads.")
        scaled = float(value) * self.samplerate_hz
        return int(math.ceil(scaled) if is_stop else math.floor(scaled))

    def _normalize_range(self, start: Any = None, stop: Any = None, unit: str = "samples") -> tuple[int, int]:
        if unit not in {"samples", "seconds"}:
            raise ValueError("unit must be 'samples' or 'seconds'.")

        if start is None:
            start_idx = 0
        elif unit == "seconds":
            start_idx = self._seconds_to_index(float(start), is_stop=False)
        else:
            start_idx = int(start)

        if stop is None:
            stop_idx = self.sample_count
        elif unit == "seconds":
            stop_idx = self._seconds_to_index(float(stop), is_stop=True)
        else:
            stop_idx = int(stop)

        start_idx = max(0, start_idx)
        stop_idx = min(self.sample_count, stop_idx)
        if stop_idx < start_idx:
            raise ValueError("stop must be greater than or equal to start.")

        return start_idx, stop_idx

    def read_interval(self, start: Any = None, stop: Any = None, unit: str = "samples"):
        payload_kind = self.get_metadata("payload_kind", "array")
        if payload_kind != "array":
            raise ValueError("Interval reads are only supported for array payloads.")

        start_idx, stop_idx = self._normalize_range(start=start, stop=stop, unit=unit)

        if self.backend == "blob":
            if self.blob_object is None:
                raise ValueError("Blob backend measurement is missing its blob payload.")
            array = self.blob_object.read_array(allow_pickle=False)
            return array[start_idx:stop_idx]

        if self.backend == "zarr":
            if self.store_path is None or self.dataset_path is None:
                raise ValueError("Zarr backend measurement is missing store metadata.")
            import zarr

            zarr_array = cast(Any, zarr.open(store=self.store_path.filepath, mode="r", path=_zarr_dataset_path(self.dataset_path)))
            return asarray(zarr_array[start_idx:stop_idx])

        raise ValueError(f"Unsupported measurement backend: {self.backend!r}")

    def read_seconds(self, start_second: float | None = None, stop_second: float | None = None):
        return self.read_interval(start=start_second, stop=stop_second, unit="seconds")

    def iter_blocks(
        self,
        block_size: float | int,
        block_unit: str = "samples",
        start: Any = None,
        stop: Any = None,
        range_unit: str = "samples",
    ) -> Iterator[ndarray]:
        if block_unit == "seconds":
            step = self._seconds_to_index(float(block_size), is_stop=True)
        elif block_unit == "samples":
            step = int(block_size)
        else:
            raise ValueError("block_unit must be 'samples' or 'seconds'.")

        if step <= 0:
            raise ValueError("block_size must be greater than 0.")

        start_idx, stop_idx = self._normalize_range(start=start, stop=stop, unit=range_unit)
        for block_start in range(start_idx, stop_idx, step):
            block_stop = min(block_start + step, stop_idx)
            yield cast(ndarray, self.read_interval(block_start, block_stop, unit="samples"))

    def copy_for_tree(self) -> "DataObject":
        clone = DataObject(
            backend=self.backend,
            samplerate_hz=self.samplerate_hz,
            metadata=self.metadata_dict(),
            content_type=self.content_type,
            time_axis=self.time_axis,
        )

        if self.backend == "blob" and self.blob_object is not None:
            clone.blob_object = self.blob_object.copy_for_tree()
        else:
            clone.store_path = None if self.store_path is None else self.store_path.copy_for_tree()
            clone.dataset_path = None if self.dataset_path is None else self.dataset_path.copy_for_tree()

        return clone


class DataWriter:
    @staticmethod
    def attach_file_reference(
        node,
        filepath_attribute: str = "_pfo_audio_wav",
        root: str = "",
        file: str = "",
        filepath: str = "",
    ) -> PathFileObj:
        pfo = PathFileObj(root=root, file=file, filepath=filepath)
        node.set_attribute(filepath_attribute, pfo)
        return pfo

    @staticmethod
    def attach_file(
        node,
        filepath: str,
        data_attribute: str = "data",
        backend: str = "blob",
        samplerate_hz: float | None = None,
        database=None,
        format: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        zarr_store_path: str | None = None,
        chunks: Optional[tuple[int, ...]] = None,
        text_encoding: str = "utf-8",
    ) -> DataObject:
        return DataWriter.ingest_file(
            node=node,
            filepath=filepath,
            data_attribute=data_attribute,
            backend=backend,
            samplerate_hz=samplerate_hz,
            database=database,
            format=format,
            metadata=metadata,
            zarr_store_path=zarr_store_path,
            chunks=chunks,
            text_encoding=text_encoding,
        )

    @staticmethod
    def rename_file(
        node,
        filepath_attribute: str = "_pfo_audio_wav",
        new_root: str = "",
        new_file: str = "",
        new_filepath: str = "",
    ) -> None:
        pfo = node.ga(filepath_attribute)
        old_filepath = pfo.filepath

        if new_filepath:
            pfo.filepath = new_filepath
        else:
            if new_root:
                pfo.root = new_root
            if new_file:
                pfo.file = new_file

        os.replace(old_filepath, pfo.filepath)

    @staticmethod
    def _resolve_parameter_separator(parameter_separator: Optional[str], parameter_seperator: Optional[str]) -> Optional[str]:
        if parameter_separator is None and parameter_seperator is not None:
            return parameter_seperator
        return parameter_separator

    @staticmethod
    def _setup_tree_directories(root: str, node, pre_parents: Optional[list[str]] = None) -> str:
        node_parents = list(pre_parents or [])
        node_parents.extend(node.gps())

        new_root = root
        for parent in node_parents:
            path = os.path.join(new_root, parent)
            os.makedirs(path, exist_ok=True)
            new_root = path

        return new_root

    @staticmethod
    def _create_filename(
        pre_parents: Optional[list[str]],
        node,
        file_type: str,
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> str:
        separator = DataWriter._resolve_parameter_separator(parameter_separator, parameter_seperator) or "_~_"
        node_parents = list(pre_parents or [])
        node_parents.extend(node.gps())
        return separator.join(node_parents) + "." + file_type

    @staticmethod
    def write_array(
        node,
        array: ndarray,
        samplerate_hz: float | None = None,
        data_attribute: str = "data",
        backend: str = "blob",
        metadata: Optional[Mapping[str, Any]] = None,
        database=None,
        zarr_store_path: str | PathFileObj | None = None,
        dataset_path: str | PathFileObj | None = None,
        chunks: Optional[tuple[int, ...]] = None,
        content_type: str = "application/x-npy",
    ) -> DataObject:
        array = asarray(array)
        if array.ndim == 0:
            raise ValueError("Measurement arrays must have at least one dimension.")

        data = DataObject(
            backend=backend,
            samplerate_hz=samplerate_hz,
            metadata=metadata,
            content_type=content_type,
        )

        if backend == "blob":
            data.write_blob_array(array, allow_pickle=False)
        elif backend == "zarr":
            if zarr_store_path is None:
                if database is None:
                    raise ValueError("database or zarr_store_path is required for the zarr backend.")
                zarr_store_path = database.data_store_path()
            zarr_store_path = cast(str | PathFileObj, zarr_store_path)

            data.write_zarr_array(
                array,
                store_path=zarr_store_path,
                dataset_path=dataset_path or PathFileObj(filepath=_node_data_path(node, data_attribute)),
                chunks=chunks,
            )
        else:
            raise ValueError("backend must be 'blob' or 'zarr'.")

        node.set_attribute(data_attribute, data)
        return data

    @staticmethod
    def write_bytes(
        node,
        data: bytes,
        data_attribute: str = "data",
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: str = "application/octet-stream",
    ) -> DataObject:
        data_object = DataObject(backend="blob", metadata=metadata, content_type=content_type)
        data_object.write_blob_bytes(data, payload_kind="bytes")
        node.set_attribute(data_attribute, data_object)
        return data_object

    @staticmethod
    def write_text(
        node,
        text: str,
        data_attribute: str = "data",
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: str = "text/plain",
        encoding: str = "utf-8",
    ) -> DataObject:
        merged_metadata = dict(metadata or {})
        merged_metadata["text_encoding"] = encoding
        data_object = DataObject(backend="blob", metadata=merged_metadata, content_type=content_type)
        data_object.write_blob_bytes(text.encode(encoding), payload_kind="text")
        node.set_attribute(data_attribute, data_object)
        return data_object

    @staticmethod
    def write_table(
        node,
        dataframe,
        data_attribute: str = "data",
        metadata: Optional[Mapping[str, Any]] = None,
        encoding: str = "utf-8",
    ) -> DataObject:
        merged_metadata = dict(metadata or {})
        merged_metadata["text_encoding"] = encoding
        merged_metadata["columns"] = list(dataframe.columns)
        csv_text = dataframe.to_csv(index=False)
        data_object = DataObject(backend="blob", metadata=merged_metadata, content_type="text/csv")
        data_object.write_blob_bytes(csv_text.encode(encoding), payload_kind="table")
        node.set_attribute(data_attribute, data_object)
        return data_object

    @staticmethod
    def ingest_file(
        node,
        filepath: str,
        data_attribute: str = "data",
        backend: str = "blob",
        samplerate_hz: float | None = None,
        database=None,
        format: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        zarr_store_path: str | PathFileObj | None = None,
        chunks: Optional[tuple[int, ...]] = None,
        text_encoding: str = "utf-8",
    ) -> DataObject:
        source_path = Path(filepath)
        source_format = (format or source_path.suffix.lstrip(".") or "raw").lower()
        merged_metadata = dict(metadata or {})
        if source_format == "wav":
            from scipy.io import wavfile
            from numpy import reshape

            source_samplerate, audio = wavfile.read(source_path)
            if len(audio.shape) == 1:
                audio = reshape(audio, (audio.shape[0], -1))
            _store_source_attributes(node, source_path, source_format, float(source_samplerate))
            return DataWriter.write_array(
                node,
                asarray(audio),
                samplerate_hz=samplerate_hz or float(source_samplerate),
                data_attribute=data_attribute,
                backend=backend,
                metadata=merged_metadata,
                database=database,
                zarr_store_path=zarr_store_path,
                chunks=chunks,
                content_type="audio/wav",
            )

        if source_format == "npy":
            from numpy import load

            array = load(source_path, allow_pickle=False)
            _store_source_attributes(node, source_path, source_format)
            return DataWriter.write_array(
                node,
                asarray(array),
                samplerate_hz=samplerate_hz,
                data_attribute=data_attribute,
                backend=backend,
                metadata=merged_metadata,
                database=database,
                zarr_store_path=zarr_store_path,
                chunks=chunks,
                content_type="application/x-npy",
            )

        if source_format in {"png", "jpg", "jpeg"}:
            from PIL import Image

            image = Image.open(source_path)
            array = asarray(image)
            merged_metadata["image_mode"] = image.mode
            _store_source_attributes(node, source_path, source_format)
            return DataWriter.write_array(
                node,
                array,
                samplerate_hz=samplerate_hz,
                data_attribute=data_attribute,
                backend=backend,
                metadata=merged_metadata,
                database=database,
                zarr_store_path=zarr_store_path,
                chunks=chunks,
                content_type=f"image/{'jpeg' if source_format in {'jpg', 'jpeg'} else source_format}",
            )

        if source_format == "csv":
            import pandas as pd

            dataframe = pd.read_csv(source_path)
            _store_source_attributes(node, source_path, source_format)
            return DataWriter.write_table(node, dataframe, data_attribute=data_attribute, metadata=merged_metadata, encoding=text_encoding)

        if source_format == "txt":
            text = source_path.read_text(encoding=text_encoding)
            _store_source_attributes(node, source_path, source_format)
            return DataWriter.write_text(node, text, data_attribute=data_attribute, metadata=merged_metadata, encoding=text_encoding)

        raw_bytes = source_path.read_bytes()
        _store_source_attributes(node, source_path, source_format)
        return DataWriter.write_bytes(node, raw_bytes, data_attribute=data_attribute, metadata=merged_metadata)

    @staticmethod
    def write_audio_wav(
        root: str,
        node,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_audio_wav",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        from scipy.io.wavfile import write

        if filename is None:
            filename = DataWriter._create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="wav",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        write(filepath, samplerate, audio)
        DataWriter.attach_file_reference(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1

    @staticmethod
    def write_audio_wav_into_tree_directories(
        root: str,
        node,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_audio_wav",
    ) -> None:
        root = DataWriter._setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        DataWriter.write_audio_wav(
            root=root,
            node=node,
            audio=audio,
            samplerate=samplerate,
            filename=filename,
            pre_parents=pre_parents,
            filepath_attribute=filepath_attribute,
        )

    @staticmethod
    def write_audio_raw(
        root: str,
        node,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_audio_raw",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = DataWriter._create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="raw",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        with open(filepath, mode="wb") as file_handle:
            file_handle.write(audio.tobytes())

        DataWriter.attach_file_reference(node, filepath_attribute=filepath_attribute, filepath=filepath)
        node.set_attribute("_audio_raw_dtype", str(audio.dtype))
        node.set_attribute("_audio_raw_samplerate", samplerate)
        return 1

    @staticmethod
    def write_audio_raw_into_tree_directories(
        root: str,
        node,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_audio_raw",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> None:
        root = DataWriter._setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        DataWriter.write_audio_raw(
            root=root,
            node=node,
            audio=audio,
            samplerate=samplerate,
            filename=filename,
            pre_parents=pre_parents,
            filepath_attribute=filepath_attribute,
            parameter_separator=parameter_separator,
            parameter_seperator=parameter_seperator,
        )

    @staticmethod
    def write_array_npy(
        root: str,
        node,
        array: ndarray,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_array_npy",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = DataWriter._create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="npy",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        save(filepath, array)
        DataWriter.attach_file_reference(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1

    @staticmethod
    def write_array_npy_into_tree_directories(
        root: str,
        node,
        array: ndarray,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_array_npy",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> None:
        root = DataWriter._setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        DataWriter.write_array_npy(
            root=root,
            node=node,
            array=array,
            filename=filename,
            pre_parents=pre_parents,
            filepath_attribute=filepath_attribute,
            parameter_separator=parameter_separator,
            parameter_seperator=parameter_seperator,
        )

    @staticmethod
    def write_table_txt(
        root: str,
        node,
        dataframe,
        header: bool = True,
        index: bool = False,
        filename: Optional[str] = None,
        pre_parents: Optional[list[str]] = None,
        filepath_attribute: str = "_pfo_table_txt",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = DataWriter._create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="txt",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        dataframe.to_csv(filepath, header=header, index=index)
        DataWriter.attach_file_reference(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1


class DataReader:
    @staticmethod
    def read(node, data_attribute: str = "data"):
        return DataReader.open(node, data_attribute).read()

    @staticmethod
    def open(node, data_attribute: str = "data") -> DataObject:
        data = node.get_attribute(data_attribute)
        if data is None:
            raise KeyError(f"Data attribute {data_attribute!r} not found.")
        return data

    @staticmethod
    def read_interval(node, data_attribute: str = "data", start: Any = None, stop: Any = None, unit: str = "samples"):
        return DataReader.open(node, data_attribute).read_interval(start=start, stop=stop, unit=unit)

    @staticmethod
    def read_seconds(node, data_attribute: str = "data", start_second: float | None = None, stop_second: float | None = None):
        return DataReader.open(node, data_attribute).read_seconds(start_second=start_second, stop_second=stop_second)

    @staticmethod
    def iter_blocks(
        node,
        data_attribute: str = "data",
        block_size: float | int = 1,
        block_unit: str = "samples",
        start: Any = None,
        stop: Any = None,
        range_unit: str = "samples",
    ) -> Iterable[ndarray]:
        return DataReader.open(node, data_attribute).iter_blocks(
            block_size=block_size,
            block_unit=block_unit,
            start=start,
            stop=stop,
            range_unit=range_unit,
        )
