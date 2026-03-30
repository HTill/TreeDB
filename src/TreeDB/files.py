from __future__ import annotations

import os
from typing import Any, List, Optional

from numpy import ndarray, save

from .paths import PathFileObj
from .tree import StorageTree


def _resolve_parameter_separator(parameter_separator: Optional[str], parameter_seperator: Optional[str]) -> Optional[str]:
    if parameter_separator is None and parameter_seperator is not None:
        return parameter_seperator
    return parameter_separator


def add_file(
    node: StorageTree,
    filepath_attribute: str = "_pfo_audio_wav",
    root: str = "",
    file: str = "",
    filepath: str = "",
) -> PathFileObj:
    pfo = PathFileObj(root=root, file=file, filepath=filepath)
    node.set_attribute(filepath_attribute, pfo)
    return pfo


def rename_file(
    node: StorageTree,
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


class FileReader:
    @staticmethod
    def load_audio(filepath: str):
        from scipy.io import wavfile
        from numpy import reshape

        samplerate, audio = wavfile.read(filepath)

        if len(audio.shape) == 1:
            audio = reshape(audio, (audio.shape[0], -1))

        return audio, samplerate

    @staticmethod
    def read_audio_wav(node: StorageTree, filepath_attribute: str = "_pfo_audio_wav"):
        return FileReader.load_audio(node.ga(filepath_attribute).filepath)

    @staticmethod
    def read_table_txt(node: StorageTree, filepath_attribute: str = "_pfo_table_txt", header: Any = "infer"):
        import pandas as pd

        filepath = node.ga(filepath_attribute).filepath
        return pd.read_table(filepath, header=header)

    @staticmethod
    def read_array_npy(node: StorageTree, filepath_attribute: str = "_pfo_array_npy"):
        from numpy import load

        filepath = node.ga(filepath_attribute).filepath
        if not filepath.endswith(".npy"):
            filepath = filepath + ".npy"

        return load(filepath)

    @staticmethod
    def read_array_mat(node: StorageTree, filepath_attribute: str = "_pfo_array_mat"):
        from scipy.io import loadmat

        filepath = node.ga(filepath_attribute).filepath
        return loadmat(filepath)


class FileWriter:
    @staticmethod
    def setup_tree_directories(root: str, node: StorageTree, pre_parents: Optional[List[str]] = None) -> str:
        node_parents = list(pre_parents or [])
        node_parents.extend(node.gps())

        new_root = root
        for parent in node_parents:
            path = os.path.join(new_root, parent)
            os.makedirs(path, exist_ok=True)
            new_root = path

        return new_root

    @staticmethod
    def create_filename(
        pre_parents: Optional[List[str]],
        node: StorageTree,
        file_type: str,
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> str:
        separator = _resolve_parameter_separator(parameter_separator, parameter_seperator) or "_~_"
        node_parents = list(pre_parents or [])
        node_parents.extend(node.gps())
        return separator.join(node_parents) + "." + file_type

    @staticmethod
    def write_audio_wav_into_tree_directories(
        root: str,
        node: StorageTree,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_audio_wav",
    ) -> None:
        root = FileWriter.setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        FileWriter.write_audio_wav(
            root=root,
            node=node,
            audio=audio,
            samplerate=samplerate,
            filename=filename,
            pre_parents=pre_parents,
            filepath_attribute=filepath_attribute,
        )

    @staticmethod
    def write_audio_raw_into_tree_directories(
        root: str,
        node: StorageTree,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_audio_raw",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> None:
        root = FileWriter.setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        FileWriter.write_audio_raw(
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
    def write_array_numpy_into_tree_directories(
        root: str,
        node: StorageTree,
        array: ndarray,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_array_npy",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> None:
        root = FileWriter.setup_tree_directories(root=root, node=node, pre_parents=pre_parents)
        FileWriter.write_array_npy(
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
    def write_audio_wav(
        root: str,
        node: StorageTree,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_audio_wav",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        from scipy.io.wavfile import write

        if filename is None:
            filename = FileWriter.create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="wav",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        write(filepath, samplerate, audio)
        add_file(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1

    @staticmethod
    def write_audio_raw(
        root: str,
        node: StorageTree,
        audio: ndarray,
        samplerate: int,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_audio_raw",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = FileWriter.create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="raw",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        with open(filepath, mode="wb") as file_handle:
            file_handle.write(audio.tobytes())

        add_file(node, filepath_attribute=filepath_attribute, filepath=filepath)
        node.set_attribute("_audio_raw_dtype", str(audio.dtype))
        node.set_attribute("_audio_raw_samplerate", samplerate)
        return 1

    @staticmethod
    def write_array_npy(
        root: str,
        node: StorageTree,
        array: ndarray,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_array_npy",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = FileWriter.create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="npy",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        save(filepath, array)
        add_file(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1

    @staticmethod
    def write_table_txt(
        root: str,
        node: StorageTree,
        dataframe,
        header: bool = True,
        index: bool = False,
        filename: Optional[str] = None,
        pre_parents: Optional[List[str]] = None,
        filepath_attribute: str = "_pfo_table_txt",
        parameter_separator: Optional[str] = None,
        parameter_seperator: Optional[str] = None,
    ) -> int:
        if filename is None:
            filename = FileWriter.create_filename(
                pre_parents=pre_parents,
                node=node,
                file_type="txt",
                parameter_separator=parameter_separator,
                parameter_seperator=parameter_seperator,
            )

        filepath = os.path.join(root, filename)
        dataframe.to_csv(filepath, header=header, index=index)
        add_file(node, filepath_attribute=filepath_attribute, filepath=filepath)
        return 1
