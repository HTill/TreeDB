from __future__ import annotations

import os
import platform
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, List, Optional, Sequence, Union

import persistent
import persistent.list


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:$")
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:(?:[\\/].*)?$")


def _is_windows_path(value: str) -> bool:
    return bool(_WINDOWS_PATH_RE.match(value)) or value.startswith("\\\\")


def _iterable_to_parts(root: Iterable[str]) -> List[str]:
    return [str(part) for part in root]


class PathFileObj(persistent.Persistent):
    def __init__(self, root: Union[str, Iterable[str]] = "", file: str = "", filepath: str = "") -> None:
        self._path_style = "posix"
        self._drive = ""
        self._is_absolute = False
        self.root_path_list = persistent.list.PersistentList()
        self._file = ""

        if filepath:
            self.filepath = filepath
        else:
            self.root = root
            self.file = file

    @property
    def root(self) -> str:
        return self._build_root_for_current_platform()

    @root.setter
    def root(self, root: Union[str, Iterable[str]]) -> None:
        if isinstance(root, str):
            self._set_root_from_string(root)
        else:
            self._set_root_from_parts(_iterable_to_parts(root))

    @property
    def file(self) -> str:
        return self._file

    @file.setter
    def file(self, file: str) -> None:
        self._file = str(file)

    @property
    def filepath(self) -> str:
        return self.to_platform_path()

    @filepath.setter
    def filepath(self, filepath: str) -> None:
        self._set_filepath_from_string(filepath)

    @property
    def path(self) -> Path:
        return Path(self.filepath)

    def as_path(self) -> Path:
        return self.path

    def as_linux_path(self) -> str:
        return self.to_platform_path(system="Linux")

    def as_windows_path(self) -> str:
        return self.to_platform_path(system="Windows")

    def to_platform_path(self, system: Optional[str] = None, release: Optional[str] = None) -> str:
        if system is None:
            platform_info = platform.uname()
            system = platform_info.system
            release = platform_info.release

        if system == "Windows":
            return self._build_windows_path(include_file=True)

        if system == "Linux" and release and "Microsoft" in release:
            return self._build_linux_path(include_file=True)

        return self._build_linux_path(include_file=True)

    def _set_filepath_from_string(self, filepath: str) -> None:
        if _is_windows_path(filepath):
            pure_path = PureWindowsPath(filepath)
            self._path_style = "windows"
            self._drive = pure_path.drive
            self._is_absolute = bool(pure_path.root)
            self.root_path_list = persistent.list.PersistentList(list(pure_path.parent.parts[1:]))
            self.file = pure_path.name
            return

        pure_posix = PurePosixPath(filepath)
        parts = list(pure_posix.parts)

        if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
            self._path_style = "windows"
            self._drive = parts[2].upper() + ":"
            self._is_absolute = True
            self.root_path_list = persistent.list.PersistentList(parts[3:-1])
            self.file = pure_posix.name
            return

        self._path_style = "posix"
        self._drive = ""
        self._is_absolute = pure_posix.is_absolute()
        if parts and parts[0] == "/":
            parts = parts[1:]
        self.root_path_list = persistent.list.PersistentList(parts[:-1])
        self.file = pure_posix.name

    def _set_root_from_string(self, root: str) -> None:
        if not root:
            self._path_style = "posix"
            self._drive = ""
            self._is_absolute = False
            self.root_path_list = persistent.list.PersistentList()
            return

        if _is_windows_path(root):
            pure_path = PureWindowsPath(root)
            self._path_style = "windows"
            self._drive = pure_path.drive
            self._is_absolute = bool(pure_path.root)
            self.root_path_list = persistent.list.PersistentList(list(pure_path.parts[1:]))
            return

        pure_posix = PurePosixPath(root)
        parts = list(pure_posix.parts)
        if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
            self._path_style = "windows"
            self._drive = parts[2].upper() + ":"
            self._is_absolute = True
            self.root_path_list = persistent.list.PersistentList(parts[3:])
            return

        self._path_style = "posix"
        self._drive = ""
        self._is_absolute = pure_posix.is_absolute()
        if parts and parts[0] == "/":
            parts = parts[1:]
        self.root_path_list = persistent.list.PersistentList(parts)

    def _set_root_from_parts(self, parts: Sequence[str]) -> None:
        normalized_parts = list(parts)
        if not normalized_parts:
            self._path_style = "posix"
            self._drive = ""
            self._is_absolute = False
            self.root_path_list = persistent.list.PersistentList()
            return

        if len(normalized_parts) >= 3 and normalized_parts[0] == "" and normalized_parts[1] == "mnt" and len(normalized_parts[2]) == 1:
            self._path_style = "windows"
            self._drive = normalized_parts[2].upper() + ":"
            self._is_absolute = True
            self.root_path_list = persistent.list.PersistentList(normalized_parts[3:])
            return

        if _WINDOWS_DRIVE_RE.match(normalized_parts[0]):
            self._path_style = "windows"
            self._drive = normalized_parts[0].upper()
            self._is_absolute = True
            self.root_path_list = persistent.list.PersistentList(normalized_parts[1:])
            return

        self._path_style = "posix"
        self._drive = ""
        self._is_absolute = normalized_parts[0] == ""
        if self._is_absolute:
            normalized_parts = normalized_parts[1:]
        self.root_path_list = persistent.list.PersistentList(normalized_parts)

    def _build_root_for_current_platform(self) -> str:
        current_system = platform.uname().system
        if current_system == "Windows":
            return self._build_windows_path(include_file=False)
        return self._build_linux_path(include_file=False)

    def _build_linux_path(self, include_file: bool) -> str:
        parts = list(self.root_path_list)
        if include_file and self.file:
            parts.append(self.file)

        if self._path_style == "windows" and self._drive:
            parts = ["mnt", self._drive[0].lower()] + parts
            return "/" + "/".join(parts) if parts else "/"

        prefix = "/" if self._is_absolute else ""
        suffix = "/".join(parts)
        if prefix and suffix:
            return prefix + suffix
        return prefix or suffix

    def _build_windows_path(self, include_file: bool) -> str:
        parts = list(self.root_path_list)
        if include_file and self.file:
            parts.append(self.file)

        if self._path_style == "windows":
            base = self._drive or ""
            if self._is_absolute:
                base += "\\"
            suffix = "\\".join(parts)
            if base and suffix:
                return base + suffix
            return base or suffix

        suffix = "\\".join(parts)
        if self._is_absolute:
            return "\\" + suffix if suffix else "\\"
        return suffix

    def check_for_platform(self) -> None:
        return None

    def copy_for_tree(self) -> "PathFileObj":
        clone = PathFileObj()
        clone._path_style = self._path_style
        clone._drive = self._drive
        clone._is_absolute = self._is_absolute
        clone.root_path_list = persistent.list.PersistentList(list(self.root_path_list))
        clone._file = self._file
        return clone
