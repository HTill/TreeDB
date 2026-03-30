import platform

from lotdb import PathFileObj


def test_path_file_obj_roundtrip_from_filepath():
    pfo = PathFileObj(filepath="/tmp/example/file.wav")

    assert pfo.file == "file.wav"
    assert pfo.as_linux_path() == "/tmp/example/file.wav"
    assert pfo.as_windows_path() == r"\tmp\example\file.wav"

    if platform.system() == "Windows":
        assert pfo.root == r"\tmp\example"
        assert pfo.filepath == r"\tmp\example\file.wav"
    else:
        assert pfo.root == "/tmp/example"
        assert pfo.filepath == "/tmp/example/file.wav"


def test_path_file_obj_from_root_and_file():
    pfo = PathFileObj(root="/tmp/demo", file="data.bin")

    assert pfo.file == "data.bin"
    assert pfo.as_linux_path() == "/tmp/demo/data.bin"
    assert pfo.as_windows_path() == r"\tmp\demo\data.bin"


def test_windows_path_is_normalized_for_linux_and_windows():
    pfo = PathFileObj(filepath=r"C:\Users\Till\audio.wav")

    assert pfo.as_windows_path() == r"C:\Users\Till\audio.wav"
    assert pfo.as_linux_path() == "/mnt/c/Users/Till/audio.wav"


def test_mnt_path_roundtrips_back_to_windows_drive():
    pfo = PathFileObj(filepath="/mnt/d/projects/sample.txt")

    assert pfo.as_linux_path() == "/mnt/d/projects/sample.txt"
    assert pfo.as_windows_path() == r"D:\projects\sample.txt"


def test_path_file_obj_read_access_does_not_mark_object_changed():
    pfo = PathFileObj(filepath="/tmp/example/file.wav")

    pfo._p_changed = False
    _ = pfo.filepath
    assert pfo._p_changed is False

    _ = pfo.root
    assert pfo._p_changed is False


def test_copy_for_tree_preserves_relative_linux_style_paths():
    pfo = PathFileObj(filepath="sensor/capture_002/data")

    copied = pfo.copy_for_tree()

    assert copied.as_linux_path() == "sensor/capture_002/data"
