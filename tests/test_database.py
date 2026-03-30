from multiprocessing import get_context
from pathlib import Path

from lotdb import LOTDB
from lotdb.utils import load_files_folder


def _read_value_in_process(db_path: str, db_name: str):
    db = LOTDB(path=db_path, name=db_name, read_only=True)
    tree = db.open_connection()
    value = tree.gns(["collection", "item"]).ga("value")
    db.close_connection()
    db.close()
    return value


def test_database_persistence_roundtrip(tmp_path):
    db = LOTDB(path=str(tmp_path), name="test.fs", new=True)

    tree = db.open_connection()
    tree.gns(["collection", "item"]).ga("value", 42)
    db.commit()
    db.close_connection()
    db.close()

    reopened = LOTDB(path=str(tmp_path), name="test.fs")
    tree = reopened.open_connection()

    assert tree.gns(["collection", "item"]).ga("value") == 42

    reopened.close_connection()
    reopened.close()


def test_database_context_manager_closes_connection(tmp_path):
    db = LOTDB(path=str(tmp_path), name="context.fs", new=True)

    with db.connection() as tree:
        tree.gns(["alpha", "beta"]).ga("seen", True)
        db.commit()

    assert db.conn_dict == {}
    db.close()


def test_load_files_folder_builds_tree(tmp_path):
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    (data_dir / "speaker1_~_utt1.wav").write_bytes(b"")
    (data_dir / "speaker1_~_utt2.wav").write_bytes(b"")

    db = LOTDB(path=str(tmp_path), name="folder.fs", new=True)
    tree = db.open_connection()
    processed = load_files_folder(tree, str(data_dir))
    db.commit()

    assert processed == 2

    node = tree.gns(["speaker1", "utt1"])
    assert Path(node.ga("_pfo_audio_wav").filepath).name == "speaker1_~_utt1.wav"

    db.close_connection()
    db.close()


def test_read_only_database_reopen_roundtrip(tmp_path):
    db = LOTDB(path=str(tmp_path), name="readonly.fs", new=True)
    tree = db.open_connection()
    tree.gns(["collection", "item"]).ga("value", 42)
    db.commit()
    db.close_connection()
    db.close()

    reader = LOTDB(path=str(tmp_path), name="readonly.fs", read_only=True)
    tree = reader.open_connection()

    assert tree.gns(["collection", "item"]).ga("value") == 42

    try:
        reader.commit()
    except RuntimeError:
        pass
    else:
        raise AssertionError("read_only LOTDB.commit() should raise RuntimeError")

    reader.close_connection()
    reader.close()


def test_read_only_database_can_minimize_cache_when_requested(tmp_path):
    db = LOTDB(path=str(tmp_path), name="cache.fs", new=True)
    tree = db.open_connection()
    for idx in range(50):
        tree.gns(["items", str(idx)]).ga("value", idx)
    db.commit()
    db.close_connection()
    db.close()

    reader = LOTDB(path=str(tmp_path), name="cache.fs", read_only=True, cache_size=50)
    tree = reader.open_connection()
    for idx in range(50):
        assert tree.gns(["items", str(idx)]).ga("value") == idx

    conn, _ = reader.conn_dict["standard"]
    before = len(conn._cache_items())
    reader.minimize_cache()
    after = len(conn._cache_items())

    assert after <= before

    reader.close_connection()
    reader.close()


def test_read_only_can_use_native_cache_size_configuration(tmp_path):
    db = LOTDB(path=str(tmp_path), name="cache_config.fs", new=True)
    tree = db.open_connection()
    tree.gns(["collection", "item"]).ga("value", 7)
    db.commit()
    db.close_connection()
    db.close()

    reader = LOTDB(path=str(tmp_path), name="cache_config.fs", read_only=True, cache_size=3)
    assert reader.db._cache_size == 3
    tree = reader.open_connection()
    assert tree.gns(["collection", "item"]).ga("value") == 7
    reader.close_connection()
    reader.close()


def test_read_only_database_supports_separate_reader_processes(tmp_path):
    db = LOTDB(path=str(tmp_path), name="mp.fs", new=True)
    tree = db.open_connection()
    tree.gns(["collection", "item"]).ga("value", 99)
    db.commit()
    db.close_connection()
    db.close()

    with get_context("spawn").Pool(processes=2) as pool:
        values = pool.starmap(_read_value_in_process, [(str(tmp_path), "mp.fs"), (str(tmp_path), "mp.fs")])

    assert values == [99, 99]
