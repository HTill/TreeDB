from pathlib import Path

from lotdb import LOTDB


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
    processed = db.load_files_folder(str(data_dir))

    assert processed == 2

    tree = db.open_connection()
    node = tree.gns(["speaker1", "utt1"])
    assert Path(node.ga("_pfo_audio_wav").filepath).name == "speaker1_~_utt1.wav"

    db.close_connection()
    db.close()
