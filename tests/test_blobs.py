import numpy as np

from lotdb import BaseNode, BlobReader, BlobWriter, LOTDB


def test_blob_bytes_persist_across_database_reopen(tmp_path):
    db = LOTDB(path=str(tmp_path), name="blob.fs", new=True)

    tree = db.open_connection()
    node = tree.gns(["sensor", "capture_001"])
    BlobWriter.write_bytes(
        node,
        b"abc123",
        blob_attribute="measurement_blob",
        content_type="application/octet-stream",
        metadata={"sensor": "imu", "unit": "raw"},
    )
    db.commit()
    db.close_connection()
    db.close()

    reopened = LOTDB(path=str(tmp_path), name="blob.fs")
    tree = reopened.open_connection()
    node = tree.gns(["sensor", "capture_001"])
    blob = node.get_attribute("measurement_blob")

    assert BlobReader.read_bytes(node, "measurement_blob") == b"abc123"
    assert blob.get_metadata("sensor") == "imu"
    assert blob.content_type == "application/octet-stream"

    reopened.close_connection()
    reopened.close()


def test_blob_array_roundtrip(tmp_path):
    db = LOTDB(path=str(tmp_path), name="array_blob.fs", new=True)

    tree = db.open_connection()
    node = tree.gns(["sensor", "capture_002"])
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    BlobWriter.write_array(
        node,
        expected,
        blob_attribute="waveform",
        metadata={"sensor": "accel", "samplerate_hz": 1000},
    )
    db.commit()
    db.close_connection()
    db.close()

    reopened = LOTDB(path=str(tmp_path), name="array_blob.fs")
    tree = reopened.open_connection()
    node = tree.gns(["sensor", "capture_002"])
    blob = node.get_attribute("waveform")
    restored = BlobReader.read_array(node, "waveform")

    np.testing.assert_array_equal(restored, expected)
    assert blob.get_metadata("sensor") == "accel"
    assert blob.get_metadata("samplerate_hz") == 1000
    assert blob.get_metadata("dtype") == "float32"
    assert blob.get_metadata("shape") == (3, 4)

    reopened.close_connection()
    reopened.close()


def test_copy_tree_clones_blob_payloads():
    tree = BaseNode(key="root")
    node = tree.gns(["session", "capture"])
    BlobWriter.write_bytes(node, b"payload", blob_attribute="measurement_blob")

    tree_copy = tree.copy_tree()
    original_blob = tree.gns(["session", "capture"]).get_attribute("measurement_blob")
    copied_blob = tree_copy.gns(["session", "capture"]).get_attribute("measurement_blob")

    assert original_blob is not copied_blob
    assert original_blob.read_bytes() == copied_blob.read_bytes() == b"payload"
