from pathlib import Path

import numpy as np

from lotdb import LOTDB, DataNode, DataReader, DataWriter


def test_blob_measurement_supports_samplerate_seconds_reads(tmp_path):
    db = LOTDB(path=str(tmp_path), name="measurement_blob.fs", new=True)
    tree = db.open_connection()
    node = tree.gns(["sensor", "capture_001"])

    signal = np.arange(20, dtype=np.float32).reshape(10, 2)
    DataWriter.write_array(
        node,
        signal,
        samplerate_hz=4,
        backend="blob",
        data_attribute="data",
        metadata={"sensor": "imu"},
    )
    db.commit()
    db.close_connection()
    db.close()

    reopened = LOTDB(path=str(tmp_path), name="measurement_blob.fs")
    tree = reopened.open_connection()
    node = tree.gns(["sensor", "capture_001"])

    seconds_slice = DataReader.read_seconds(node, "data", 1.0, 2.0)
    np.testing.assert_array_equal(seconds_slice, signal[4:8])

    blocks = list(DataReader.iter_blocks(node, "data", block_size=0.5, block_unit="seconds"))
    assert len(blocks) == 5
    np.testing.assert_array_equal(blocks[0], signal[:2])
    np.testing.assert_array_equal(blocks[-1], signal[8:10])

    reopened.close_connection()
    reopened.close()


def test_zarr_measurement_links_and_streams_blocks(tmp_path):
    db = LOTDB(path=str(tmp_path), name="measurement_zarr.fs", new=True)
    tree = db.open_connection()
    node = tree.gns(["sensor", "capture_002"])

    signal = np.arange(30, dtype=np.float32).reshape(15, 2)
    data = DataWriter.write_array(
        node,
        signal,
        samplerate_hz=5,
        backend="zarr",
        data_attribute="data",
        database=db,
        chunks=(4, 2),
        metadata={"sensor": "emg"},
    )
    db.commit()
    db.close_connection()
    db.close()

    assert data.store_path is not None
    assert Path(data.store_path).exists()

    reopened = LOTDB(path=str(tmp_path), name="measurement_zarr.fs")
    tree = reopened.open_connection()
    node = tree.gns(["sensor", "capture_002"])
    linked_data = DataReader.open(node, "data")

    assert linked_data.backend == "zarr"
    assert linked_data.get_metadata("sensor") == "emg"
    assert linked_data.get_metadata("zarr_dataset_path") == "sensor/capture_002/data"
    np.testing.assert_array_equal(DataReader.read_interval(node, "data", 3, 7), signal[3:7])

    blocks = list(
        DataReader.iter_blocks(
            node,
            "data",
            block_size=1.0,
            block_unit="seconds",
            start=1.0,
            stop=3.0,
            range_unit="seconds",
        )
    )
    assert len(blocks) == 2
    np.testing.assert_array_equal(blocks[0], signal[5:10])
    np.testing.assert_array_equal(blocks[1], signal[10:15])

    reopened.close_connection()
    reopened.close()


def test_get_data_node_creates_specialized_node_and_uses_defaults(tmp_path):
    db = LOTDB(path=str(tmp_path), name="measurement_node.fs", new=True)
    tree = db.open_connection()
    data_node = tree.get_data_node(
        ["session", "capture_003"],
        samplerate_hz=2,
        backend="blob",
        data_attribute="imu",
    )

    assert isinstance(data_node, DataNode)

    signal = np.arange(12, dtype=np.float32).reshape(6, 2)
    data_node.write_data(signal, database=db)
    db.commit()
    db.close_connection()
    db.close()

    reopened = LOTDB(path=str(tmp_path), name="measurement_node.fs")
    tree = reopened.open_connection()
    data_node = tree.get_data_node(["session", "capture_003"], create=False)

    np.testing.assert_array_equal(data_node.read_seconds(1.0, 2.0), signal[2:4])
    blocks = list(data_node.iter_data_blocks(block_size=1.0, block_unit="seconds"))
    assert len(blocks) == 3
    np.testing.assert_array_equal(blocks[0], signal[:2])

    reopened.close_connection()
    reopened.close()


def test_data_reader_writer_cover_legacy_file_helpers(tmp_path):
    node = DataNode(key="capture", samplerate_hz=1)
    array = np.arange(6, dtype=np.float32).reshape(3, 2)

    DataWriter.write_array_npy(root=str(tmp_path), node=node, array=array, filename="capture.npy")

    restored = DataReader.read_array_npy(node)
    np.testing.assert_array_equal(restored, array)
