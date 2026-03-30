import lotdb
import TreeDB
import StorageTree


def test_new_and_legacy_imports_are_available():
    assert hasattr(lotdb, "StorageTree")
    assert hasattr(lotdb, "LOTDB")
    assert hasattr(TreeDB, "StorageTree")
    assert hasattr(TreeDB, "LOTDB")
    assert hasattr(StorageTree, "StorageTree")
