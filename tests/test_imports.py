import lotdb


def test_primary_public_imports_are_available():
    assert hasattr(lotdb, "BaseNode")
    assert hasattr(lotdb, "DataNode")
    assert hasattr(lotdb, "LOTDB")
    assert not hasattr(lotdb, "FileReader")
    assert not hasattr(lotdb, "FileWriter")
    assert not hasattr(lotdb, "filter_node_list")
    assert not hasattr(lotdb, "node_process_cruncher")
