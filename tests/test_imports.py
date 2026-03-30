import lotdb


def test_primary_public_imports_are_available():
    assert hasattr(lotdb, "BaseNode")
    assert hasattr(lotdb, "DataNode")
    assert hasattr(lotdb, "LOTDB")
