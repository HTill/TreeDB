from lotdb import BaseNode


def test_node_creation_and_attribute_access():
    tree = BaseNode(key="root")
    node = tree.get_node_path(["artist", "album", "track"])

    node.set_attribute("duration", 123)

    assert node.key == "track"
    assert node.get_attribute("duration") == 123
    assert tree.gns(["artist", "album", "track"]).ga("duration") == 123


def test_key_rename_updates_parent_mapping():
    tree = BaseNode(key="root")
    node = tree.gn("old_name")

    node.key = "new_name"

    assert "old_name" not in tree.all_node_keys()
    assert tree.gn("new_name") is node


def test_add_tree_without_copy_keeps_same_object():
    root = BaseNode(key="root")
    source = BaseNode(key="child")

    added = root.add_tree(source, copy=False)

    assert added is source
    assert root.gn("child") is source
    assert source.parent is root


def test_copy_tree_creates_detached_deep_copy():
    tree = BaseNode(key="root")
    leaf = tree.gns(["a", "b"])
    leaf.ga("value", 7)

    tree_copy = tree.copy_tree()
    copied_leaf = tree_copy.gns(["a", "b"])

    assert tree_copy is not tree
    assert copied_leaf is not leaf
    assert copied_leaf.ga("value") == 7
    assert tree_copy.parent is None


def test_merge_tree_merges_recursively():
    base = BaseNode(key="root")
    base.gns(["a", "shared"]).ga("left", 1)

    incoming = BaseNode(key="incoming")
    incoming.gns(["a", "shared"]).ga("right", 2)

    base.merge_tree(incoming)

    merged = base.gns(["a", "shared"])
    assert merged.ga("left") == 1
    assert merged.ga("right") == 2


def test_delete_node_only_node_promotes_children():
    tree = BaseNode(key="root")
    branch = tree.gns(["branch"])
    branch.gns(["child_one"])
    branch.gns(["child_two"])

    tree.delete_node("branch", only_node=True)

    assert "branch" not in tree.all_node_keys()
    assert sorted(tree.all_node_keys()) == ["child_one", "child_two"]


def test_iterate_tree_level_and_leaves():
    tree = BaseNode(key="root")
    tree.gns(["a", "b"])
    tree.gns(["a", "c"])

    deepest_keys = sorted(node.key for node in tree.iterate_tree_level("deepest"))
    leaf_keys = sorted(node.key for node in tree.iterate_tree_leaves())

    assert deepest_keys == ["b", "c"]
    assert leaf_keys == ["b", "c"]
