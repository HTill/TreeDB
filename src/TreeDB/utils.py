from __future__ import annotations

import os
import pickle
import re
from itertools import repeat
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional

from numpy import ndarray

from .tree import StorageTree


def filter_node_list(filter_str: str, node_list: List[StorageTree], invers: bool = False) -> List[StorageTree]:
    filtered_node_list = []

    for node in node_list:
        matches = re.search(filter_str, node.key) is not None
        if invers and not matches:
            filtered_node_list.append(node)
        elif not invers and matches:
            filtered_node_list.append(node)

    return filtered_node_list


def save_to_file(oject: Any, path: Optional[str] = None) -> None:
    if path is None:
        path = "saved_obj"

    with open(f"{path}.pyobj", mode="wb") as file_handle:
        pickle.dump(oject, file_handle)


def load_from_file(path: str):
    with open(path, mode="rb") as file_handle:
        oject = pickle.load(file_handle)

    return oject


def node_process_cruncher(
    function,
    node_list: List[StorageTree],
    other_args_dic: Optional[Dict[str, Any]] = None,
    processes: Optional[int] = None,
):
    if not node_list:
        return []

    main_tree = node_list[0].get_main_tree()
    copy_node_list = []

    for node in node_list:
        copy_node = node.copy_tree()
        copy_node.ga("parents", node.gps())
        copy_node.parent = None
        copy_node_list.append(copy_node)

    if processes is None:
        cpu_count = os.cpu_count() or 1
        processes = max(cpu_count - 1, 1)

    kwargs_iter = repeat(other_args_dic or {})

    with get_context("spawn").Pool(processes=processes) as pool:
        results = starmap_with_kwargs(pool, function, copy_node_list, kwargs_iter)

    for result_node in results:
        target_node = main_tree.gns(result_node.ga("parents"))
        target_node.merge_tree(result_node, overwrite=True)
        target_node.delete_attribute("parents")

    return results


def starmap_with_kwargs(pool, fn, node_list: Iterable[StorageTree], kwargs_iter: Iterable[Dict[str, Any]]):
    args_for_starmap = zip(repeat(fn), node_list, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, node: StorageTree, kwargs: Dict[str, Any]):
    return fn(node, **kwargs)


def tree_to_dataframe(tree: StorageTree, attribute_list: List[str]):
    import pandas as pd

    dataframe = pd.DataFrame()
    last_pidx = 0

    for pidx in range(tree.get_max_depth()):
        parent_name = f"TreeLevel_{pidx + 1}"
        dataframe.insert(pidx, column=parent_name, value=[])
        last_pidx = pidx

    for node in tree.iterate_tree_leaves():
        tree_key = tree.key
        node_parents = node.gps()
        cur_idx = node_parents.index(tree_key) + 1 if tree_key in node_parents else 0
        node_parents = node_parents[cur_idx:]
        dataframe = dataframe._append(pd.DataFrame([node_parents], columns=dataframe.columns[: len(node_parents)]), ignore_index=True)

    last_pidx += 1
    for att in attribute_list:
        att_list = []
        for node in tree.iterate_tree_leaves():
            cur_att = node.ga(att)
            if isinstance(cur_att, ndarray):
                cur_att = cur_att.tolist()
            att_list.append(cur_att)

        dataframe.insert(last_pidx, column=att, value=att_list)
        last_pidx += 1

    return dataframe


def node_list_to_dataframe(node_list: List[StorageTree], attribute_list: List[str]):
    import pandas as pd

    dataframe = pd.DataFrame()
    last_pidx = 0

    if not node_list:
        return dataframe

    for pidx in range(len(node_list[0].gps())):
        parent_name = f"TreeLevel_{pidx + 1}"
        dataframe.insert(pidx, column=parent_name, value=[])
        last_pidx = pidx

    for node in node_list:
        node_parents = node.gps()
        dataframe = dataframe._append(pd.DataFrame([node_parents], columns=dataframe.columns[: len(node_parents)]), ignore_index=True)

    last_pidx += 1
    for att in attribute_list:
        att_list = []
        for node in node_list:
            cur_att = node.ga(att)
            if isinstance(cur_att, ndarray):
                cur_att = cur_att.tolist()
            att_list.append(cur_att)

        dataframe.insert(last_pidx, column=att, value=att_list)
        last_pidx += 1

    return dataframe


def test_counter(node: StorageTree, offset: int):
    node.ga("c", node.ga("a") + node.ga("b") + offset)
    return node
