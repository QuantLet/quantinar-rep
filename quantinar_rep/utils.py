import numpy as np
import pandas as pd

from typing import Dict, List

from quantinar_rep.config import NODE_WEIGHTS, NODE_TYPE_MAPPER
from quantinar_rep.constants import PAGE_COLOR, USER_COLOR, COURSE_COLOR, \
    REVIEW_COLOR, ORDER_COLOR, COURSELET_COLOR, PAGE_BASE, \
    USER_BASE, COURSE_BASE, COURSELET_BASE, REVIEW_BASE, ORDER_BASE, \
    EPOCH_NODE_COLOR


def epoch_weights(decay: float, nb_periods: int):
    period_weights = [[k, 1.0 * (decay ** (nb_periods - k - 1))] for k in
                      range(nb_periods)]
    period_weights = pd.DataFrame(period_weights, columns=["period", "weight"])
    period_weights.set_index("period", inplace=True)

    return period_weights


def is_contributor(node):
    return NODE_TYPE_MAPPER.get(node) == "user"


def get_all_nodes_from_user_id(nodes, user_id):
    return [n.split("_")[0] == user_id for n in nodes]


def node_attributes_mapper(node, timestamp=None, node_weights_mapper=None):
    if node_weights_mapper is None:
        node_weights_mapper = NODE_WEIGHTS
    if node.split("_")[0] == PAGE_BASE:
        return {"color": PAGE_COLOR, "label": "pv", "timestamp": timestamp,
                "weight": node_weights_mapper[PAGE_BASE]}
    elif NODE_TYPE_MAPPER.get(node) == USER_BASE:
        return {"color": USER_COLOR, "label": "user", "timestamp": timestamp,
                "weight": node_weights_mapper[USER_BASE]}
    elif NODE_TYPE_MAPPER.get(node) == COURSE_BASE:
        return {"color": COURSE_COLOR, "label": "course",
                "timestamp": timestamp,
                "weight": node_weights_mapper[COURSE_BASE]}
    elif NODE_TYPE_MAPPER.get(node) == COURSELET_BASE:
        return {"color": COURSELET_COLOR, "label": "courselet",
                "timestamp": timestamp,
                "weight": node_weights_mapper[COURSELET_BASE]}
    elif NODE_TYPE_MAPPER.get(node) == REVIEW_BASE:
        return {"color": REVIEW_COLOR, "label": "review",
                "timestamp": timestamp,
                "weight": node_weights_mapper[REVIEW_BASE]}
    elif NODE_TYPE_MAPPER.get(node) == ORDER_BASE:
        return {"color": ORDER_COLOR, "label": "order", "timestamp": timestamp,
                "weight": node_weights_mapper[ORDER_BASE]}
    elif node == "seed":
        return {"color": "yellow", "label": "seed", "timestamp": timestamp,
                "weight": np.nan}
    elif NODE_TYPE_MAPPER.get(node.split("_")[0]) == USER_BASE:
        return {"color": EPOCH_NODE_COLOR, "label": "epoch_node",
                "timestamp":  timestamp, "weight": node_weights_mapper[
                USER_BASE]}
    else:
        raise ValueError(node)


def set_weights(edges: pd.DataFrame, weights: Dict):
    """
    Set the edges weights from weights dictionary

    Parameters
    ----------
    edges: Edges DataFrame
    weights: New weights dictionary with edge's type as keys.

    Returns
    -------

    """
    for edge in weights:
        weight = weights[edge]
        source_type = edge.split("_")[0]
        target_type = edge.split("_")[1]
        mask = (edges["source_type"] == source_type) & (
                edges["target_type"] == target_type)
        if edge != "pv_course":
            assert sum(mask) > 0
            edges.loc[mask, "weight"] = weight
    return edges


def compute_minted_amount_from_nodes(nodes: List,
                                     node_weights_mapper=NODE_WEIGHTS):
    """
    Compute the sum of QNAR that should be minted for the given nodes.

    Args:
        nodes: list of nodes
        node_weights_mapper: Dictionary of node weights

    Returns:

    """
    nodes_weights = {
        n: node_attributes_mapper(n,
                                  node_weights_mapper=node_weights_mapper)[
            "weight"] for n in nodes
    }
    return sum(nodes_weights.values())
