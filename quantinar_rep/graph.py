import datetime as dt
import itertools

import networkx as nx
import numpy as np
import pandas as pd

from quantinar_rep.config import DEFAULT_BETA, DEFAULT_GAMMA_FORWARD, \
    DEFAULT_GAMMA_BACKWARD

from typing import Optional

from quantinar_rep.utils import node_attributes_mapper, is_contributor


def normalize_children_edges_weight(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Normalized children edges weights for each parent nodes from the
    given graph edges.

    Args:
        edges:  Given graph edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type", "target_type"

    Returns:
        Edges dataframe with normalized weights
    """
    # Get the residual probability for each node (sum of defined transition
    # probability (alpha, beta, etc.))
    res_proba = edges.groupby("source", group_keys=False)["proba"].sum()
    # Normalize the weights with the residual probability
    norm_weights = edges.groupby(
        "source", group_keys=False)["weight"].apply(
        lambda x: get_transition_proba_from_weights(x,
                                                    proba=res_proba[x.name]))
    # Fill missing weights with defined transition probability
    mask = norm_weights.index[norm_weights.isna()]
    norm_weights[mask] = edges.loc[mask, "proba"]
    edges["norm_weight"] = norm_weights.astype(np.float32)

    return edges


def get_transition_proba_from_weights(weights: np.ndarray,
                                      proba=0.1) -> np.ndarray:
    """
    Normalized the edges weights given the residual transition probability

    Args:
        weights: Array of weights
        proba: Residual probability for the outbound edges

    Returns:
        Normalized weights
    """
    div = np.sum(weights) / (1 - proba)
    return weights / div


def create_graph(graph_edges: pd.DataFrame) -> nx.DiGraph:
    """
    Create a directed graph from edges in the given dataframe
    Args:
        graph_edges: All edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type", "target_type"

    Returns:

    """
    G = nx.from_pandas_edgelist(
        graph_edges,
        edge_attr=['weight', 'timestamp', 'source_type', 'target_type'],
        create_using=nx.DiGraph()
    )
    nx.set_node_attributes(G, {n: node_attributes_mapper(n) for n in list(
        G.nodes)})
    return G


def add_webbing_edges(prev_edges: pd.DataFrame, period: int, date: dt.datetime,
                      gamma_f: float = DEFAULT_GAMMA_FORWARD,
                      gamma_b: float = DEFAULT_GAMMA_BACKWARD) -> pd.DataFrame:
    """
    Create webbing edges betweeen epoch nodes for the given edges

    Args:
        prev_edges: Given edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type", "target_type"
        period: Period number
        date: Date of the period
        gamma_f: Forward transition probability between epoch nodes
        gamma_b: Backward transition probability between epoch nodes

    Returns:
        Webbing edges in a dataframe
    """
    # Get contributors from previous period
    contributors = set(prev_edges[prev_edges["target_type"] == "user"][
                           "target"])

    # Create webbing edges front- and back-ward
    webbing_edges = [
        [f"{n}_t{period - 1}", f"{n}_t{period}", gamma_f, date,
         f"user_epoch_t{period - 1}", f"user_epoch_t{period}"] for n in
        contributors
    ]
    webbing_edges += [
        [f"{n}_t{period}", f"{n}_t{period - 1}", gamma_b, date,
         f"user_epoch_t{period}", f"user_epoch_t{period - 1}"] for n in
        contributors
    ]
    webbing_edges = pd.DataFrame(
        webbing_edges,
        columns=["source", "target", "proba", "timestamp",
                 "source_type", "target_type"]
    )
    return webbing_edges


def add_epoch_contributor_edges(edges: pd.DataFrame, period: int, beta: float,
                                date: dt.datetime) -> pd.DataFrame:
    """
    Create edges from epoch nodes to global contributor nodes for all
    contributors in edges

    Args:
        edges: All edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type",
        "target_type" for the given period.
        period: period number
        beta: Transition probability from epoch nodes to author node
        date: Current period date

    Returns:
        Epoch global author node edges in a dataframe
    """
    # Rename current period contributors nodes to f"nodes_t{period}"
    user_contrib_mask = (edges["source_type"] == "user") & (
                edges["target_type"] != "seed")
    contrib_user_mask = (edges["source_type"] != "seed") & (
                edges["target_type"] == "user")

    edges.loc[user_contrib_mask, "source"] = edges.loc[
                                                 user_contrib_mask,
                                                 "source"] + f"_t{period}"
    edges.loc[user_contrib_mask, "source_type"] = f"user_epoch_t{period}"

    edges.loc[contrib_user_mask, "target"] = edges.loc[
                                                 contrib_user_mask,
                                                 "target"] + f"_t{period}"
    edges.loc[contrib_user_mask, "target_type"] = f"user_epoch_t{period}"

    # Create edge from current period contributor to contributor node
    period_contributors = set(
        edges.loc[user_contrib_mask, "source"].tolist() +
        edges.loc[contrib_user_mask, "target"].tolist()
    )
    period_author_edges = [
        [
            p_contrib,
            p_contrib[:-len(f'_t{period}')],
            beta,
            date,
            f"user_epoch_t{period}",
            "user"
        ] for p_contrib in period_contributors
    ]
    period_author_edges = pd.DataFrame(
        period_author_edges,
        columns=["source", "target", "proba", "timestamp",
                 "source_type", "target_type"]
    )

    return period_author_edges


def create_graph_edges_with_webbing(
        period_edges: pd.DataFrame, period: int, end: dt.datetime,
        prev_edges: Optional[pd.DataFrame] = None,  beta: float = DEFAULT_BETA,
        gamma_f: float = DEFAULT_GAMMA_FORWARD,
        gamma_b: float = DEFAULT_GAMMA_BACKWARD,
) -> pd.DataFrame:
    """
    Create the edges for the contribution graph as defined in the paper (
    with webbing edges).

    Args:
        period_edges: All edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type", "target_type"
        for the given period.
        period: Current period number
        end: Last included date in the current period
        prev_edges: Edges from the previous period
        beta: Transition probability from epoch nodes to author node
        gamma_f: Forward transition probability between epoch nodes
        gamma_b: Backward transition probability between epoch nodes

    Returns:

    """
    if period > 0:
        assert prev_edges is not None, "You must provide prev_edges for " \
                                       "period > 0"
    # Create new period contributor nodes and edges with contributor nodes

    if prev_edges is not None:
        edges = prev_edges.drop("norm_weight", axis=1)
        webbing_edges = add_webbing_edges(edges, period, end,
                                          gamma_f=gamma_f,  gamma_b=gamma_b)
    else:
        edges = pd.DataFrame()

    # Get edges for the period

    if prev_edges is not None:
        assert len(period_edges) > 0, "At first iteration you must give a " \
                                      "valid period! Verify the start date!"

    if len(period_edges) > 0:
        # Add edges from epoch nodes to global contributor nodes
        period_author_edges = add_epoch_contributor_edges(period_edges, period,
                                                          beta, end)
        period_contributors = period_author_edges["target"].tolist()
        assert len(set(period_contributors)) == len(period_contributors)
        period_edges = pd.concat([period_edges, period_author_edges])

    if prev_edges is not None:
        # Now some contributors from previous period are not in current period
        # Let s add the missing epoch node to global node edges:
        period_prev_authors = prev_edges.loc[
            prev_edges["target_type"] == "user", "target"]
        if len(period_edges) > 0:
            period_prev_authors = list(set(period_prev_authors) - set(
                period_contributors))
        period_prev_author_edges = [
            [
                f"{p_contrib}_t{period}",
                p_contrib,
                beta,
                end,
                f"user_epoch_t{period}",
                "user"
            ] for p_contrib in period_prev_authors
        ]
        period_prev_author_edges = pd.DataFrame(
            period_prev_author_edges,
            columns=["source", "target", "proba", "timestamp",
                     "source_type", "target_type"]
        )
        period_edges = pd.concat([period_edges, period_prev_author_edges],
                                 ignore_index=True)
        period_edges = pd.concat([period_edges, webbing_edges],
                                 ignore_index=True)

    # Now add the edges: period edges and edges to contributor node
    edges = pd.concat([edges, period_edges], ignore_index=True)

    # Normalize all weights with existing transition probabilities
    edges = normalize_children_edges_weight(edges)
    edges = edges.sort_values(by="timestamp")
    assert sum(edges.duplicated()) == 0
    assert edges["norm_weight"].isna().sum() == 0
    assert all(edges.groupby("source", group_keys=False)["norm_weight"].sum()
               <= 1)
    assert set(edges[edges["source_type"] == "user"]["target_type"]) == set(
        ["seed"]), "Problem with global contributor nodes children"
    assert set(edges[edges["target_type"] == "user"]["source_type"]) == set(
        [f"user_epoch_t{i}" for i in range(period+1)]), "Problem with global" \
                                                        "contributor nodes" \
                                                        "parents"

    webbing_frontward = edges[
        (edges["source_type"] == f"user_epoch_t{period - 1}") & (
                edges["target_type"] == f"user_epoch_t{period}")]
    webbing_backward = edges[
        (edges["source_type"] == f"user_epoch_t{period}") & (
                edges["target_type"] == f"user_epoch_t{period - 1}")]
    assert len(webbing_frontward) == len(webbing_backward), "Problem with " \
                                                            "webbing edges"
    return edges


def create_contributor_subgraph(edges: pd.DataFrame) -> nx.DiGraph:
    """
    Create a graph with contributors only. We connect the nodes if a path
    that does not go through the seed node, exists in the original graph
    between contributors

    Args:
        edges: All edges in a dataframe with columns "source",
        "target", "weight", "timestamp", "proba", "source_type", "target_type"

    Returns:

    """
    sub_edges = edges[["pv" not in l for l in edges[
        "source_type"]]]
    sub_edges = sub_edges[["seed" != l for l in sub_edges["target_type"]]]
    sub_edges = sub_edges[["seed" != l for l in sub_edges["source_type"]]]
    sub_edges["weight"] = 1.
    sub_graph = create_graph(sub_edges)

    contributors = [n for n in sub_graph.nodes if is_contributor(n)]
    contributors = np.unique(contributors).tolist()
    connected = []
    for i, j in itertools.product(contributors, contributors):
        if i == j:
            pass
        else:
            try:
                path = nx.shortest_path(sub_graph, source=i, target=j,
                                        method='dijkstra')
                # if there is more than two users in the path then there are some user in the middle of the path
                # and the two users are not directly linked via a contribution
                n_is_user = [is_contributor(n) for n in path]
                if sum(n_is_user) == 2:
                    connected.append((i, j))
            except nx.NetworkXNoPath as _exc:
                print(_exc)

    simple_edges = []
    for c in contributors:
        simple_edges.append(("seed", c))
    simple_edges += connected

    simple_graph = nx.DiGraph()
    simple_graph.add_edges_from(simple_edges)
    nx.set_node_attributes(simple_graph,
                           {n: node_attributes_mapper(n) for n in
                            set(np.array(simple_edges).flatten())})
    simple_graph.remove_node("seed")

    return simple_graph
