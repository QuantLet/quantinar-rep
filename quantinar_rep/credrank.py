import numpy as np
import pandas as pd

from typing import Dict

from quantinar_rep.config import LOGGER, NODE_WEIGHTS, NODE_TYPE_MAPPER
from quantinar_rep.utils import epoch_weights, \
    compute_minted_amount_from_nodes


def get_users_rank(pr_contribs: pd.DataFrame,
                   period_weights: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        pr_contribs: Pagerank for all nodes in a dataframe with columns
        "index", "rank", "label", "contributor", "period", "userID"
        period_weights: Period discount rate with a column "weight"

    Returns:

    """

    user_ranks = {}
    for user_id in set(pr_contribs.userID):
        user_pr = pr_contribs.loc[
            pr_contribs["userID"] == user_id].sort_values(by="period", axis=0)
        user_ranks[user_id] = [(user_pr.set_index("period")["rank"] *
                                period_weights["weight"]).dropna().sum()]
    user_ranks = pd.DataFrame(user_ranks).T

    user_ranks.sort_values(by=0, axis=0, inplace=True, ascending=False)
    user_ranks.reset_index(inplace=True)
    user_ranks.columns = ["userID", "rank"]

    return user_ranks


def get_historical_users_pagerank(pr_result: Dict,
                                  weighting_method: str = "exponential",
                                  **kwargs):
    """
    Compute the timeseries of users scores for each period in history given
    the epoch nodes (equation (4) in the paper).

    Args:
        pr_result: All periods result from time_pagerank script in a dictionary
        weighting_method: Name of the period weighting method
        **kwargs:

    Returns:

    """
    keys = list(pr_result.keys())
    nb_periods = len(keys)

    users_ranks = {}
    for i in range(nb_periods):
        LOGGER.debug(f"Get users epoch score at epoch {i}, steps to go: "
                     f"{nb_periods - i}")
        k = keys[i]
        res_k = pd.read_json(pr_result[k])
        res_k = res_k.reset_index()
        res_k["node_type"] = [NODE_TYPE_MAPPER.get(node.split("_")[0]) for
                              node in res_k["index"]]
        # Get users epoch nodes
        res_k = res_k[res_k["node_type"] == "user"]
        mask = res_k["index"].apply(lambda x: any(
            [x.split("_")[-1] == f"t{j}" for j in range(i + 1)]))
        pr_contribs = res_k[mask].copy()
        pr_contribs["period"] = pr_contribs["index"].apply(
            lambda x: np.array(range(i + 1))[[x.split("_")[-1] == f"t{j}"
                                              for j in range(i + 1)]][0]
        )
        pr_contribs["userID"] = pr_contribs["index"].apply(
            lambda x: x.split("_")[0]
        )

        if weighting_method == "exponential":
            decay = kwargs.get("decay", 0.9)
            period_weights = epoch_weights(decay, i + 1)
        else:
            raise NotImplementedError()
        users_ranks[k] = get_users_rank(pr_contribs, period_weights)

    ts_users_rank = pd.DataFrame()
    for k in users_ranks:
        ts_users_rank = pd.concat([ts_users_rank,
                                   pd.DataFrame(
                                       users_ranks[k].set_index("userID")[
                                           "rank"])
                                   ],
                                  axis=1)

    ts_users_rank = ts_users_rank.T
    ts_users_rank.index = pd.to_datetime(keys)

    # Sort columns by last value
    ts_users_rank = ts_users_rank[
        ts_users_rank.iloc[-1].sort_values(ascending=False).index]

    return users_ranks, ts_users_rank


def get_rank_from_pagerank(df_pagerank):
    """
    Compute the rank of each contributor given the pagerank scores.

    Args:
        df_pagerank:

    Returns:

    """
    order = pd.DataFrame(
        df_pagerank.apply(
            axis=1,
            func=lambda x: x.sort_values(na_position="first").index.tolist()
        )
    )
    ranking = pd.DataFrame(columns=df_pagerank.columns,
                           index=df_pagerank.index)
    for c in ranking.columns:
        ranking[c] = [[i for (i, a) in enumerate(order.loc[d][0]) if a == c][0]
                      for d in order.index]
    ranking[df_pagerank.isna()] = np.nan

    ranking = ranking.max(1).values.reshape(-1, 1) - ranking + 1

    return ranking


def compute_period_minted_amount(result: Dict,
                                 node_weights_mapper=NODE_WEIGHTS) -> pd.Series:
    """
    Compute the minted QNAR per period

    Args:
        result: All periods result from time_pagerank script in a dictionary
        node_weights_mapper:

    Returns:

    """
    keys = list(result.keys())
    m_k = {}
    for i in range(len(keys)):
        k = keys[i]
        pagerank_k = pd.read_json(result[k])
        if i == 0:
            prev_nodes = set()
            new_nodes = set(pagerank_k.index)
        else:
            prev_nodes = set(pd.read_json(result[keys[i - 1]]).index)
            new_nodes = set(pagerank_k.index) - prev_nodes

        assert all([n not in prev_nodes for n in new_nodes])
        m_k[k] = compute_minted_amount_from_nodes(
            new_nodes,
            node_weights_mapper=node_weights_mapper)
    m_k = pd.Series(m_k)
    m_k.index = pd.to_datetime(m_k.index)

    return m_k

