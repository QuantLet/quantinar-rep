"""
Script for running pagerank algorithm on Quantinar graph. The connections
are the following:

- page view -> courselet
- student (user) -> courselet
- review -> courselet
- review -> user
- courselet -> user

The maximum review is 5 for a course, so the maximum weight for review ->
courselet is  5. Reviews that are 0 do not contribute to the courselet rank
with a weight of 0.

Since having a student enrolled is, we could say, equally  important as
having a good review, we set the student -> courselet weight to 5.
That way if a  student is enrolled to a courselet and left a review of 5,
the total  contribution of that user to the courselet is 10.

A user that reviews, correctly, is more important than a user that does not
review, that is why we have the connection review -> user. If a user reviews
correctly, that is his review is close to the average review, than the
weight is set to 1 otherwise 0: a bad review does not improve the rank of
the user.

The page view -> courselet is the weakest connection, we set it to 5/100.

The courselet -> author is a identity connection. No connection ends with
authors except if it comes from a courselet.

"""
import logging
import os

import pandas as pd

from quantinar_rep.config import LOGGER, DEFAULT_ALPHA, FREQ_DAYS, \
    PERSONALIZATION_METHOD, DEFAULT_WEIGHTS, DEFAULT_GAMMA_FORWARD, \
    DEFAULT_GAMMA_BACKWARD, DEFAULT_BETA, \
    EDGES_HISTORY_PATH, PV_HISTORY_PATH
from quantinar_rep.graph import create_graph, \
    create_graph_edges_with_webbing
from quantinar_rep.pagerank import pagerank

import datetime as dt
import json

from quantinar_rep.utils import set_weights, is_contributor


def personalization_mapper(method: str, value=None):
    if method is None:
        return None
    elif method == "seed":
        if value:
            return value
        else:
            return {"seed": 1.}
    elif method == "pagerank":
        return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Create graph from data and run PageRank algorithm.')
    parser.add_argument("--alpha", type=float, help="alpha",
                        default=DEFAULT_ALPHA)
    parser.add_argument("--gamma_f", type=float, help="gamma_f",
                        default=DEFAULT_GAMMA_FORWARD)
    parser.add_argument("--gamma_b", type=float, help="gamma_b",
                        default=DEFAULT_GAMMA_BACKWARD)
    parser.add_argument("--beta", type=float, help="beta",
                        default=DEFAULT_BETA)
    parser.add_argument('--no_pv',  action="store_true",
                        help="do not include pageviews",
                        default=True)
    parser.add_argument('--plot', action="store_true", help="plot the graph")

    parser.add_argument("--edges_history_path", type=str,
                        help="Path to the edges csv",
                        default=EDGES_HISTORY_PATH)
    parser.add_argument("--pv_history_path", type=str,
                        help="Path to the pv edges csv",
                        default=PV_HISTORY_PATH)
    parser.add_argument("--node_label_mapper", type=str,
                        help="Path to the node label mapper")
    args = parser.parse_args()

    if args.node_label_mapper is not None:
        NODE_LABEL_MAPPER = json.load(open(args.node_label_mapper, "r"))

    if not os.path.isdir("./results/"):
        os.makedirs("./results/")

    save_dir = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_pagerank"
    save_dir = f"./results/{save_dir}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    fh = logging.FileHandler(f"./{save_dir}/main.log")
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)

    LOGGER.info(f"Running Time Page Rank with alpha={args.alpha}, "
                f"beta={args.beta}, gamma_f={args.gamma_f}, gamma_b"
                f"={args.gamma_b}, pageview={not args.no_pv}...")

    script_weights = DEFAULT_WEIGHTS.copy()
    script_proba = {}
    script_proba["alpha"] = args.alpha
    script_proba["gamma_b"] = args.gamma_b
    script_proba["gamma_f"] = args.gamma_f
    script_proba["beta"] = args.beta

    params = {
        "freq_days": FREQ_DAYS,
        "proba": script_proba,
        "weights": script_weights,
        "personalization_method": PERSONALIZATION_METHOD,
        "pv_history_path": args.pv_history_path,
    }
    json.dump(params,
              open(f"{save_dir}/params.json",
                   "w"))

    # Read from file
    edges_history = pd.read_csv(args.edges_history_path, parse_dates=[3])
    if args.pv_history_path is not None:
        assert os.path.isfile(args.pv_history_path)
        pv_edges_history = pd.read_csv(args.pv_history_path, parse_dates=[3])
    else:
        pv_edges_history = None

    # Set weights
    edges_history = set_weights(edges_history, params["weights"])
    edges_history.to_csv(f"./{save_dir}/all_edges.csv", index=False)
    # Get periods
    all_dates = pd.date_range(min(edges_history["timestamp"]).date(),
                              max(edges_history["timestamp"]).date(), freq="D")
    all_dates = all_dates[range(0, len(all_dates), params["freq_days"])]
    LOGGER.info("Done.")

    json.dump([str(d.date()) for d in all_dates],
              open(f"{save_dir}/all_dates.json", "w"))

    all_results = {}
    window_edges = {}
    for i in range(len(all_dates)):
        start = all_dates[i]
        LOGGER.info(f"Iter: {i}, start: {str(start.date())}")
        if i == 0:
            prev_edges = None
        end = start + dt.timedelta(days=params["freq_days"] - 1, hours=23,
                                   minutes=59, seconds=59)
        period_edges = edges_history.loc[
                       (edges_history["timestamp"] >= start) & (
                               edges_history["timestamp"] <= end)]
        graph_edges = create_graph_edges_with_webbing(
            period_edges, period=i, end=end, prev_edges=prev_edges,
            beta=params["proba"]["beta"], gamma_f=params["proba"]["gamma_f"],
            gamma_b=params["proba"]["gamma_b"])
        prev_edges = graph_edges.copy()

        if pv_edges_history is not None:
            # Add pageviews edges
            period_pv_edges = pv_edges_history[
                (pv_edges_history["timestamp"] >= start) & (
                        pv_edges_history["timestamp"] <= end)].copy()
            period_pv_edges["norm_weight"] = period_pv_edges["weight"]
            graph_edges = pd.concat([graph_edges, period_pv_edges])
        graph_edges.drop(["proba", "weight"], axis=1, inplace=True)
        graph_edges.rename({"norm_weight": "weight"}, axis=1, inplace=True)
        graph = create_graph(graph_edges)
        window_edges[i] = graph_edges.copy()

        if i == 0:
            nstart = None
        personalization = personalization_mapper(
            params["personalization_method"], nstart)
        LOGGER.info(f"Running pagerank on graph with {len(graph.nodes)} "
                    f"nodes")
        pr, errors = pagerank(graph, nstart=nstart, alpha=1 - params["proba"][
            "alpha"], personalization=personalization)
        nstart = pr.copy()

        pr = pd.DataFrame(pd.Series(pr), columns=["rank"])
        pr = pr.sort_values(by="rank", ascending=False)
        if args.node_label_mapper:
            pr["label"] = [NODE_LABEL_MAPPER.get(node) for node in pr.index]
            all_results[str(start.date())] = pr.drop("label", axis=1).to_json()
        else:
            all_results[str(start.date())] = pr.to_json()
        LOGGER.debug(pr[list(map(is_contributor, pr.index))].head(20))

    LOGGER.info(f"Saving results to {save_dir}")
    json.dump(all_results,
              open(f"{save_dir}/pagerank.json", "w"))
