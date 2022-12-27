import json
import matplotlib.pyplot as plt
import os
import pandas as pd

from quantinar_rep.config import LOGGER, NODE_WEIGHTS
from quantinar_rep.credrank import get_historical_users_pagerank, \
    get_rank_from_pagerank, compute_period_minted_amount
from quantinar_rep.graph import create_graph, \
    normalize_children_edges_weight, \
    create_contributor_subgraph
from quantinar_rep.plot import plot_graph_pr_scale, plot_user_rank, \
    plot_subgraph, epoch_importance_animation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Pagerank output directory",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.95,
        help="Decay factor for credrank computation",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize legends in plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots",
    )
    parser.add_argument("--node_label_mapper", type=str,
                        help="Path to the node label mapper")

    args = parser.parse_args()
    if args.node_label_mapper is not None:
        node_label_mapper = json.load(open(args.node_label_mapper, "r"))
    else:
        node_label_mapper = None
    output_dir = args.output_dir
    decay = args.decay
    save_dir = f"{output_dir}/analysis"
    LOGGER.info(f"Saving results to {save_dir}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # # Load result
    LOGGER.info("Load result...")
    result = json.load(open(f"{output_dir}/pagerank.json", "r"))
    all_dates = pd.to_datetime(
        json.load(open(f"{output_dir}/all_dates.json", "r")))
    params = json.load(open(f"{output_dir}/params.json", "r"))
    freq_days = params["freq_days"]
    weights = params["weights"]
    all_edges = pd.read_csv(f"{output_dir}/all_edges.csv", parse_dates=[3])
    keys = list(result.keys())

    # Get historical user pagerank
    LOGGER.info("Get historical user pagerank...")
    users_pagerank, ts_users_pagerank = get_historical_users_pagerank(result,
                                                                      decay=decay)
    ts_users_pagerank = ts_users_pagerank.T[
        ~ts_users_pagerank.columns.duplicated()].T
    ts_users_rank = get_rank_from_pagerank(ts_users_pagerank)

    # Save output
    ts_users_pagerank.to_csv(f"{save_dir}/users_pagerank.csv")
    ts_users_rank.to_csv(f"{save_dir}/ts_users_rank.csv")

    LOGGER.info("Plot user rank...")
    if node_label_mapper is not None:
        save_path = f"{save_dir}/users_rank.png"
    else:
        save_path = f"{save_dir}/users_rank_ano.png"
    plot_user_rank(ts_users_rank, ylim=(40, 0), save_path=save_path,
                   labels=node_label_mapper, show=args.show)

    # # Get historical nodes contribution
    # ## Node weights mint
    LOGGER.info("Compute total nodes weights...")
    m_k = compute_period_minted_amount(result, NODE_WEIGHTS)
    m_k.to_csv(f"{save_dir}/total_qnar.csv")
    LOGGER.info("Done.")
    plt.plot(m_k)
    plt.savefig(f"{save_dir}/total_qnar.png", bbox_inches="tight",
                transparent=True)
    if args.show:
        plt.show()
    plt.close()

    plt.plot(m_k.cumsum())
    plt.savefig(f"{save_dir}/total_qnar_cum.png", bbox_inches="tight",
                transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # ## Cred Rank
    ts_credrank = (ts_users_pagerank / ts_users_pagerank.sum(
        1).values.reshape(-1, 1)) * m_k.values.reshape(
        -1, 1)
    rel_ts_credrank = ts_credrank / ts_credrank.sum(1).values.reshape(-1, 1)
    ts_credrank.to_csv(f"{save_dir}/users_credrank.csv")
    _ = plt.plot(ts_credrank)
    if node_label_mapper is not None:
        plt.legend([node_label_mapper[c] for c in ts_credrank.columns[:10]],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        save_path = f"{save_dir}/users_credrank.png"
    else:
        save_path = f"{save_dir}/users_credrank_ano.png"
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if args.show:
        plt.show()
    plt.close()

    _ = plt.plot(rel_ts_credrank)
    if node_label_mapper is not None:
        plt.legend([node_label_mapper[c] for c in rel_ts_credrank.columns[
                                                  :10]],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        save_path = f"{save_dir}/users_rel_credrank.png"
    else:
        save_path = f"{save_dir}/users_rel_credrank_ano.png"
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Cumulative cred
    cum_cred = ts_credrank.cumsum()
    cum_cred = cum_cred[cum_cred.iloc[-1].sort_values(ascending=False).index]
    plt.figure(figsize=(15, 6))
    for i, c in enumerate(cum_cred.columns):
        if i >= 15:
            break
        _ = plt.plot(cum_cred[c], label=c)
    if args.show:
        plt.show()
    plt.close()

    # # Plot graph
    i = 5
    k = keys[i]
    pagerank_k = pd.read_json(result[k])
    period_edges = all_edges[all_edges["timestamp"] < k]
    period_edges = normalize_children_edges_weight(period_edges)
    period_edges = period_edges.sort_values(by="timestamp")
    period_edges.drop(["proba", "weight"], axis=1, inplace=True)
    period_edges.rename({"norm_weight": "weight"}, axis=1, inplace=True)
    period_graph = create_graph(period_edges)

    if node_label_mapper is not None:
        save_path = f"{save_dir}/graph_{k.replace('-', '')}.png"
    else:
        save_path = f"{save_dir}/graph_{k.replace('-', '')}_ano.png"
    plot_graph_pr_scale(period_graph, pagerank_k, scale=100000,
                        labels=node_label_mapper, save_path=save_path,
                        show=args.show)
    # Filter edges
    simple_graph = create_contributor_subgraph(period_edges)
    if node_label_mapper is not None:
        save_path = f"{save_dir}/contributor_graph_{k.replace('-', '')}.png"
    else:

        save_path = f"{save_dir}/contributor_graph_" \
                    f"{k.replace('-', '')}_ano.png"
    plot_subgraph(simple_graph, pagerank_k, labels=node_label_mapper,
                  save_path=save_path, max_nodes=1000000, show=args.show)

    # # Plot last contributor graph
    i = len(keys) - 1
    k = keys[i]
    pagerank_k = pd.read_json(result[k])
    period_edges = all_edges[all_edges["timestamp"] < k]
    period_edges = normalize_children_edges_weight(period_edges)
    period_edges = period_edges.sort_values(by="timestamp")
    period_edges.drop(["proba", "weight"], axis=1, inplace=True)
    period_edges.rename({"norm_weight": "weight"}, axis=1, inplace=True)
    # period_graph = create_graph(period_edges)
    # if node_label_mapper is not None:
    #     save_path = f"{save_dir}/graph_{k.replace('-', '')}.png"
    # else:
    #     save_path = f"{save_dir}/graph_{k.replace('-', '')}_ano.png"
    # plot_graph_pr_scale(period_graph, pagerank_k, scale=100000,
    #                     labels=node_label_mapper, save_path=save_path,
    #                     show=args.show)
    # Filter edges
    simple_graph = create_contributor_subgraph(period_edges)
    if node_label_mapper is not None:
        save_path = f"{save_dir}/contributor_graph_{k.replace('-', '')}.png"
    else:

        save_path = f"{save_dir}/contributor_graph_" \
                    f"{k.replace('-', '')}_ano.png"
    plot_subgraph(simple_graph, pagerank_k, labels=node_label_mapper,
                  save_path=save_path, max_nodes=1000000, show=args.show)

    # Epoch importance
    for k in result:
        result[k] = pd.DataFrame(json.loads(result[k]))
    username, user_id = "Wolfgang Karl Härdle", \
                        "04c703b5-70c0-489b-8a49-5252a91165f8"
    save_path = f"{save_dir}/epoch_importance" \
                f"_{username.replace(' ', '')}.gif"
    epoch_importance_animation(user_id, result, savepath=save_path,
                               show=args.show)

    username, user_id = "Bruno Spilak", "7d6a9e86-356c-4514-8af1-d60f04fc8df8"
    save_path = f"{save_dir}/epoch_importance" \
                f"_{username.replace(' ', '')}.gif"
    epoch_importance_animation(user_id, result, savepath=save_path,
                               show=args.show)

    username, user_id = "Raul Christian Bag", \
                        "08205a8f-7093-4de3-b110-fb5fa0df7c83"
    save_path = f"{save_dir}/epoch_importance" \
                f"_{username.replace(' ', '')}.gif"
    epoch_importance_animation(user_id, result, savepath=save_path,
                               show=args.show)

    username, user_id = "Julian Winkel", "b8bbab2f-4ed2-4bcb-b4bc-e86072375c4f"
    save_path = f"{save_dir}/epoch_importance" \
                f"_{username.replace(' ', '')}.gif"
    epoch_importance_animation(user_id, result, savepath=save_path,
                               show=args.show)
