import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import time

from typing import Dict

from quantinar_rep.config import NODE_TYPE_MAPPER, LOGGER
from quantinar_rep.constants import USER_BASE, COURSE_BASE, COURSELET_BASE, \
    REVIEW_BASE, ORDER_BASE
from quantinar_rep.utils import epoch_weights, get_all_nodes_from_user_id


def plot_graph_pr_scale(graph, pagerank, scale=100000,
                        labels: Dict = None, save_path: str = None,
                        show: bool = False):
    """
    Plot the graph with node size defined by the given pagerank score

    Args:
        graph:
        pagerank:
        scale:
        labels:
        save_path:
        show:

    Returns:

    """
    # Order node list for easier visualisation
    seed = []
    orders = []
    reviews = []
    courses = []
    courselets = []
    users = []
    for n in list(graph.nodes):
        if n == "seed":
            seed.append(n)
        else:
            if NODE_TYPE_MAPPER.get(n.split("_")[0]) == ORDER_BASE:
                orders.append(n)
            elif NODE_TYPE_MAPPER.get(n.split("_")[0]) == REVIEW_BASE:
                reviews.append(n)
            elif NODE_TYPE_MAPPER.get(n.split("_")[0]) == COURSE_BASE:
                courses.append(n)
            elif NODE_TYPE_MAPPER.get(n.split("_")[0]) == COURSELET_BASE:
                courselets.append(n)
            elif NODE_TYPE_MAPPER.get(n.split("_")[0]) == USER_BASE:
                users.append(n)
            else:
                raise ValueError(n)
    nodelist = seed + orders + reviews + courses + courselets + users
    # Define node plot attributes
    node_size = pagerank.loc[nodelist, "rank"] / pagerank.loc[
        nodelist, "rank"].sum() * scale
    node_color = [graph.nodes[n]['color'] for n in nodelist]
    subgraph = graph.subgraph(nodelist)
    # pos = nx.spring_layout(subgraph) #  default spring_layout
    pos = nx.kamada_kawai_layout(subgraph)
    plt.figure(figsize=(20, 15))
    nx.draw(subgraph,
            pos=pos,
            nodelist=nodelist,
            node_color=node_color,
            node_size=node_size,
            alpha=.5,
            arrowstyle="->",
            labels=labels if labels is None else {n: labels.get(n) for n in
                                                  nodelist},
            font_size=15,
            with_labels=not (labels is None))
    plt.box(False)
    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_subgraph(graph: nx.Graph, score: pd.DataFrame, max_nodes: int = 100,
                  labels: Dict = None, save_path: str = None,
                  show: bool = False,  node_color: str = "red",
                  width: float = 0.5, alpha: float = 0.1,):
    """
    Plot the subgraph with top max_nodes score values.

    Args:
        graph:
        score:
        max_nodes:
        labels:
        save_path:
        show:
        node_color:
        width:
        alpha:

    Returns:

    """
    nodelist = list(graph.nodes)
    nodesize = score.loc[nodelist].sort_values(
        "rank", ascending=False).iloc[:max_nodes]
    nodelist = list(nodesize.index)
    # Rescaled node size
    nodesize = nodesize.loc[nodelist, "rank"] / nodesize.loc[
        nodelist, "rank"].sum() * 10000
    # Create a subgraph with selected nodes
    subgraph = graph.subgraph(nodelist)
    pos = nx.kamada_kawai_layout(subgraph)

    LOGGER.debug(f"Plotting {len(nodelist)} contributors. This might take "
                 f"a while...")

    plt.figure(figsize=(20, 15))
    t1 = time.time()
    _ = nx.draw_networkx_nodes(subgraph,
                               pos,
                               node_color=node_color,
                               nodelist=nodelist,
                               node_size=nodesize.values,
                               )
    if labels is not None:
        _ = nx.draw_networkx_labels(subgraph,
                                    pos,
                                    labels={n: labels.get(n) for n in
                                            nodelist},
                                    font_color='blue',
                                    font_size=15,
                                    )
    _ = nx.draw_networkx_edges(subgraph,
                               pos,
                               arrowstyle="->",
                               width=width,
                               nodelist=nodelist,
                               node_size=nodesize.values,
                               alpha=alpha,
                               )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    t2 = time.time()
    LOGGER.debug(f"Plot {len(nodelist)} contributors in "
                 f"{round((t2-t1)/60, 2)} mins.")
    plt.close()


def plot_user_rank(df_user_rank: pd.DataFrame, ylim: tuple = (40, 0),
                   save_path: str = None, labels: Dict = None,
                   show: bool = False):
    """
    Plot the rank timeserie of all users

    Args:
        df_user_rank:
        ylim:
        save_path:
        labels:
        show:

    Returns:

    """
    cols = df_user_rank.iloc[-1].sort_values().index.tolist()

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 12))
    for i, name in enumerate(cols):
        if not labels:
            text = f"user{i}"
        else:
            text = labels.get(name)
        axs.plot(df_user_rank[name], label=name)  # , c=c)
        axs.annotate(xy=(df_user_rank.index[-1], df_user_rank[name][-1]),
                     xytext=(33, 0), textcoords='offset points', text=text,
                     va='center')
        axs.set_ylim(ylim)
    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def epoch_importance_animation(user_id, result, savepath=None, show=False,
                               decay=0.95):
    """
    Plot the animation giving the relative epoch node importance for the
    given user.

    Args:
        user_id:
        result:
        savepath:
        show:
        decay:

    Returns:

    """
    epochs = list(result.keys())
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(figsize=(10, 5))
    user_rank = result[epochs[0]].loc[
        get_all_nodes_from_user_id(result[epochs[0]].index, user_id)]

    period_weights = epoch_weights(decay, 1)
    if len(user_rank):
        user_rank = user_rank[list(map(lambda x: "t" in x, user_rank.index))]
        ind = user_rank.index.tolist()
        ind.sort(key=lambda x: int(x.split("_")[-1][1:]))
        user_rank = (user_rank.loc[ind, "rank"] * period_weights[
            "weight"].values[-len(ind):])
        user_rank /= user_rank.sum()
        ind = [int(x.split("_")[-1][1:]) for x in ind]
        ind.sort()
    else:
        ind = 0
        user_rank = 0.

    line, = ax.plot(ind, user_rank, color='b')

    ax.set_ylim([0, 1.])
    _ = ax.set_xticks(range(0, len(epochs), 4))
    _ = ax.set_xticklabels(np.array(epochs)[range(0, len(epochs), 4)],
                           rotation=30)

    def update(num, x, y, line):
        k = epochs[num]
        user_rank = result[k].loc[
            get_all_nodes_from_user_id(result[k].index, user_id)]
        period_weights = epoch_weights(decay, num + 1)

        if len(user_rank):
            user_rank = user_rank[
                list(map(lambda x: "t" in x, user_rank.index))]
            ind = user_rank.index.tolist()
            ind.sort(key=lambda x: int(x.split("_")[-1][1:]))
            user_rank = (user_rank.loc[ind, "rank"] * period_weights[
                "weight"].values[-len(ind):])
            user_rank /= user_rank.sum()
            ind = [int(x.split("_")[-1][1:]) for x in ind]
            ind.sort()

        else:
            ind = num
            user_rank = 0.

        line.set_data(ind, user_rank)

        return line,

    ani = animation.FuncAnimation(fig, update, len(epochs), fargs=[x, y, line],
                                  interval=5, blit=True)
    if savepath:
        ani.save(savepath,
                 savefig_kwargs={'transparent': True, 'bbox_inches': 'tight'})
    if show:
        plt.show()
    plt.close()