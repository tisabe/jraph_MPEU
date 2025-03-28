"""Test loss functions in loss module."""

import numpy as np
import jraph

from jraph_MPEU.loss import WeightedGlobalsNodesEdgesLoss


def get_uniform_graph(n_features, factor) -> jraph.GraphsTuple:
    n_node = np.random.randint(1, 10, (1,))
    n_edge = np.random.randint(1, 20, (1,))
    graph = jraph.GraphsTuple(
        nodes=np.ones((n_node[0], n_features))*factor,
        edges=np.ones((n_edge[0], n_features))*factor,
        senders=np.random.randint(0, n_node[0], (n_edge[0],)),
        receivers=np.random.randint(0, n_node[0], (n_edge[0],)),
        n_node=n_node,
        n_edge=n_edge,
        globals={'test_key': np.ones((n_features,))*factor}
    )
    return graph


def test_WeightedGlobalsNodesEdgesLoss():
    loss = WeightedGlobalsNodesEdgesLoss(3, 2, 1)
    graph = get_uniform_graph(10, 1)
    print(loss(graph, graph))

def test_padded_graph():
    # NOTE: when graph is not padded with jraph.pad_with_graphs,
    # then number of padding graphs/nodes/edges is equal to
    # total number of graphs/nodes/edges
    return
    

if __name__ == "__main__":
    test_WeightedGlobalsNodesEdgesLoss()