"""Test loss functions in loss module."""

import numpy as np
import jax
import jraph

from jraph_MPEU.loss import WeightedGlobalsNodesEdgesLoss


def get_uniform_graph(
    n_node=4, n_edge=8, n_features=10, factor=1, node_feats=True, edge_feats=True
) -> jraph.GraphsTuple:
    n_node = np.array([n_node])
    n_edge = np.array([n_edge])
    nodes=np.ones((n_node[0], n_features))*factor if node_feats else None
    edges=np.ones((n_edge[0], n_features))*factor if edge_feats else None
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=np.random.randint(0, n_node[0], (n_edge[0],)),
        receivers=np.random.randint(0, n_node[0], (n_edge[0],)),
        n_node=n_node,
        n_edge=n_edge,
        globals={'test_key': np.ones((1, n_features))*factor}
    )
    return graph


def get_nested_graph(n_node=4, n_edge=8, n_features=10, factor=1) -> jraph.GraphsTuple:
    n_node = np.array([n_node])
    n_edge = np.array([n_edge])
    graph = jraph.GraphsTuple(
        nodes={
            'node_key_0': np.ones((n_node[0], n_features))*factor,
            'node_key_1': np.ones((n_node[0], n_features))*factor
        },
        edges={
            'edge_key_0': np.ones((n_edge[0], n_features))*factor,
            'edge_subkey': {
                'edge_key_0': np.ones((n_edge[0], n_features))*factor,
                'edge_key_1': np.ones((n_edge[0], n_features))*factor
            }
        },
        senders=np.random.randint(0, n_node[0], (n_edge[0],)),
        receivers=np.random.randint(0, n_node[0], (n_edge[0],)),
        n_node=n_node,
        n_edge=n_edge,
        globals={'test_key': np.ones((1, n_features))*factor}
    )
    return graph


def test_WeightedGlobalsNodesEdgesLoss_zero():
    """Test graph loss on unpadded graph."""
    loss = WeightedGlobalsNodesEdgesLoss(3, 2, 1)
    graph = get_uniform_graph(10, 1)
    assert loss(graph, graph) == 0


def test_WeightedGlobalsNodesEdgesLoss_zero_pad():
    """Test graph loss on unpadded graph."""
    loss = WeightedGlobalsNodesEdgesLoss(3, 2, 1)
    graph = get_uniform_graph(4, 8, 10, 1, edge_feats=False)
    graph = jraph.pad_with_graphs(graph, 8, 16)
    assert loss(graph, graph) == 0


def test_WeightedGlobalsNodesEdgesLoss_zero_batch():
    """Test graph loss on batched graph."""
    loss = WeightedGlobalsNodesEdgesLoss(3, 2, 1)
    graph = get_uniform_graph(10, 1)
    graphs = jraph.batch([graph, graph])
    assert loss(graphs, graphs) == 0


def test_nested_graph_loss():
    globals_weight, nodes_weight, edges_weight = 1, 1, 1
    loss = WeightedGlobalsNodesEdgesLoss(
        globals_weight, nodes_weight, edges_weight
    )
    n_node, n_edge, n_feat = 1, 1, 10
    graph_0 = get_nested_graph(n_node, n_edge, n_feat, factor=1)
    graph_1 = get_nested_graph(n_node, n_edge, n_feat, factor=2)

    loss_globals = globals_weight * 1 * n_feat
    loss_nodes = nodes_weight * 2 * n_feat / n_node
    loss_edges = edges_weight * 3 * n_feat / n_edge
    loss_total = loss_globals + loss_nodes + loss_edges

    assert loss(graph_0, graph_1) == loss(graph_1, graph_0)
    assert loss(graph_0, graph_1) == loss_total

    graph_0 = jraph.pad_with_graphs(graph_0, 2, 2)
    graph_1 = jraph.pad_with_graphs(graph_1, 2, 2)

    assert loss(graph_0, graph_1) == loss(graph_1, graph_0)
    assert loss(graph_0, graph_1) == loss_total


def test_graph_loss_jit():
    globals_weight, nodes_weight, edges_weight = 1, 1, 1
    loss = WeightedGlobalsNodesEdgesLoss(
        globals_weight, nodes_weight, edges_weight
    )
    n_node, n_edge, n_feat = 1, 1, 10
    graph_0 = get_nested_graph(n_node, n_edge, n_feat, factor=1)
    graph_1 = get_nested_graph(n_node, n_edge, n_feat, factor=2)
    loss = jax.jit(loss)
    loss_globals = globals_weight * 1 * n_feat
    loss_nodes = nodes_weight * 2 * n_feat / n_node
    loss_edges = edges_weight * 3 * n_feat / n_edge
    loss_total = loss_globals + loss_nodes + loss_edges

    assert loss(graph_0, graph_1) == loss(graph_1, graph_0)
    assert loss(graph_0, graph_1) == loss_total


if __name__ == "__main__":
    test_WeightedGlobalsNodesEdgesLoss_zero()
    test_WeightedGlobalsNodesEdgesLoss_zero_pad()
    test_WeightedGlobalsNodesEdgesLoss_zero_batch()
    test_nested_graph_loss()
    test_graph_loss_jit()