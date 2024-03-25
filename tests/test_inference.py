"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""
from typing import NamedTuple
import unittest

import numpy as np
import jax
import jraph
import haiku as hk

from jraph_MPEU.inference import get_predictions


def summing_gnn(graphs):
    graph_net = jraph.GraphNetwork(
        update_edge_fn=None,
        update_node_fn=lambda nodes, senders, receivers, globals: nodes,
        aggregate_nodes_for_globals_fn=jraph.segment_sum,
        update_global_fn=lambda nodes, edges, globals: nodes
    )
    return graph_net(graphs)


def get_graph_dataset(num_graphs: int, latent_size: int):
    """Return a dummy dataset of graphs."""
    graphs = []
    for _ in range(num_graphs):
        n_node = np.random.randint(5, 10)
        graph = jraph.GraphsTuple(
            nodes=np.random.random_sample((n_node, latent_size)),
            edges=np.random.random_sample((n_node, latent_size)),
            receivers=np.arange(n_node), senders=np.arange(n_node),
            globals=None, n_node=np.array([n_node]), n_edge=np.array([0]))
        graphs.append(graph)
    return graphs


class TestInference(unittest.TestCase):
    """Test inference.py methods and classes."""
    def test_get_predictions(self):
        """Test case: no dropout, scalar predictions"""
        latent_size = 1
        dataset = get_graph_dataset(100, latent_size)
        dataset_copy = dataset[:]
        net = hk.transform_with_state(summing_gnn)
        params, state = net.init(jax.random.PRNGKey(42), dataset[0])
        predictions = get_predictions(dataset, net, params, state, 'scalar')

        predictions_expected = []
        for graph in dataset:
            prediction_expected = np.sum(graph.nodes)
            predictions_expected.append(prediction_expected)
        predictions_expected = np.array(predictions_expected)

        np.testing.assert_allclose(
            predictions[0], predictions_expected, rtol=1e-6)
        # test that the dataset was not modified in place
        for graph, graph_copy in zip(dataset, dataset_copy):
            np.testing.assert_array_equal(graph.nodes, graph_copy.nodes)
            np.testing.assert_array_equal(graph.edges, graph_copy.edges)
            np.testing.assert_array_equal(graph.globals, graph_copy.globals)

    def test_get_results_df(self):
        """Test getting the results dataframe."""
        raise NotImplementedError("TODO: implement this test")


if __name__ == '__main__':
    unittest.main()
