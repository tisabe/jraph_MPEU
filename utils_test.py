"""Test the functions in the utils module."""

import tempfile
from typing import Generator, Mapping, Tuple
import unittest

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
import ml_collections

from utils import *


def get_random_graph(key) -> jraph.GraphsTuple:
    graph = jraph.GraphsTuple(nodes=None,
                      edges=None,
                      senders=None,
                      receivers=None,
                      n_node=jax.random.randint(key, (1,), minval=1, maxval=10),
                      n_edge=None,
                      globals=None)
    return graph

class TestUtilsFunctions(unittest.TestCase):
    def test_save_and_load_config(self):
        self.test_dir = tempfile.TemporaryDirectory()
        config = ml_collections.ConfigDict()
        config.a = 1
        config.b = 0.1
        config.c = 'c'
        config.array = [0, 1, 3]
        save_config(config, self.test_dir.name)

        config_loaded = load_config(self.test_dir.name)
        self.assertEqual(config_loaded.a, config.a)
        self.assertEqual(config_loaded.b, config.b)
        self.assertEqual(config_loaded.c, config.c)


    def test_normalize_targets_mean(self):
        n = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        for i in range(n):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            graphs.append(graph)
            labels.append(jax.random.uniform(subkey))

        aggregation_type = 'mean'
        expected_targets = np.zeros(n)
        expected_mean = 0
        expected_std = 0
        for i in range(n):
            expected_mean += labels[i]
        expected_mean = expected_mean/n
        
        for i in range(n):
            expected_std += np.square(labels[i] - expected_mean)
        expected_std = np.sqrt(expected_std/n)

        for i in range(n):
            expected_targets[i] = (labels[i] - expected_mean)/expected_std
        
        # calculate function values
        targets, mean, std = normalize_targets(graphs, labels, aggregation_type)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, expected_mean)
        np.testing.assert_almost_equal(std, expected_std)
    
    def test_normalize_targets_sum(self):
        n = 1000 # number of graphs and labels to generate
        graphs = []
        labels = []
        n_nodes = []
        seed = 1
        key = jax.random.PRNGKey(seed)
        for i in range(n):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            n_nodes.append(int(graph.n_node))
            graphs.append(graph)
            labels.append(np.float32(jax.random.uniform(subkey)))
        
        aggregation_type = 'sum'
        expected_targets = np.zeros(n)
        expected_mean = 0
        expected_std = 0
        for i in range(n):
            expected_mean += labels[i]/graphs[i].n_node
        expected_mean = expected_mean/n
        
        for i in range(n):
            expected_std += np.square(labels[i]/graphs[i].n_node - expected_mean)
        expected_std = np.sqrt(expected_std/n)
        
        for i in range(n):
            expected_targets[i] = (labels[i] - graphs[i].n_node*expected_mean)/expected_std
        
        # calculate function values
        targets, mean, std = normalize_targets(graphs, labels, aggregation_type)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, float(expected_mean))
        np.testing.assert_almost_equal(std, expected_std)
    
    def test_scale_targets_mean(self):
        n = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        for i in range(n):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            graphs.append(graph)
            labels.append(jax.random.uniform(subkey))

        aggregation_type = 'mean'
        outputs, mean, std = normalize_targets(graphs, labels, aggregation_type)
        outputs_scaled = scale_targets(graphs, outputs, mean, std, aggregation_type)
        np.testing.assert_almost_equal(np.array(labels), outputs_scaled)

    def test_scale_targets_sum(self):
        print("testing rescaling")
        n = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        n_atoms = []
        for i in range(n):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            n_atoms.append(graph.n_node)
            graphs.append(graph)
            
        labels = np.random.randint(low=0, high=10, size=(n,1)).astype(np.float32)
        
        aggregation_type = 'sum'
        outputs, mean, std = normalize_targets(graphs, labels, aggregation_type)
        outputs_scaled = scale_targets(graphs, outputs, mean, std, aggregation_type)
        
        np.testing.assert_almost_equal(np.array(labels), outputs_scaled)

    def test_get_atomization_energies_QM9(self):
        graph = jraph.GraphsTuple(nodes=[0,1,2,3,4],
                      edges=None,
                      senders=None,
                      receivers=None,
                      n_node=[5],
                      n_edge=None,
                      globals=None)
        #print(graph)
        graphs = [graph]
        labels = [0]
        outputs = get_atomization_energies_QM9(graphs, labels, 'U0')
        print(outputs)

    def test_dynamic_batch_budget(self):
        n = 1000 # number of graphs
        graph = jraph.GraphsTuple(nodes=np.asarray([0,1,2,3,4]),
                      edges=np.ones((6,2)),
                      senders=None,
                      receivers=None,
                      n_node=np.asarray([5]),
                      n_edge=np.asarray([6]),
                      globals=None)
        graphs = [graph] * n
        
        batch_size = 32
        budget = estimate_padding_budget_for_batch_size(graphs, batch_size,
            num_estimation_graphs=100)

        # calculate the average number of nodes, edges and graphs in a batch
        expected_n_node = 192 # 32 * 5 + 32 (rounded up by 32)
        expected_n_edge = 256 # 32 * 6 + 64 (rounded up by 64)
        expected_n_graph = 32
        self.assertEqual(budget.n_node, expected_n_node)
        self.assertEqual(budget.n_edge, expected_n_edge)
        self.assertEqual(budget.n_graph, expected_n_graph)

    def test_dynamic_batch_labels(self):
        n = 1000 # number of graphs
        graph = jraph.GraphsTuple(nodes=np.asarray([0,1,2,3,4]),
                      edges=np.ones((6,2)),
                      senders=np.array([0, 1]), 
                      receivers=np.array([2, 2]),
                      n_node=np.asarray([5]),
                      n_edge=np.asarray([6]),
                      globals=None)
        graphs = [graph] * n
        labels = [i for i in range(n)]
        
        batch_size = 32
        budget = estimate_padding_budget_for_batch_size(graphs, batch_size,
            num_estimation_graphs=100)
        graphs = add_labels_to_graphs(graphs, labels)

        graph_example = graphs[10]
        global_example = graph_example.globals
        label_example = labels[10]
        print(global_example)
        print(label_example)
        np.testing.assert_equal(global_example, label_example)

        batch_generator = jraph.dynamically_batch(iter(graphs), 
            budget.n_node,
            budget.n_edge,
            budget.n_graph)
        print(budget.n_node)
        print(budget.n_edge)
        print(budget.n_graph)

        global_sum = 0
        for batch in batch_generator:
            global_sum += sum(batch.globals)
        print(global_sum)
        print(sum(labels))


    




if __name__ == '__main__':
    unittest.main()

