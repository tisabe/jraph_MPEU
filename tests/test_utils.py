"""Test the functions in the utils module."""

import tempfile
import unittest

import jax
import jraph
import numpy as np
import ml_collections

from jraph_MPEU.utils import (
    Config_iterator,
    save_config,
    load_config,
    normalize_targets,
    normalize_targets_dict,
    scale_targets,
    estimate_padding_budget_for_batch_size,
    add_labels_to_graphs,
    update_config_fields,
    get_num_pairs,
    str_to_list
)


def get_random_graph(key) -> jraph.GraphsTuple:
    """Return a random graphsTuple using a jax random key."""
    graph = jraph.GraphsTuple(
        nodes=None,
        edges=None,
        senders=None,
        receivers=None,
        n_node=jax.random.randint(key, (1,), minval=1, maxval=10),
        n_edge=None,
        globals=None)
    return graph


class TestUtilsFunctions(unittest.TestCase):
    """Testing class for utility functions."""
    def test_str_to_list(self):
        text = '[1  2 3 4]'
        res = str_to_list(text)
        print(res)
        print(type(res))
        for i in res:
            print(type(i))

    def test_get_num_pairs(self):
        """Test function get_num_pairs."""
        numbers = [1, 1, 1, 1, 4, 4, 5]
        pairs = get_num_pairs(numbers)
        pairs_expected = [
            [1, 1],
            [1, 4],
            [1, 5],
            [4, 1],
            [4, 4],
            [4, 5],
            [5, 1],
            [5, 4],
            [5, 5]
        ]
        np.testing.assert_array_equal(pairs, pairs_expected)
        # with numpy array
        numbers = np.array([1, 1, 1, 1, 4, 4, 5])
        pairs = get_num_pairs(numbers)
        np.testing.assert_array_equal(pairs, pairs_expected)
        # test single number
        numbers = [1]
        pairs = get_num_pairs(numbers)
        pairs_expected = [[1, 1]]
        np.testing.assert_array_equal(pairs, pairs_expected)


    def test_update_config_fields(self):
        """Test updateing a test config with values from the default config."""
        config = ml_collections.ConfigDict()
        config.one = 1
        config.letter = 'c'
        config = update_config_fields(config)
        # test that old entries stayed the same
        self.assertEqual(config.one, 1)
        self.assertEqual(config.letter, 'c')
        # test that new fields were written
        self.assertEqual(config.label_str, 'U0')
        self.assertEqual(config.latent_size, 64)

    def test_config_iterator(self):
        """Test config_iterator by producing config grid and printing them."""
        config = ml_collections.ConfigDict()
        config.list = [1, 2]
        config.list2 = ['d', 'e', 'f']
        config.single = ['c']
        iterator = Config_iterator(config)

        configs = list(iterator)
        self.assertEqual(configs[0].list, 1)
        self.assertEqual(configs[0].list2, 'd')
        self.assertEqual(configs[0].single, 'c')

        self.assertEqual(configs[1].list, 1)
        self.assertEqual(configs[1].list2, 'e')
        self.assertEqual(configs[1].single, 'c')

        self.assertEqual(configs[2].list, 1)
        self.assertEqual(configs[2].list2, 'f')
        self.assertEqual(configs[2].single, 'c')

        self.assertEqual(configs[3].list, 2)
        self.assertEqual(configs[3].list2, 'd')
        self.assertEqual(configs[3].single, 'c')

        self.assertEqual(configs[4].list, 2)
        self.assertEqual(configs[4].list2, 'e')
        self.assertEqual(configs[4].single, 'c')

        self.assertEqual(configs[5].list, 2)
        self.assertEqual(configs[5].list2, 'f')
        self.assertEqual(configs[5].single, 'c')

    def test_save_and_load_config(self):
        """Test saving and loading a config.json file by saving and loading."""
        test_dir = tempfile.TemporaryDirectory()
        config = ml_collections.ConfigDict()
        config.a = 1
        config.b = 0.1
        config.c = 'c'
        config.array = [0, 1, 3]
        save_config(config, test_dir.name)

        config_loaded = load_config(test_dir.name)
        self.assertEqual(config_loaded.a, config.a)
        self.assertEqual(config_loaded.b, config.b)
        self.assertEqual(config_loaded.c, config.c)

    def test_normalize_targets_dict(self):
        """Test the normalization of targets for dict inputs."""
        n_graph = 10 # number of graphs and labels to generate
        graphs = {}
        labels = {}
        seed = 0
        key = jax.random.PRNGKey(seed)
        for i in range(n_graph):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            graphs[i] = graph
            labels[i] = jax.random.uniform(subkey)

        aggregation_type = 'mean'
        expected_targets = np.zeros(n_graph)
        expected_mean = 0
        expected_std = 0
        for i in range(n_graph):
            expected_mean += labels[i]
        expected_mean = expected_mean/n_graph

        for i in range(n_graph):
            expected_std += np.square(labels[i] - expected_mean)
        expected_std = np.sqrt(expected_std/n_graph)

        for i in range(n_graph):
            expected_targets[i] = (labels[i] - expected_mean)/expected_std

        # calculate function values
        targets, mean, std = normalize_targets_dict(
            graphs, labels, aggregation_type)
        targets = list(targets.values())

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, expected_mean)
        np.testing.assert_almost_equal(std, expected_std)

    def test_normalize_targets_mean(self):
        """Test the normalization of targets when doing mean aggregation."""
        n_graph = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        for i in range(n_graph):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            graphs.append(graph)
            labels.append(jax.random.uniform(subkey))

        aggregation_type = 'mean'
        expected_targets = np.zeros(n_graph)
        expected_mean = 0
        expected_std = 0
        for i in range(n_graph):
            expected_mean += labels[i]
        expected_mean = expected_mean/n_graph

        for i in range(n_graph):
            expected_std += np.square(labels[i] - expected_mean)
        expected_std = np.sqrt(expected_std/n_graph)

        for i in range(n_graph):
            expected_targets[i] = (labels[i] - expected_mean)/expected_std

        # calculate function values
        targets, mean, std = normalize_targets(graphs, labels, aggregation_type)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, expected_mean)
        np.testing.assert_almost_equal(std, expected_std)

    def test_normalize_targets_sum(self):
        """Test the normalization of targets when doing sum aggregation."""
        n_graph = 1000 # number of graphs and labels to generate
        graphs = []
        labels = []
        n_nodes = []
        seed = 1
        key = jax.random.PRNGKey(seed)
        for i in range(n_graph):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(key)
            n_nodes.append(int(graph.n_node))
            graphs.append(graph)
            labels.append(np.float32(jax.random.uniform(subkey)))

        aggregation_type = 'sum'
        expected_targets = np.zeros(n_graph)
        expected_mean = 0
        expected_std = 0
        for i in range(n_graph):
            expected_mean += labels[i]/graphs[i].n_node
        expected_mean = expected_mean/n_graph

        for i in range(n_graph):
            expected_std += np.square(labels[i]/graphs[i].n_node - expected_mean)
        expected_std = np.sqrt(expected_std/n_graph)

        for i in range(n_graph):
            expected_targets[i] = (labels[i] - graphs[i].n_node*expected_mean)/expected_std

        # calculate function values
        targets, mean, std = normalize_targets(graphs, labels, aggregation_type)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, float(expected_mean))
        np.testing.assert_almost_equal(std, expected_std)

    def test_scale_targets_mean(self):
        """Test scaling targets back to original values with mean
        aggregation."""
        n_graph = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        for _ in range(n_graph):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(subkey)
            graphs.append(graph)
            labels.append(jax.random.uniform(subkey))

        aggregation_type = 'mean'
        outputs, mean, std = normalize_targets(graphs, labels, aggregation_type)
        outputs_scaled = scale_targets(graphs, outputs, mean, std, aggregation_type)
        np.testing.assert_almost_equal(np.array(labels), outputs_scaled)

    def test_scale_targets_sum(self):
        """Test scaling targets back to original values with sum
        aggregation."""
        n_graph = 10 # number of graphs and labels to generate
        graphs = []
        labels = []
        seed = 0
        key = jax.random.PRNGKey(seed)
        n_atoms = []
        for _ in range(n_graph):
            key, subkey = jax.random.split(key)
            graph = get_random_graph(subkey)
            n_atoms.append(graph.n_node)
            graphs.append(graph)

        labels = np.random.randint(
            low=0, high=10, size=(n_graph, 1)).astype(np.float32)

        aggregation_type = 'sum'
        outputs, mean, std = normalize_targets(graphs, labels, aggregation_type)
        outputs_scaled = scale_targets(graphs, outputs, mean, std, aggregation_type)

        np.testing.assert_almost_equal(np.array(labels), outputs_scaled)

    def test_dynamic_batch_budget(self):
        """Test dynamic batching budget of graphs by looking at sizes of the
        graphs."""
        n_graph = 1000 # number of graphs
        graph = jraph.GraphsTuple(
            nodes=np.asarray([0, 1, 2, 3, 4]),
            edges=np.ones((6, 2)),
            senders=None,
            receivers=None,
            n_node=np.asarray([5]),
            n_edge=np.asarray([6]),
            globals=None)
        graphs = [graph] * n_graph

        batch_size = 32
        budget = estimate_padding_budget_for_batch_size(
            graphs, batch_size, num_estimation_graphs=100)

        # calculate the average number of nodes, edges and graphs in a batch
        expected_n_node = 192 # 32 * 5 + 32 (rounded up by 32)
        expected_n_edge = 256 # 32 * 6 + 64 (rounded up by 64)
        expected_n_graph = 32
        self.assertEqual(budget.n_node, expected_n_node)
        self.assertEqual(budget.n_edge, expected_n_edge)
        self.assertEqual(budget.n_graph, expected_n_graph)

    def test_dynamic_batch_labels(self):
        """Test dynamic batching of the graph labels with checksums."""
        n_graph = 1000 # number of graphs
        graph = jraph.GraphsTuple(
            nodes=np.asarray([0, 1, 2, 3, 4]),
            edges=np.ones((6, 2)),
            senders=np.array([0, 1]),
            receivers=np.array([2, 2]),
            n_node=np.asarray([5]),
            n_edge=np.asarray([6]),
            globals=None)
        graphs = [graph] * n_graph
        labels = [i for i in range(n_graph)]

        batch_size = 32
        budget = estimate_padding_budget_for_batch_size(
            graphs, batch_size, num_estimation_graphs=100)
        graphs = add_labels_to_graphs(graphs, labels)

        graph_example = graphs[10]
        global_example = graph_example.globals
        label_example = labels[10]
        print(global_example)
        print(label_example)
        np.testing.assert_equal(global_example, label_example)

        batch_generator = jraph.dynamically_batch(
            iter(graphs), budget.n_node, budget.n_edge, budget.n_graph)
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
