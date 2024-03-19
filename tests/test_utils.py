"""Test the functions in the utils module."""

import tempfile
import unittest

from parameterized import parameterized
import jraph
import numpy as np
import ml_collections

from jraph_MPEU.utils import (
    Config_iterator,
    save_config,
    load_config,
    normalize_targets,
    get_normalization_dict,
    scale_targets,
    estimate_padding_budget_for_batch_size,
    add_labels_to_graphs,
    update_config_fields,
    get_num_pairs,
    str_to_list
)


def get_random_graph(rng, n_features) -> jraph.GraphsTuple:
    """Return a random graphsTuple with n_node from 1 to 10 and global feature
    vector of size n_features."""
    graph = jraph.GraphsTuple(
        nodes=None,
        edges=None,
        senders=None,
        receivers=None,
        n_node=rng.integers(1, 10, (1,)),
        n_edge=None,
        globals=rng.random((n_features,)))
    return graph


def get_random_graph_list(rng, n_graphs, n_features):
    """Return a list of n_graphs jraph.GraphsTuple."""
    graphs = []
    for _ in range(n_graphs):
        graph = get_random_graph(rng, n_features)
        graphs.append(graph)
    return graphs


class TestUtilsFunctions(unittest.TestCase):
    """Testing class for utility functions."""
    def setUp(self):
        self.rng = np.random.default_rng()

    def test_estimate_padding_budget(self):
        """Test estimator by generating graph lists with different
        distributions of n_node and n_edge."""
        # first example has one outlier, so outlier should fit in budget
        single_graph_10 = jraph.GraphsTuple(
            n_node=np.asarray([10]), n_edge=np.asarray([10]),
            nodes=np.ones((10, 4)), edges=np.ones((10, 5)),
            globals=np.ones((1, 6)),
            senders=np.zeros((10)), receivers=np.zeros((10))
        )
        single_graph_100 = jraph.GraphsTuple(
            n_node=np.asarray([100]), n_edge=np.asarray([100]),
            nodes=np.ones((100, 4)), edges=np.ones((100, 5)),
            globals=np.ones((1, 6)),
            senders=np.zeros((100)), receivers=np.zeros((100))
        )
        dataset = [single_graph_10]*10
        dataset.append(single_graph_100)
        batch_size = 2
        num_estimation_graphs = 100
        estimate = estimate_padding_budget_for_batch_size(
            dataset, batch_size, num_estimation_graphs)
        self.assertEqual(estimate.n_node, 100)
        self.assertEqual(estimate.n_edge, 100)
        self.assertEqual(estimate.n_graph, batch_size)

        # second example has graphs of all same sizes
        dataset = [single_graph_10]*10
        batch_size = 2
        num_estimation_graphs = 100
        estimate = estimate_padding_budget_for_batch_size(
            dataset, batch_size, num_estimation_graphs)
        self.assertEqual(estimate.n_node, 64)
        self.assertEqual(estimate.n_edge, 64)
        self.assertEqual(estimate.n_graph, batch_size)

        # third example has one outlier, but it fits in normal budget
        dataset = [single_graph_10]*100
        dataset.append(single_graph_100)
        batch_size = 10
        num_estimation_graphs = 1000
        estimate = estimate_padding_budget_for_batch_size(
            dataset, batch_size, num_estimation_graphs)
        self.assertEqual(estimate.n_node, 128)
        self.assertEqual(estimate.n_edge, 128)
        self.assertEqual(estimate.n_graph, batch_size)

    def test_str_to_list(self):
        text = '[1  2 3 4]'
        res = str_to_list(text)
        list_expected = [1, 2, 3, 4]
        self.assertListEqual(res, list_expected)

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
        with tempfile.TemporaryDirectory() as test_dir:
            config = ml_collections.ConfigDict()
            config.a = 1
            config.b = 0.1
            config.c = 'c'
            config.array = [0, 1, 3]
            save_config(config, test_dir)

            config_loaded = load_config(test_dir)
            self.assertEqual(config_loaded.a, config.a)
            self.assertEqual(config_loaded.b, config.b)
            self.assertEqual(config_loaded.c, config.c)

    @parameterized.expand(['sum', 'per_atom_standard', 'mean', 'standard'])
    def test_get_normalization_dict(self, normalization_type):
        """Test if values of norm_dict have the right shape."""
        n_graphs = 10
        n_features = 4
        graphs = get_random_graph_list(self.rng, n_graphs, n_features)
        norm_dict = get_normalization_dict(graphs, normalization_type)
        self.assertTupleEqual(norm_dict['mean'].shape, (n_features,))
        self.assertTupleEqual(norm_dict['std'].shape, (n_features,))
        self.assertEqual(norm_dict['type'], normalization_type)

    @parameterized.expand(['mean', 'standard'])
    def test_normalize_targets_mean(self, normalization_type):
        """Test the normalization of targets when doing mean aggregation."""
        n_features = 4
        mean = np.array([1, 2, 3, 4])
        std = np.array([0, 1, 2, 3])
        n_graphs = 10 # number of graphs and labels to generate
        graphs = get_random_graph_list(self.rng, n_graphs, n_features)
        targets = np.array([graph.globals for graph in graphs])
        normalization = {
            'type': normalization_type,
            'mean': mean, 
            'std': std}
        norm_targets = normalize_targets(graphs, targets, normalization)

        self.assertTupleEqual(norm_targets.shape, (n_graphs, n_features))
        std = np.where(std==0, 1, std)
        norm_targets_expected = (targets - mean)/std
        np.testing.assert_array_almost_equal(
            norm_targets, norm_targets_expected)

    @parameterized.expand(['sum', 'per_atom_standard'])
    def test_normalize_targets_sum(self, normalization_type):
        """Test the normalization of targets when doing sum aggregation."""
        n_features = 4
        mean = np.array([1, 2, 3, 4])
        std = np.array([0, 1, 2, 3])
        n_graphs = 10 # number of graphs and labels to generate
        graphs = get_random_graph_list(self.rng, n_graphs, n_features)
        targets = np.array([graph.globals for graph in graphs])
        n_nodes = np.array([graph.n_node[0] for graph in graphs])

        normalization = {
            'type': normalization_type,
            'mean': mean, 
            'std': std}
        norm_targets = normalize_targets(graphs, targets, normalization)

        self.assertTupleEqual(norm_targets.shape, (n_graphs, n_features))
        std = np.where(std==0, 1, std)
        norm_targets_expected = (targets - np.outer(n_nodes, mean))/std
        np.testing.assert_array_almost_equal(
            norm_targets, norm_targets_expected)

    @parameterized.expand(['sum', 'per_atom_standard', 'mean', 'standard'])
    def test_norm_and_scale(self, normalization_type):
        """Test normalizing and scaling back targets and check for identity."""
        n_graph = 10 # number of graphs and labels to generate
        n_features = 4
        graphs = get_random_graph_list(self.rng, n_graph, n_features)
        targets = np.array([graph.globals for graph in graphs])

        norm_dict = get_normalization_dict(graphs, normalization_type)
        norm_targets = normalize_targets(graphs, targets, norm_dict)
        targets_rescaled = scale_targets(graphs, norm_targets, norm_dict)
        np.testing.assert_array_almost_equal(targets, targets_rescaled)

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
        np.testing.assert_equal(global_example, label_example)

        batch_generator = jraph.dynamically_batch(
            iter(graphs), budget.n_node, budget.n_edge, budget.n_graph)
        self.assertEqual(budget.n_graph, batch_size)
        self.assertEqual(budget.n_node, 192)
        self.assertEqual(budget.n_edge, 256)

        # check the sum of globals returned from batch generator
        global_sum = 0
        for batch in batch_generator:
            global_sum += sum(batch.globals)
        self.assertEqual(global_sum, sum(labels))


if __name__ == '__main__':
    unittest.main()
