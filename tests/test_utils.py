"""Test the functions in the utils module."""

import os
import tempfile
import unittest

from parameterized import parameterized
import jax
import jraph
import numpy as np
import ml_collections

from jraph_MPEU.utils import (
    ConfigIterator,
    save_config,
    load_config,
    normalize_targets,
    get_normalization_dict,
    normalize_graph_globals,
    scale_targets,
    estimate_padding_budget_for_batch_size,
    pad_graph_to_nearest_power_of_two,
    add_labels_to_graphs,
    update_config_fields,
    get_num_pairs,
    str_to_list,
    save_norm_dict,
    load_norm_dict,
    dist_matrix,
    _nearest_bigger_power_of_two,
    replace_globals,
    get_valid_mask
)


def get_random_graph_no_edge(rng, n_features) -> jraph.GraphsTuple:
    """Return a random graphsTuple with n_node from 1 to 10 and global feature
    vector of size n_features and no edges."""
    graph = jraph.GraphsTuple(
        nodes=None,
        edges=None,
        senders=None,
        receivers=None,
        n_node=rng.integers(1, 10, (1,)),
        n_edge=None,
        globals=rng.random((n_features,)))
    return graph

def get_random_graph(rng, n_features) -> jraph.GraphsTuple:
    """Return a random graphsTuple with n_node from 1 to 10 and global feature
    vector of size n_features."""
    n_node = rng.integers(1, 10, (1,))
    n_edge = rng.integers(1, 20, (1,))
    graph = jraph.GraphsTuple(
        nodes=None,
        edges=None,
        senders=rng.integers(0, n_node[0], (n_edge[0],)),
        receivers=rng.integers(0, n_node[0], (n_edge[0],)),
        n_node=n_node,
        n_edge=n_edge,
        globals=rng.random((n_features,)))
    return graph


def get_random_graph_list(rng, n_graphs, n_features, make_edges=True):
    """Return a list of n_graphs jraph.GraphsTuple."""
    graphs = []
    for _ in range(n_graphs):
        if make_edges:
            graph = get_random_graph(rng, n_features)
        else:
            graph = get_random_graph_no_edge(rng, n_features)
        graphs.append(graph)
    return graphs


class TestUtilsFunctions(unittest.TestCase):
    """Testing class for utility functions."""
    def setUp(self):
        self.rng = np.random.default_rng()

    def test_replace_globals(self):
        """Test that globals are replaced by zeros."""
        graph = get_random_graph(self.rng, 4)
        graph = replace_globals(graph)
        np.testing.assert_array_equal(graph.globals, [[0]])

        n_graphs = 10
        graphs = get_random_graph_list(self.rng, n_graphs, 4)
        graph = jraph.batch(graphs)
        graph = replace_globals(graph)
        np.testing.assert_array_equal(graph.globals, [[0]]*n_graphs)

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

        with self.assertRaises(ValueError):
            # batch size has to be larger than 1
            estimate = estimate_padding_budget_for_batch_size(
                dataset, 1, num_estimation_graphs)
        with self.assertRaises(ValueError):
            # dataset cannot contain batched graphs
            estimate = estimate_padding_budget_for_batch_size(
                dataset+[jraph.batch(dataset[:3])],
                batch_size, num_estimation_graphs)

    def test_pad_graph_to_nearest_power_of_two(self):
        """Test pad_graph_to_nearest_power_of_two function."""
        n_features = 4
        n_graphs = 100
        graphs = get_random_graph_list(self.rng, n_graphs, n_features)
        for graph in graphs:
            graph_padded = pad_graph_to_nearest_power_of_two(graph)
            self.assertEqual(sum(graph_padded.n_node)%2, 1)
            self.assertEqual(sum(graph_padded.n_edge)%2, 0)
            self.assertTupleEqual(graph_padded.n_node.shape, (2,))
            self.assertTupleEqual(graph_padded.n_edge.shape, (2,))
            mask = get_valid_mask(graph_padded)
            np.testing.assert_array_equal(mask, [[True], [False]])

    def test_str_to_list(self):
        """Test for str_to_list function."""
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
        """Test ConfigIterator by producing config grid and checking it."""
        config = ml_collections.ConfigDict()
        config.list = [1, 2]
        config.list2 = ['d', 'e', 'f']
        config.single = ['c']
        iterator = ConfigIterator(config)

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

    def test_normalization_raises(self):
        """Test that an incompatible normalization type raises ValueError."""
        normalization_type = 'test'
        n_graphs = 10
        n_features = 4
        graphs = get_random_graph_list(self.rng, n_graphs, n_features)
        with self.assertRaises(ValueError):
            _ = get_normalization_dict(graphs, normalization_type)
        normalization = {
            'type': normalization_type,
            'mean': np.array([1, 2, 3, 4]), 
            'std': np.array([0, 1, 2, 3])}
        targets = np.array([graph.globals for graph in graphs])
        with self.assertRaises(ValueError):
            _ = normalize_targets(graphs, targets, normalization)
        with self.assertRaises(ValueError):
            _ = scale_targets(graphs, targets, normalization)


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

    @parameterized.expand(
        ['sum', 'per_atom_standard', 'mean', 'standard',
        'scalar_non_negative'])
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

    def test_normalize_graph_globals_mean(self):
        """Test normalization of targets inside globals of graphs."""
        n_graph = 10
        n_features = 4
        mean = np.array([1, 2, 3, 4])
        std = np.array([1, 2, 3, 4])
        normalization = {
            'type': 'standard',
            'mean': mean, 
            'std': std}
        graphs = get_random_graph_list(self.rng, n_graph, n_features)
        graphs_norm = normalize_graph_globals(graphs, normalization)
        for graph_norm, graph in zip(graphs_norm, graphs):
            self.assertTupleEqual(graph_norm.globals.shape, (n_features,))
            np.testing.assert_array_almost_equal(
                graph_norm.globals, (graph.globals-mean)/std)

    def test_normalize_graph_globals_sum(self):
        """Test normalization of targets inside globals of graphs."""
        n_graph = 10
        n_features = 4
        mean = np.array([1, 2, 3, 4])
        std = np.array([1, 2, 3, 4])
        normalization = {
            'type': 'per_atom_standard',
            'mean': mean, 
            'std': std}
        graphs = get_random_graph_list(self.rng, n_graph, n_features)
        graphs_norm = normalize_graph_globals(graphs, normalization)
        for graph_norm, graph in zip(graphs_norm, graphs):
            self.assertTupleEqual(graph_norm.globals.shape, (n_features,))
            np.testing.assert_array_almost_equal(
                graph_norm.globals, (graph.globals-graph.n_node[0]*mean)/std)

    def test_save_load_norm_dict(self):
        """Test saving and loading of a dummy normalization dict."""
        with tempfile.TemporaryDirectory() as test_dir:
            norm_path = os.path.join(test_dir, 'norm.json')
            norm_dict = {'string': 'test', 'np.array': np.array([1.,2.])}
            save_norm_dict(norm_dict, norm_path)
            norm_dict_load = load_norm_dict(norm_path)
            np.testing.assert_array_equal(
                norm_dict['np.array'], norm_dict_load['np.array'])
            self.assertEqual(norm_dict['string'], norm_dict_load['string'])
        
        # test empty dict return when no norm_dict was saved
        norm_dict_load = load_norm_dict('norm.json')
        self.assertEqual(norm_dict_load, {})

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

    def test_dist_matrix(self):
        """Test dist_matrix function manually with small list of positions."""
        pos = np.array([
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        expected = np.sqrt(np.array([
            [0, 1, 1, 1],
            [1, 0, 2, 2],
            [1, 2, 0, 2],
            [1, 2, 2, 0]
        ]))
        dist = dist_matrix(pos)
        np.testing.assert_array_almost_equal(dist, expected)

    @parameterized.expand([
        (1, 2), (2, 2), (42, 64), (111, 128)
    ])
    def test_nearest_bigger_power_of_two(self, input_val, expected):
        """Test _nearest_bigger_power_of_two with examples."""
        self.assertEqual(_nearest_bigger_power_of_two(input_val), expected)


if __name__ == '__main__':
    unittest.main()
