"""Test the functions in the utils module."""

import tempfile
import unittest

import jax
import jax.numpy as jnp
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
    str_to_list,
    get_line_graph
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


def assert_graph_equal(graph, graph_expected):
    """Check that two graphs are similar."""
    np.testing.assert_array_equal(graph.nodes, graph_expected.nodes)
    np.testing.assert_array_equal(graph.edges, graph_expected.edges)
    np.testing.assert_array_equal(graph.n_node, graph_expected.n_node)
    np.testing.assert_array_equal(graph.globals, graph_expected.globals)
    np.testing.assert_array_equal(graph.senders, graph_expected.senders)
    np.testing.assert_array_equal(graph.receivers, graph_expected.receivers)


def _make_nest(array):
  """Returns a nest given an array."""
  return {'a': array,
          'b': [jnp.ones_like(array), {'c': jnp.zeros_like(array)}]}


class TestUtilsFunctions(unittest.TestCase):
    """Testing class for utility functions."""
    def test_get_line_graph(self):
        """Test the line graph function on different test graphs."""
        methane = jraph.GraphsTuple(
            nodes=np.asarray([[6], [1], [1], [1], [1]]),
            edges=np.asarray([[1]]*8),
            receivers=np.array([0, 0, 0, 0, 1, 2, 3, 4]),
            senders=np.array([1, 2, 3, 4, 0, 0, 0, 0]),
            globals=None,
            n_node=np.asarray([5]),
            n_edge=np.asarray([8])
        )
        line_graph_methane = get_line_graph(methane)
        line_graph_methane_expected = jraph.GraphsTuple(
            nodes=np.asarray([[1]]*8),
            edges=np.asarray([[6]]*12),
            receivers=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
            senders=np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
            n_node=np.asarray([8]),
            n_edge=np.asarray([12]),
            globals=None
        )
        assert_graph_equal(line_graph_methane, line_graph_methane_expected)

        chain = jraph.GraphsTuple(
            nodes=np.asarray([[6]]),
            edges=np.asarray([[2]]*2),
            receivers=np.array([0, 0]),
            senders=np.array([0, 0]),
            globals=None,
            n_node=np.asarray([1]),
            n_edge=np.asarray([2])
        )
        line_graph_chain = get_line_graph(chain)
        line_graph_chain_expected = jraph.GraphsTuple(
            nodes=np.asarray([[2]]*2),
            edges=np.asarray([[6]]*2),
            receivers=np.array([0, 1]),
            senders=np.array([1, 0]),
            n_node=np.asarray([2]),
            n_edge=np.asarray([2]),
            globals=None
        )
        assert_graph_equal(line_graph_chain, line_graph_chain_expected)

        # test batching and unbatching the line graphs
        batch = jraph.batch([methane, chain])
        line_batch = get_line_graph(batch)
        line_g1, line_g2 = jraph.unbatch(line_batch)
        assert_graph_equal(line_graph_methane, line_g1)
        assert_graph_equal(line_graph_chain, line_g2)


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
