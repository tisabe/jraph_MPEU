import jax
import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
from spektral.datasets import QM9

from typing import Generator, Mapping, Tuple
import unittest

from utils import *
import config
from model import Model
import graph_net_fn as gnf

def get_random_graph(key) -> jraph.GraphsTuple:
    graph = jraph.GraphsTuple(nodes=None,
                      edges=None,
                      senders=None,
                      receivers=None,
                      n_node=jax.random.randint(key, (1,), minval=1, maxval=10),
                      n_edge=None,
                      globals=None)
    return graph

class TestHelperFunctions(unittest.TestCase):
    def test_k_nn_print(self):
        position_matrix = np.array([[0, 0], [1, 1], [2, 2], [2, 3]])*1.0
        euclidian_distance = dist_matrix(position_matrix)

        #print(euclidian_distance)

        senders, receivers = get_knn_adj_from_dist(euclidian_distance, 5)
        #print(senders)
        #print(receivers)


    def test_k_nn_random(self):
        num_nodes = 5
        dimensions = 3
        k = 3
        position_matrix = np.random.randint(0, 10, size=(num_nodes, dimensions))
        distances = dist_matrix(position_matrix)
        #print(distances)
        senders, receivers = get_knn_adj_from_dist(distances, k)

        expected_senders = []
        expected_receivers = []

        for row in range(num_nodes):
            idx_list = []
            last_idx = 0
            for ik in range(k):
                min_val_last = 9999.9  # temporary last saved minimum value, initialized to high value
                for col in range(num_nodes):
                    if col == row or (col in idx_list):
                        continue    # do nothing on the diagonal, or if column has already been included
                    else:
                        val = distances[row, col]
                        if val < min_val_last:
                            min_val_last = val
                            last_idx = col
                idx_list.append(last_idx)
                expected_senders.append(last_idx)
                expected_receivers.append(row)

        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)

    '''
    def test_dist_matrix_random(self):
        # Test dist_matrix_batch with random positions matrix.
        num_nodes = 10
        dimensions = 3
        position_matrix = np.random.rand(num_nodes, dimensions)
        expected_euclidian_distance = np.zeros([num_nodes, num_nodes])
        # calculate differences manually for each entry of distance matrix
        for j in range(num_nodes):
            for k in range(num_nodes):
                difference_vector = position_matrix[j, :] - position_matrix[k, :]
                expected_euclidian_distance[j, k] = np.linalg.norm(difference_vector)
        euclidian_distance = dist_matrix(position_matrix)

        self.assertEqual(np.shape(expected_euclidian_distance), np.shape(euclidian_distance))
        np.testing.assert_allclose(euclidian_distance, expected_euclidian_distance)  # expected floating point error
    '''
    def test_get_cutoff_adj_from_dist_random(self):
        num_nodes = 10
        dimensions = 3
        cutoff = 0.7
        position_matrix = np.random.rand(num_nodes, dimensions)
        distances = dist_matrix(position_matrix)

        senders, receivers = get_cutoff_adj_from_dist(distances, cutoff)

        expected_senders = []
        expected_receivers = []

        for sender in range(num_nodes):
            for receiver in range(num_nodes):
                if not (sender == receiver):
                    if distances[sender, receiver] < cutoff:
                        expected_senders.append(sender)
                        expected_receivers.append(receiver)

        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)


    def test_reader_integration(self):
        file_str = 'QM9/graphs_U0K.csv'
        batch_size = 4
        inputs, outputs, auids = get_data_df_csv(file_str)

        train_in, test_in, train_out, test_out, train_auids, test_auids = sklearn.model_selection.train_test_split(
            inputs, outputs, auids, test_size=0.1, random_state=0)
        #print("train_out type: {}".format(type(train_out)))
        #print("train_out shape: {}".format(np.shape(train_out)))

        train_out, mean_train, std_train = normalize_targets(train_in, train_out)
        #print("train_out normd type: {}".format(type(train_out)))
        #print("train_out normd shape: {}".format(np.shape(train_out)))

        reader = DataReader(train_in, train_out, batch_size=batch_size)
        
        input_reader, output_reader = next(reader)

        input_reader = pad_graph_to_nearest_power_of_two(input_reader)
        in_type_reader = type(input_reader)
        out_type_reader = type(output_reader)
        #print("Reader Input n_node: {}".format(input_reader.n_node))
        #print(output_reader)
        #print("Reader Input type: {}".format(in_type_reader))
        #print("Reader output type: {}".format(out_type_reader))
        #print("Reader output shape: {}".format(np.shape(output_reader)))

        # replicate data processing in train_epoch of model
        graphs = []
        labels = []
        for i in range(batch_size):
            graphs.append(train_in[i])
            labels.append([train_out[i]]) # maybe [] around train_out[i]
        graph, label = jraph.batch(graphs), np.stack(labels)
        graph = pad_graph_to_nearest_power_of_two(graph)
        #print("Model input n_node: {}".format(graph.n_node))
        #print(label)
        #print("Model input type: {}".format(type(graph)))
        #print("Model output type: {}".format(type(label)))
        #print("Model output shape: {}".format(np.shape(label)))

        np.testing.assert_array_equal(input_reader.n_node, graph.n_node)
        self.assertEqual(first=in_type_reader, second=type(graph))
        self.assertEqual(first=out_type_reader, second=type(label))
        self.assertEqual(first=np.shape(output_reader), second=np.shape(label))

    def test_normalize_targets_avg(self):
        #print("testing normalization")
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
            #print(graphs[i])
            #print(labels[i])
        
        config.AVG_READOUT = True
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
        targets, mean, std = normalize_targets(graphs, labels)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, expected_mean)
        np.testing.assert_almost_equal(std, expected_std)
    
    def test_normalize_targets_sum(self):
        #print("testing normalization")
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
        
        #print(n_nodes)
        #print(labels)
        
        config.AVG_READOUT = False
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
        targets, mean, std = normalize_targets(graphs, labels)

        #print(expected_targets)
        #print(targets)
        #print("Expected mean: {}".format(expected_mean))
        #print("Actual mean: {}".format(mean))
        print(expected_std)
        print(std)

        np.testing.assert_almost_equal(targets, expected_targets, decimal=5)
        np.testing.assert_almost_equal(mean, float(expected_mean))
        np.testing.assert_almost_equal(std, expected_std)
    
    def test_scale_targets_avg(self):
        #print("testing normalization")
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
            #print(graphs[i])
            #print(labels[i])
        
        config.AVG_READOUT = True
        outputs, mean, std = normalize_targets(graphs, labels)
        outputs_scaled = scale_targets(graphs, outputs, mean, std)
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
            #print(graphs[i])
            #print(labels[i])
        labels = np.random.randint(low=0, high=10, size=(n,1)).astype(np.float32)
        
        config.AVG_READOUT = False
        outputs, mean, std = normalize_targets(graphs, labels)
        outputs_scaled = scale_targets(graphs, outputs, mean, std)
        
        np.testing.assert_almost_equal(np.array(labels), outputs_scaled)


    def test_shifted_softplus(self):
        n = 10
        res = gnf.shifted_softplus(jnp.zeros((n)))
        np.testing.assert_almost_equal(np.array(res), np.zeros(n))


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

