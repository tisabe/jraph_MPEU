import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk

from typing import Generator, Mapping, Tuple
import unittest

from utils import *


class TestHelperFunctions(unittest.TestCase):
    def test_k_nn_print(self):
        position_matrix = np.array([[0, 0], [1, 1], [2, 2], [2, 3]])*1.0
        euclidian_distance = dist_matrix(position_matrix)

        print(euclidian_distance)

        senders, receivers = get_knn_adj_from_dist(euclidian_distance, 5)
        print(senders)
        print(receivers)


    def test_k_nn_random(self):
        num_nodes = 5
        dimensions = 3
        k = 3
        position_matrix = np.random.randint(0, 10, size=(num_nodes, dimensions))
        distances = dist_matrix(position_matrix)
        print(distances)
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




        print(senders)
        print(np.array(expected_senders))
        print(receivers)
        print(np.array(expected_receivers))

        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)


    def test_dist_matrix_random(self):
        '''Test dist_matrix_batch with random positions matrix.'''
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


    def test_spektral_to_jraph(self):
        dataset = QM9(amount=1)
        graph, label = spektral_to_jraph(dataset[0])
        label_size = len(label)

        #print(graph)
        #print(label)
        print(graph.senders)
        print(type(graph.senders))




if __name__ == '__main__':
    unittest.main()

