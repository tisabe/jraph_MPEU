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
    def test_k_nn(self):
        position_matrix = np.array([[0, 0], [1, 1], [2, 2], [2, 3]])*1.0
        euclidian_distance = dist_matrix(position_matrix)

        print(euclidian_distance)

        n_pos = len(euclidian_distance)
        k = 3 # cutoff number of neighbours
        tops, nn = jax.lax.top_k(euclidian_distance, k)
        print(nn)

        rows = jnp.linspace(0, n_pos, num=k*n_pos, endpoint=False)
        print(rows)
        rows = jnp.int16(jnp.floor(rows))
        print(rows)
        nn = nn.flatten()
        print(nn)


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
                if distances[sender, receiver] < cutoff:
                    expected_senders.append(sender)
                    expected_receivers.append(receiver)

        '''print(np.array(senders))
        print(receivers)
        print(np.array(expected_senders))
        print(np.array(expected_receivers))'''

        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)


if __name__ == '__main__':
    unittest.main()

