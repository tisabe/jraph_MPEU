import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk
import functools
import optax

from typing import Generator, Mapping, Tuple

from spektral.datasets import QM9


def dist_matrix(position_matrix):
    """Return the pairwise distance matrix of positions in euclidian space.

    See this link:
    https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow

    We multiply the position matrices together. And then reduce the sum by summing
    over the distance to the origin origin axis of the squared position matrix.

    Args:
      position_matrix: Position matrix given as an array.
        Size: (number of positions, dimension of positions)

    Returns:
      Euclidian distances between nodes as a jnp array.
        Size: (number of positions, number of positions)
    """

    row_norm_squared = jnp.sum(position_matrix * position_matrix, axis=-1)
    # Turn r into column vector
    row_norm_squared = jnp.reshape(row_norm_squared, [-1, 1])
    # 2*pos*potT
    distance_matrix = 2 * jnp.matmul(position_matrix, jnp.transpose(position_matrix))
    # Stick this equation into our overleaf.
    distance_matrix = row_norm_squared + jnp.transpose(row_norm_squared) - distance_matrix
    distance_matrix = jnp.abs(distance_matrix)  # to avoid negative numbers before sqrt
    return jnp.sqrt(distance_matrix)


def get_cutoff_adj_from_dist(distance_matrix, cutoff):
    '''Return the adjacency matrix in the form of senders and receivers from a distance matrix and a cutoff.'''

    #print(distance_matrix)
    assert(len(np.shape(distance_matrix)) == 2) # distance matrix cannot be batched

    bool_dist = distance_matrix < cutoff
    senders, receivers = jnp.nonzero(bool_dist)

    return senders, receivers


def get_knn_adj_from_dist(distance_matrix, k):
    '''Return the adjacency matrix with k-nearest neighbours in form of senders and receivers from a distance matrix'''
    n_pos = len(distance_matrix)
    tops, nn = jax.lax.top_k(distance_matrix, k)

    rows = jnp.linspace(0, n_pos, num=k * n_pos, endpoint=False)
    rows = jnp.int16(jnp.floor(rows))
    nn = nn.flatten()
    receivers = rows
    senders = nn

    return senders, receivers


def spektral_to_jraph(graph_s: spektral.data.graph.Graph, cutoff=10) -> jraph.GraphsTuple:
    '''Return graph in jraph format and separate label (globals) from graph'''
    nodes = graph_s.x
    edges = graph_s.e
    #globals = graph_s.y
    globals = [graph_s.y[2]]

    # construct adjacency matrix from positions in node features
    distances = dist_matrix_batch(nodes[:, 5:8])
    senders, receivers = get_adj_from_dist(distances, cutoff=cutoff)


    graph_j = jraph.GraphsTuple(
        n_node=np.asarray([len(nodes)]),
        n_edge=np.asarray([len(edges)]),
        nodes=nodes, edges=edges,
        globals=None,
        senders=senders, receivers=receivers)

    return graph_j, globals


class DataReader:
    """Data Reader for QM9 dataset from Spektral library. Inspired by ogb_example in Jraph."""

    def __init__(self, dataset_in, batch_size=1):
        self._dataset = dataset_in
        self._repeat = False
        self._batch_size = batch_size
        self._generator = self._make_generator()
        self.total_num_graphs = len(self._dataset)
        # initial non-shuffled, non-split list of indices
        self._ids = np.arange(self.total_num_graphs)

    def repeat(self):
        self._repeat = True

    def shuffle(self):
        # shuffle the index array
        np.random.shuffle(self._ids)

    def __next__(self):
        graphs = []
        labels = []
        for _ in range(self._batch_size):
            graph, label = next(self._generator)
            graphs.append(graph)
            labels.append(label)
        # return jraph.batch(graphs), np.concatenate(labels, axis=0)
        return jraph.batch(graphs), np.stack(labels)

    def get_graph_by_idx(self, idx):
        """Gets a graph by an integer index."""
        # Gather the graph information
        graph_s = self._dataset[idx]
        return spektral_to_jraph(graph_s)

    def _make_generator(self):
        """Makes a single example generator."""
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if not self._repeat:
                if idx == self.total_num_graphs:
                    return
            else:
                # This will reset the index to 0 if we are at the end of the dataset.
                idx = idx % self.total_num_graphs

            graph, label = self.get_graph_by_idx(int(self._ids[idx]))
            idx += 1
            yield graph, label


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(
        graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                                 pad_graphs_to)



