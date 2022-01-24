import jax
import jax.numpy as jnp
import jraph
import numpy as np
import haiku as hk
import functools
import optax
from ast import literal_eval
import sklearn
import pandas

import time
import logging

import config
from typing import Dict, Generator, Mapping, Tuple, NamedTuple

def str_to_array(str_array):
    '''Return a numpy array converted from a single string, representing an array.'''
    return np.array(literal_eval(str_array))


def str_to_array_replace(str_array):
    '''Return a numpy array converted from a single string, representing an array,
    while replacing spaces with commas.'''
    str_array = str_array.replace('  ', ' ')
    str_array = str_array.replace('[ ', '[')
    str_array = str_array.replace(' ', ',')
    return np.array(literal_eval(str_array))


def str_to_array_float(str_array):
    '''Return a numpy array converted from a single string, representing an array,
    while replacing spaces with commas.'''
    for _ in range(10):
        str_array = str_array.replace('  ', ' ')
    str_array = str_array.replace('[ ', '[')
    str_array = str_array.replace(' ', ',')
    return np.array(literal_eval(str_array))


def dist_matrix(position_matrix):
    '''Return the pairwise distance matrix of positions in euclidian space.

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
    '''

    row_norm_squared = jnp.sum(position_matrix * position_matrix, axis=-1)
    # Turn r into column vector
    row_norm_squared = jnp.reshape(row_norm_squared, [-1, 1])
    # 2*pos*potT
    distance_matrix = 2 * jnp.matmul(position_matrix, jnp.transpose(position_matrix))
    distance_matrix = row_norm_squared + jnp.transpose(row_norm_squared) - distance_matrix
    distance_matrix = jnp.abs(distance_matrix)  # to avoid negative numbers before sqrt
    return jnp.sqrt(distance_matrix)


def fill_diagonal(a, val):
    '''Set all elements of the main diagonal in matrix a to val.
    a can have batch dimensions.

    Jax numpy version of fill_diagonal in numpy. See:
    https://github.com/google/jax/issues/2680#issuecomment-804269672'''
    # TODO: more docstring
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def set_diag_high(mat, value=9999.9):
    '''Set the values on diagonal of matrix mat to a high value (default 9999.9). '''
    return fill_diagonal(mat, value)


def get_cutoff_adj_from_dist(distance_matrix, cutoff):
    '''Return the adjacency matrix in the form of senders and receivers from a distance matrix and a cutoff.'''

    distance_matrix = set_diag_high(distance_matrix) # set diagonal to delete self-edges
    assert(len(np.shape(distance_matrix)) == 2) # distance matrix cannot be batched

    bool_dist = distance_matrix < cutoff
    senders, receivers = jnp.nonzero(bool_dist)

    return senders, receivers


def get_knn_adj_from_dist(distance_matrix, k):
    '''Return the adjacency matrix with k-nearest neighbours in form of senders and receivers from a distance matrix'''

    distance_matrix = set_diag_high(distance_matrix)  # set diagonal to delete self-edges

    n_pos = len(distance_matrix)
    k = min([n_pos-1, k]) # function should return all neighbours if k >= number of neighbours-1 in graph
    # so the number of neighbours is always smaller than number of positions, since self edges are not included
    tops, nn = jax.lax.top_k(-1*distance_matrix, k)

    rows = jnp.linspace(0, n_pos, num=k * n_pos, endpoint=False)
    rows = jnp.int16(jnp.floor(rows))
    nn = nn.flatten()
    receivers = rows
    senders = nn

    return senders, receivers


def normalize_targets(inputs, outputs):
    '''return normalized outputs, based on which aggregation function is saved in config.
    Also return mean and standard deviation for later rescaling.
    Inputs, i.e. graphs, are used to get number of atoms, for atom-wise scaling.'''
    outputs = np.reshape(outputs, len(outputs)) # convert to 1-D array
    n_atoms = np.zeros(len(outputs)) # save all numbers of atoms in array
    for i in range(len(outputs)):
        n_atoms[i] = inputs[i].n_node[0]
    
    if config.AVG_READOUT:
        scaled_targets = outputs
    else:
        scaled_targets = np.array(outputs)/n_atoms
    mean = np.mean(scaled_targets)
    std = np.std(scaled_targets)

    if config.AVG_READOUT:
        return (scaled_targets - mean)/std, mean, std
    else:
        return (outputs - (mean*n_atoms))/std, mean, std

def normalize_targets_config(inputs, outputs, config):
    '''return normalized outputs, based on which aggregation function is saved in config.
    Also return mean and standard deviation for later rescaling.
    Inputs, i.e. graphs, are used to get number of atoms, for atom-wise scaling.'''
    outputs = np.reshape(outputs, len(outputs)) # convert to 1-D array
    n_atoms = np.zeros(len(outputs)) # save all numbers of atoms in array
    for i in range(len(outputs)):
        n_atoms[i] = inputs[i].n_node[0]
    
    if config.avg_aggregation_readout:
        scaled_targets = outputs
    else:
        scaled_targets = np.array(outputs)/n_atoms
    mean = np.mean(scaled_targets)
    std = np.std(scaled_targets)

    if config.avg_aggregation_readout:
        return (scaled_targets - mean)/std, mean, std
    else:
        return (outputs - (mean*n_atoms))/std, mean, std

def scale_targets(inputs, outputs, mean, std):
    '''Return scaled targets. Inverse of normalize_targets, 
    scales targets back to the original size.
    Args:
        inputs: list of jraph.GraphsTuple, to get number of atoms in graphs
        outputs: 1D-array of normalized target values
        mean: mean of original targets
        std: standard deviation of original targets
    Returns:
        numpy.array of scaled target values 
    '''
    outputs = np.reshape(outputs, len(outputs))
    if config.AVG_READOUT:
        return (outputs * std) + mean
    else:
        n_atoms = np.zeros(len(outputs)) # save all numbers of atoms in array
        for i in range(len(outputs)):
            n_atoms[i] = inputs[i].n_node
        return np.reshape((outputs * std) + (n_atoms * mean), (len(outputs), 1))

index_to_str = ['A','B','C','mu','alpha','homo','lumo','gap',
    'r2','zpve','U0','U','H','G','Cv',
    'U0_atom','U_atom','H_atom','G_atom']

def get_data_df_csv(file_str, label_str="U0", include_no_edge_graphs=False):
    '''Import data from a pandas.DataFrame saved as csv. 
    Return as inputs, outputs (e.g. train_inputs, train_outputs).

    Graphs will have one label, specified by label_str.

    If include_no_edge_graphs=False (default), graphs without edges will be skipped.
    Otherwise, they will be included (which might break the GNN).
    '''
    df = pandas.read_csv(file_str)
    inputs = []
    outputs = []
    auids = []

    label_index = index_to_str.index(label_str)
    # iterate over rows in dataframe and append inputs and outputs
    for index, row in df.iterrows():
        nodes = str_to_array_replace(row['nodes'])
        auid = row['auid']
        senders = str_to_array_replace(row['senders'])
        receivers = str_to_array_replace(row['receivers'])
        edges = str_to_array_float(row['edges'])
        print(row['label'])
        labels = str_to_array_replace(row['label'])
        label = label[label_index]

        if (not len(edges)==0) or include_no_edge_graphs:
            graph = jraph.GraphsTuple(
                n_node=np.asarray([len(nodes)]),
                n_edge=np.asarray([len(senders)]),
                nodes=nodes, edges=edges,
                globals=None,
                senders=np.asarray(senders), receivers=np.asarray(receivers))
            inputs.append(graph)
            outputs.append(label)
            auids.append(auid)

    return inputs, outputs, auids

def get_atomization_energies_QM9(graphs, labels, label_str):
    '''Return the atomization energies in the QM9 dataset by subtracting
    reference energies from atomref_QM9.txt.
    
    Args:
        graphs: list of jraph.GraphsTuple
        labels: list of labels (regression targets)
        label_str: name of the label, one of the attributes in atomref_QM9.txt

    '''
    ref_energies_df = pandas.read_csv('atomref_QM9.txt')
    outputs = []

    index_to_str = ['A','B','C','mu','alpha','homo','lumo','gap',
    'r2','zpve','U0','U','H','G','Cv',
    'U0_atom','U_atom','H_atom','G_atom']

    if label_str in index_to_str:
        for graph, label in zip(graphs, labels):
            # sum up the reference energies for the specific graph
            atomic_energy = sum(ref_energies_df[label_str][graph.nodes])
            outputs.append(label - atomic_energy)
        return outputs
    else:
        for graph, label in zip(graphs, labels):
            outputs.append(label)
        return outputs
    


class DataReader:
    """Data Reader data generated by datahandler. Inspired by ogb_example in Jraph."""

    def __init__(self, dataset_in, dataset_out, batch_size=1):
        self._dataset = dataset_in
        self._dataset_out = dataset_out
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
        "Returns batches of graphs, wrapping around and shuffling if at the end of the dataset."
        graphs = []
        labels = []
        for _ in range(self._batch_size):
            graph, label = next(self._generator)
            graphs.append(graph)
            labels.append([label])
        # return jraph.batch(graphs), np.concatenate(labels, axis=0)
        return jraph.batch(graphs), np.stack(labels)

    def get_graph_by_idx(self, idx):
        """Gets a graph by an integer index."""
        # Gather the graph information
        graph = self._dataset[idx]
        label = self._dataset_out[idx]
        return graph, label

    def _make_generator(self):
        """Returns a generator that returns single graphs."""
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if idx == self.total_num_graphs:
                if not self._repeat:
                    return
                else:
                    self.shuffle()
                    idx = 0
            

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

class GraphsTupleSize(NamedTuple):
  """Helper class to represent padding and graph sizes."""
  n_node: int
  n_edge: int
  n_graph: int

def get_graphs_tuple_size(graph: jraph.GraphsTuple):
  """Returns the number of nodes, edges and graphs in a GraphsTuple."""
  return GraphsTupleSize(
      n_node=np.sum(graph.n_node),
      n_edge=np.sum(graph.n_edge),
      n_graph=np.shape(graph.n_node)[0])

      
def estimate_padding_budget_for_batch_size(
    dataset,
    batch_size: int,
    num_estimation_graphs: int) -> GraphsTupleSize:
  """Estimates the padding budget for a dataset of unbatched GraphsTuples.
  Args:
    dataset: A dataset of unbatched GraphsTuples.
    batch_size: The intended batch size. Note that no batching is performed by
      this function.
    num_estimation_graphs: How many graphs to take from the dataset to estimate
      the distribution of number of nodes and edges per graph.
  Returns:
    padding_budget: The padding budget for batching and padding the graphs
    in this dataset to the given batch size.
  """

  def next_multiple_of_64(val: float):
    """Returns the next multiple of 64 after val."""
    return 64 * (1 + int(val // 64))

  if batch_size <= 1:
    raise ValueError('Batch size must be > 1 to account for padding graphs.')

  total_num_nodes = 0
  total_num_edges = 0
  for graph in dataset[:num_estimation_graphs]:
    graph_size = get_graphs_tuple_size(graph)
    if graph_size.n_graph != 1:
      raise ValueError('Dataset contains batched GraphTuples.')

    total_num_nodes += graph_size.n_node
    total_num_edges += graph_size.n_edge

  num_nodes_per_graph_estimate = total_num_nodes / num_estimation_graphs
  num_edges_per_graph_estimate = total_num_edges / num_estimation_graphs

  padding_budget = GraphsTupleSize(
      n_node=next_multiple_of_64(num_nodes_per_graph_estimate * batch_size),
      n_edge=next_multiple_of_64(num_edges_per_graph_estimate * batch_size),
      n_graph=batch_size)
  return padding_budget

def add_labels_to_graphs(graphs, labels):
    '''Return a list of jraph.GraphsTuple with the labels as globals.'''
    graphs_with_globals = []
    for graph, label in zip(graphs, labels):
        graph_new = graph
        graph_new = graph_new._replace(globals = np.array([label]))
        graphs_with_globals.append(graph_new)
    return graphs_with_globals

def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Replaces the globals attribute with a constant feature for each graph."""
  return graphs._replace(
      globals=jnp.zeros([graphs.n_node.shape[0], 1]))

def get_valid_mask(labels: jnp.ndarray,
                   graphs: jraph.GraphsTuple) -> jnp.ndarray:
  graph_mask = jraph.get_graph_padding_mask(graphs)

  return jnp.expand_dims(graph_mask, 1)

class Time_logger:
    def __init__(self, config):
        self.start = time.time()
        self.step_max = config.num_train_steps_max

    def log_eta(self, step):
        time_elapsed = time.time() - self.start
        eta = int(time_elapsed * (self.step_max/step - 1))
        logging.info(f'step {step}, ETA: {eta//3600}h{(eta%3600)//60}m')




