"""Module to define utility functions used in training, evaluation and data
processing."""

import os
import time
import json
from ast import literal_eval
import logging
from typing import NamedTuple

import jax.numpy as jnp
import jraph
import numpy as np
import ml_collections
from sklearn.model_selection import ParameterGrid

from jraph_MPEU_configs.default import get_config


def update_config_fields(
        config: ml_collections.ConfigDict
    ) -> ml_collections.ConfigDict:
    """Update config with missing parameters taken from default config."""
    config_default = get_config()
    for field in config_default:
        if not field in config:
            config[field] = config_default[field]
    return config


def dict_to_config(config_dict: dict) -> ml_collections.ConfigDict:
    """Convert a dict with parameter fields to a ml_collections.ConfigDict."""
    config = ml_collections.ConfigDict()
    for key in config_dict.keys():
        config[key] = config_dict[key]
    return config


class Config_iterator:
    """Constructs an iterator for all combinations of elements in config.

    Essentially gives all the elements for a grid search.
    """
    def __init__(self, config: ml_collections.config_dict):
        config_dict = vars(config)['_fields']
        self.grid = iter(list(ParameterGrid(config_dict)))

    def __iter__(self):
        return self

    def __next__(self):
        config = dict_to_config(next(self.grid))
        return config

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


def save_config(config: ml_collections.ConfigDict, workdir: str):
    """"Save the config in a human and machine-readable json file.

    File will be saved as config.json at workdir directory.
    """
    config_dict = vars(config)['_fields']
    with open(os.path.join(workdir, 'config.json'), 'w') as config_file:
        json.dump(config_dict, config_file, indent=4, separators=(',', ': '))


def load_config(workdir: str) -> ml_collections.ConfigDict:
    """Load and return a config from the directory workdir."""
    with open(os.path.join(workdir, 'config.json'), 'r') as config_file:
        config_dict = json.load(config_file)
    config = dict_to_config(config_dict)
    return update_config_fields(config)


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


def normalize_targets_dict(graphs_dict, labels_dict, aggregation_type):
    """Wrapper for normalize targets, when graphs and labels are dicts."""
    # convert from dicts to arrays
    inputs = list(graphs_dict.values())
    outputs = list(labels_dict.values())

    res, mean, std = normalize_targets(inputs, outputs, aggregation_type)

    # convert results back into dict
    res_dict = {}
    for id_single, label in zip(labels_dict.keys(), res):
        res_dict[id_single] = label

    return res_dict, mean, std


def normalize_targets(inputs, outputs, aggregation_type):
    """Return normalized outputs, based on aggregation type.

    Also return mean and standard deviation for later rescaling.
    Inputs, i.e. graphs, are used to get number of atoms,
    for atom-wise scaling."""
    outputs = np.reshape(outputs, len(outputs)) # convert to 1-D array
    n_atoms = np.zeros(len(outputs)) # save all numbers of atoms in array
    for i in range(len(outputs)):
        n_atoms[i] = inputs[i].n_node[0]

    if aggregation_type == 'sum':
        scaled_targets = np.array(outputs)/n_atoms
    else:
        scaled_targets = outputs
    mean = np.mean(scaled_targets)
    std = np.std(scaled_targets)

    if aggregation_type == 'sum':
        return (outputs - (mean*n_atoms))/std, mean, std
    else:
        return (scaled_targets - mean)/std, mean, std


def scale_targets(inputs, outputs, mean, std, aggregation_type):
    '''Return scaled targets. Inverse of normalize_targets,
    scales targets back to the original size.
    Args:
        inputs: list of jraph.GraphsTuple, to get number of atoms in graphs
        outputs: 1D-array of normalized target values
        mean: mean of original targets
        std: standard deviation of original targets
        aggregation_type: type of aggregation function
    Returns:
        numpy.array of scaled target values
    '''
    outputs = np.reshape(outputs, len(outputs))
    if aggregation_type == 'sum':
        n_atoms = np.zeros(len(outputs)) # save all numbers of atoms in array
        for i in range(len(outputs)):
            n_atoms[i] = inputs[i].n_node
        return np.reshape((outputs * std) + (n_atoms * mean), (len(outputs), 1))
    else:
        return (outputs * std) + mean


def _nearest_bigger_power_of_two(num: int) -> int:
    """Computes the nearest power of two greater than num for padding."""
    base = 2
    while base < num:
        base *= 2
    return base


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
        num_estimation_graphs: int
) -> GraphsTupleSize:
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
    """Return a list of jraph.GraphsTuple with the labels as globals."""
    graphs_with_globals = []
    for graph, label in zip(graphs, labels):
        graph_new = graph
        graph_new = graph_new._replace(globals=np.array([label]))
        graphs_with_globals.append(graph_new)
    return graphs_with_globals


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(
        globals=jnp.zeros([graphs.n_node.shape[0], 1]))


def get_valid_mask(
        graphs: jraph.GraphsTuple
) -> jnp.ndarray:
    """Return a boolean mask of batched graphsTuple graphs.

    Returned array has shape [total_num_graphs, 1], values are True for real
    graphs and False for padding graphs.
    """
    graph_mask = jraph.get_graph_padding_mask(graphs)

    return jnp.expand_dims(graph_mask, 1)


class Time_logger:
    """Class to keep track of time spent per gradient step. WIP"""
    def __init__(self, config):
        self.start = time.time()
        self.step_max = config.num_train_steps_max
        self.time_limit = config.time_limit

    def log_eta(self, step):
        """Log the estimated time of arrival, with the maximum number of steps."""
        time_elapsed = time.time() - self.start
        eta = int(time_elapsed * (self.step_max/step - 1))
        logging.info(f'step {step}, ETA: {eta//3600}h{(eta%3600)//60}m')

    def get_time_stop(self):
        """Return whether the time is over the time limit."""
        time_elapsed = time.time() - self.start
        if self.time_limit is not None:
            return (time_elapsed/3600) > self.time_limit
        else:
            return False
