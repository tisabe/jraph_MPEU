"""Module to define utility functions used in training, evaluation and data
processing."""

import os
import json
import pickle
from typing import NamedTuple, List, Dict

import jax.numpy as jnp
import jraph
import numpy as np
import ml_collections
from sklearn.model_selection import ParameterGrid

from jraph_MPEU_configs.default import get_config


def get_num_pairs(numbers):
    """Return unique pairs of numbers from list numbers."""
    unique = set(list(numbers))
    pairs = []
    for i in unique:
        for j in unique:
            pairs.append([i, j])
    return pairs


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


class ConfigIterator:
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


def str_to_list(text):
    """Convert a string into a list of integers, by removing [] and ' '."""
    return list(map(int, filter(None, text.lstrip('[').rstrip(']').split(' '))))


def save_config(config: ml_collections.ConfigDict, workdir: str):
    """"Save the config in a human and machine-readable json file.

    File will be saved as config.json at workdir directory.
    """
    config_dict = vars(config)['_fields']
    with open(os.path.join(workdir, 'config.json'), 'w', encoding="utf-8") as config_file:
        json.dump(config_dict, config_file, indent=4, separators=(',', ': '))


def load_config(workdir: str) -> ml_collections.ConfigDict:
    """Load and return a config from the directory workdir."""
    with open(os.path.join(workdir, 'config.json'), 'r', encoding="utf-8") as config_file:
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
    row_norm_squared = np.sum(position_matrix * position_matrix, axis=-1)
    # Turn r into column vector
    row_norm_squared = np.reshape(row_norm_squared, [-1, 1])
    # 2*pos*potT
    distance_matrix = 2 * np.matmul(position_matrix, np.transpose(position_matrix))
    distance_matrix = row_norm_squared + np.transpose(row_norm_squared) - distance_matrix
    distance_matrix = np.abs(distance_matrix)  # to avoid negative numbers before sqrt
    return np.sqrt(distance_matrix)


def get_normalization_dict(graphs, normalization_type):
    """Return the normalization dict given list of graphs and normalization
    type.
    
    Args:
      graphs: list of jraph.GraphsTuple with targets as global feature
      normalization_type: string that describes which normalization to apply
    
    Returns:
      dict with values for normalization (e.g. mean, standard deviation of 
      features)
    """
    targets = np.array([graph.globals['target'] for graph in graphs])

    match normalization_type:
        case 'per_atom_standard'|'sum':
            n_nodes = np.array([graph.n_node for graph in graphs])
            scaled_targets = targets/n_nodes
            norm_dict = {
                "type": normalization_type,
                "mean": np.mean(scaled_targets, axis=0),
                "std": np.std(scaled_targets, axis=0)}
        case 'standard'|'mean':
            norm_dict = {
                "type": normalization_type,
                "mean": np.mean(targets, axis=0),
                "std": np.std(targets, axis=0)}
        case 'scalar_non_negative':
            # this type is meant for fitting band gaps, or other non-zero targets,
            # when the output of the last layer is also non-negative
            norm_dict = {
                "type": normalization_type,
                "std": np.std(targets, axis=0)}
        case _:
            raise ValueError(
                f"Unrecognized normalization type: {normalization_type}")
    return norm_dict


def save_norm_dict(norm_dict, path):
    """Save norm_dict at path."""
    with open(path, 'w+', encoding="utf-8") as file:
        json.dump(pickle.dumps(norm_dict).decode('latin-1'), file)


def load_norm_dict(path):
    """Load saved norm_dict at path."""
    with open(path, 'r', encoding="utf-8") as file:
        norm_dict = pickle.loads(json.load(file).encode('latin-1'))
    return norm_dict


def normalize_targets(
    graphs: List[jraph.GraphsTuple], targets: np.array, normalization: Dict
    ) -> List[float]:
    """Return normalized targets, based on normalization type.

    Args:
      graphs: list of jraph.GraphsTuple
      targets: target values pre normalization, np.array of shape (N, F)
      normalization: dict that contains values and description string to apply
        normalization

    Returns:
      normalized targets, np.array of shape (N, F)
    """
    norm_type = normalization['type']
    if 'mean' in normalization:
        mean = normalization['mean']
    if 'std' in normalization:
        std = normalization['std']
        # prevent division by zero
        std = np.where(std==0, 1, std)

    match norm_type:
        case 'per_atom_standard'|'sum':
            n_nodes = np.array([graph.n_node[0] for graph in graphs])
            return (targets - np.outer(n_nodes, mean))/std
        case 'standard'|'mean':
            return (targets - mean)/std
        case 'scalar_non_negative':
            return targets/std
        case _:
            raise ValueError(f"Unrecognized normalization type: {norm_type}")


def normalize_graph_globals(
    graphs: List[jraph.GraphsTuple], normalization: Dict
    ) -> List[jraph.GraphsTuple]:
    """Return list of graphs with normalized globals according to normalization."""
    # TODO(dts): change the global dict here.

    targets = np.array([graph.globals['target'] for graph in graphs])
    targets_norm = normalize_targets(graphs, targets, normalization)
    graphs_norm = []
    for graph, target in zip(graphs, targets_norm):
        globals = graph.globals
        globals['target'] = target
        graph = graph._replace(globals=globals)
        graphs_norm.append(graph)
    return graphs_norm


def scale_targets(graphs, targets, normalization):
    """Return scaled targets. Inverse of normalize_targets,
    scales targets back to the original size.
    Args:
        graphs: list of jraph.GraphsTuple, to get number of atoms in graphs
        targets: array of normalized target values, np.array of shape (N, F)
        normalization: dict that contains values and description string to apply
            normalization
    Returns:
        numpy.array of scaled target values, np.array of shape (N, F)
    """
    norm_type = normalization['type']
    if 'mean' in normalization:
        mean = normalization['mean']
    if 'std' in normalization:
        std = normalization['std']
        std = np.where(std==0, 1, std)

    match norm_type:
        case 'per_atom_standard'|'sum':
            n_nodes = np.array([graph.n_node[0] for graph in graphs])
            return targets*std + np.outer(n_nodes, mean)
        case 'standard'|'mean':
            return targets*std + mean
        case 'scalar_non_negative':
            return targets*std
        case _:
            raise ValueError(f"Unrecognized normalization type: {norm_type}")


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
    max_nodes = 0
    max_edges = 0
    num_estimation_graphs = min((num_estimation_graphs, len(dataset)))
    for graph in dataset[:num_estimation_graphs]:
        graph_size = get_graphs_tuple_size(graph)
        if graph_size.n_graph != 1:
            raise ValueError('Dataset contains batched GraphTuples.')

        total_num_nodes += graph_size.n_node
        total_num_edges += graph_size.n_edge
        max_nodes = max((max_nodes, graph_size.n_node))
        max_edges = max((max_edges, graph_size.n_edge))

    num_nodes_per_graph_estimate = total_num_nodes / num_estimation_graphs
    num_edges_per_graph_estimate = total_num_edges / num_estimation_graphs

    n_node = max(
        (next_multiple_of_64(num_nodes_per_graph_estimate * batch_size), max_nodes)
    )
    n_edge = max(
        (next_multiple_of_64(num_edges_per_graph_estimate * batch_size), max_edges)
    )

    padding_budget = GraphsTupleSize(
        n_node=n_node,
        n_edge=n_edge,
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
