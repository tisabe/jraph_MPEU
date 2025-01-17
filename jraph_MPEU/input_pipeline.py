"""Functions to import data and engineer data into expected graph format.

The functions in this file interface with the ase.db to make lists of jraph
graphs, where the labels are standardized and the nodes have the right numbers
as input features.
"""

from typing import Generator, Iterator, Sequence

import os
from xmlrpc.client import Boolean
import random
from typing import Sequence, Tuple
import warnings
import json

from absl import logging
import jax
import jraph
from jraph._src import graph as gn_graph
import sklearn.model_selection
import numpy as np
import ase.db
import ase
from ase import Atoms
from ase.neighborlist import NeighborList

import functools

from jraph_MPEU.utils import (
    estimate_padding_budget_for_batch_size,
    get_node_edge_distribution_for_batch,
    get_static_budget_for_constant_size,
    load_config,
    pad_graph_to_nearest_power_of_two,
    pad_graph_to_nearest_multiple_of_64,
    pad_graph_to_constant_size,
    get_normalization_metrics,
    normalize_graphs,
    load_config
)


def load_data(workdir):
    """Load datasets only using the working directory."""
    config = load_config(workdir)
    dataset, mean, std = get_datasets(config, workdir)  # might refactor
    return dataset, mean, std


def get_graph_fc(atoms: Atoms):
    """Return the graph features, with fully connected edges.

    Turn the ase.Atoms object into graph features, i.e. nodes and edges.
    The edges will be fully connected, so every node is connected to every other node.
    """
    if np.any(atoms.get_pbc()):
        #raise Exception('Received Atoms object with periodic boundary conditions. ' +
        #    'Fully connected graph cannot be generated.')
        raise Exception('PBC not allowed for fully connected graph.')
    nodes = [] # initialize arrays, to be filled in loop later
    senders = []
    receivers = []
    edges = []
    atom_numbers = atoms.get_atomic_numbers() # get array of atomic numbers
    atom_positions = atoms.get_positions(wrap=False)
    n_atoms = len(atoms)

    for i in range(n_atoms):
        nodes.append(atom_numbers[i])
        for j in range(n_atoms):
            # get all edges except self edges
            if i != j:
                i_pos = atom_positions[i]
                j_pos = atom_positions[j]
                dist_vec = i_pos - j_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                senders.append(j)
                receivers.append(i)
                edges.append(dist)

    return (
        np.array(nodes),
        atom_positions,
        np.array(edges),
        np.array(senders),
        np.array(receivers)
    )


def get_graph_cutoff(atoms: Atoms, cutoff):
    '''Return the graph features, with cutoff adjacency.
    Inspired by https://github.com/peterbjorgensen/msgnet/blob/master/src/msgnet/dataloader.py'''

    nodes = [] # initialize arrays, to be filled in loop later
    senders = []
    receivers = []
    edges = []
    atom_numbers = atoms.get_atomic_numbers() # get array of atomic numbers

    # divide cutoff by 2, because ASE defines two atoms as neighbours
    # when their spheres of radii r overlap
    radii = [cutoff/2] * len(atoms) # make a list with length len(atoms)
    neighborhood = NeighborList(
        radii, skin=0.0, self_interaction=False, bothways=True
    )
    neighborhood.update(atoms)

    if np.any(atoms.get_pbc()):
        atom_positions = atoms.get_positions(wrap=True)
    else:
        atom_positions = atoms.get_positions(wrap=False)

    unitcell = atoms.get_cell()

    for i in range(len(atoms)):
        nodes.append(atom_numbers[i])

    # Loop over the atoms in the unit cell.
    for i in range(len(atoms)):
        # Get the neighbourhoods of atom i
        neighbor_indices, offset = neighborhood.get_neighbors(i)
        # Loop over the neighbours of atom i. Offset helps us calculate the
        # distance to atoms in neighbouring unit cells.
        for j, offs in zip(neighbor_indices, offset):
            i_pos = atom_positions[i]
            j_pos = atom_positions[j] + np.dot(offs, unitcell)
            dist_vec = i_pos - j_pos
            dist = np.sqrt(np.dot(dist_vec, dist_vec))

            senders.append(j)
            receivers.append(i)
            edges.append(dist)

    if len(edges) == 0:
        warnings.warn("Generated graph has zero edges")
        edges = np.zeros((0, 1))

    return (
        np.array(nodes),
        atom_positions,
        np.array(edges),
        np.array(senders),
        np.array(receivers)
    )

def get_graph_knearest(
        atoms: Atoms, num_neighbors, initial_radius=3.0):
    '''Return the graph features, with knearest adjacency.
    Inspired by https://github.com/peterbjorgensen/msgnet/blob/master/src/msgnet/dataloader.py'''

    atoms.wrap() # put atoms inside unit cell by wrapping their positions
    atom_numbers = atoms.get_atomic_numbers()
    unitcell = atoms.get_cell()

    # We want to calculate k nearest neighbors, so we start within a sphere
    # with radius R. In this sphere we are calculating the number of neighbors,
    # if there are not enough, i.e. the number of neighbors within the sphere
    # is smaller than k, R is increased until we found enough neighbors. After
    # that we discard all neighbors except the k nearest.
    for multiplier in range(1, 11):
        if multiplier == 10:
            raise RuntimeError("Reached maximum radius")
        radii = [initial_radius * multiplier] * len(atoms)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=False, bothways=True
        )
        neighborhood.update(atoms)

        nodes = []
        edges = []
        senders = []
        receivers = []
        if np.any(atoms.get_pbc()):
            atom_positions = atoms.get_positions(wrap=True)
        else:
            atom_positions = atoms.get_positions(wrap=False)
        keep_edges = []
        keep_senders = []
        keep_receivers = []

        for i in range(len(atoms)):
            nodes.append(atom_numbers[i])

        early_exit = False
        for i in range(len(atoms)):
            this_edges = []
            this_senders = []
            this_receivers = []
            neighbor_indices, offset = neighborhood.get_neighbors(i)
            if len(neighbor_indices) < num_neighbors:
                # Not enough neigbors, so exit and increase radius
                early_exit = True
                break
            for j, offs in zip(neighbor_indices, offset):
                i_pos = atom_positions[i]
                j_pos = atom_positions[j] + np.dot(offs, unitcell)
                dist_vec = i_pos - j_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                this_edges.append([dist])
                this_senders.append(j)
                this_receivers.append(i)
            edges.append(np.array(this_edges))
            senders.append(np.array(this_senders))
            receivers.append(np.array(this_receivers))
        if early_exit:
            continue
        else:
            for e_ind, s_ind, r_ind in zip(edges, senders, receivers):
                # Keep only num_neighbors closest indices
                keep_ind = np.argsort(e_ind[:, 0])[0:num_neighbors]
                keep_edges.append(e_ind[keep_ind])
                keep_senders.append(s_ind[keep_ind])
                keep_receivers.append(r_ind[keep_ind])
        break
    return (
        np.array(nodes),
        atom_positions,
        np.concatenate(keep_edges).flatten(),
        np.concatenate(keep_senders),
        np.concatenate(keep_receivers),
    )


def ase_row_to_jraph(row: ase.db.row.AtomsRow) -> jraph.GraphsTuple:
    """Return the ASE row as a graph."""
    senders = row.data['senders']
    receivers = row.data['receivers']
    edges = row.data['edges']
    atoms = row.toatoms()
    nodes = atoms.get_atomic_numbers()

    graph = jraph.GraphsTuple(
        n_node=np.asarray([len(nodes)]),
        n_edge=np.asarray([len(senders)]),
        nodes=nodes, edges=edges,
        globals=None,
        senders=np.asarray(senders), receivers=np.asarray(receivers))

    return graph

def asedb_to_graphslist(
        file: str,
        label_str: str,
        selection: str = None,
        num_edges_max: int = None,
        limit: int = None
    ) -> Tuple[Sequence[jraph.GraphsTuple], list]:
    """Return a list of graphs, by loading rows from local ase database at file.

    Args:
        file: string where the database file is located
        label_str: which property to grab from database as label for regression.
            It is saved as Global of the respective graph.
        selection: ase.db selection parameter, can be integer id, string or
            list of strings or tuples.
        num_edges_max: integer, cutoff for the maximum number of edges in the
            graph. Graphs with more edges are discarded.
            Note: if limit is not None, fewer graph may be returned than limit,
            if graphs are discarded with too many edges.
        limit: maximum number of graphs queried from the database.

    Returns:
        list of jraph.GraphsTuple, list of labels as single scalars,
        ids with the corresponding asedb id for each graph
    """
    graphs = []
    labels = []
    ids = []
    ase_db = ase.db.connect(file)
    if limit is None:
        count = ase_db.count(selection=selection)
    else:
        count = limit
    logging.info(f'Number of entries selected: {count}')

    for _, row in enumerate(ase_db.select(selection=selection, limit=limit)):
        graph = ase_row_to_jraph(row)
        n_edge = int(graph.n_edge)
        if num_edges_max is not None:
            if n_edge > num_edges_max:  # do not include graphs with too many edges
                # TODO: test this
                continue
        if n_edge == 0:  # do not include graphs without edges
            continue
        graphs.append(graph)
        label = row.key_value_pairs[label_str]
        labels.append(label)
        ids.append(row.id)

    return graphs, labels, ids


def atoms_to_nodes_list(
        graphs_dict: dict, num_list: list) -> dict:
    """Encodes the atomic numbers of nodes in a graph in compact fashion.

    Return graphs with atomic numbers as graph-nodes turned into
    nodes with atomic numbers as classes, and return the list that transforms
    atomic numbers to classes for reproducibility.
    This gets rid of atomic numbers that are not present in the dataset.

    Example: atomic numbers as nodes before:
    [1 1 1 1 6] Methane
    [1 1 1 1 1 1 6 6] Ethane
    [1 1 1 1 6 8] Carbon Monoxide
    Will be turned into:
    [0 0 0 0 1]
    [0 0 0 0 0 0 1 1]
    [0 0 0 0 1 2]
    as there are only three different atomic numbers present in the list.
    """
    # Transform atomic numbers into classes. Meaning relabel the atomic number
    # compactly with a new compact numbering system.
    for graph in graphs_dict.values():
        nodes = graph.nodes
        for i, num in enumerate(nodes):
            nodes[i] = num_list.index(num)
        graph._replace(nodes=nodes)

    return graphs_dict


def get_atom_num_list(graphs_dict):
    """Return the atomic num list. See atoms_to_nodes_list for details."""
    num_list = [] # List with atomic numbers in the graphs list.

    for graph in graphs_dict.values():  # Loop over all graphs.
        nodes = graph.nodes  # Grab information about nodes in the graph.
        for num in nodes:  # Loop over nodes in a graph.
            if not num in num_list:
                # Append unseen atomic numbers to num_list.
                num_list.append(int(num))
    return num_list


def label_list_to_class_dict(label_list):
    """Return class dict, i.e. which class corresponds to which integer.
    example: label_list = ['metal', 'metal', 'non-metal'],
    then class_dict = {'metal': 0, 'non-metal': 1}
    """
    unique_labels = sorted(set(label_list))  # ensure reproducibility by sorted
    return {key: value for (value, key) in enumerate(unique_labels)}


def label_list_to_int_class_list(label_list, class_dict):
    """Convert string or numerical classes to integers using class_dict."""
    return [class_dict[label] for label in label_list]


class DataReader:
    """Data reader class.

    Returns batches of graphs as a generator. The graphs are batched
    dynamically with jraph.dynamically_batch. The generator can either loop
    (for training data) or end after one pass through the dataest (for
    evaluation).
    After evaluation in non-loop mode, the generator can no longer be used,
    and we suggest making a new DataReader Object, by initializing it
    with the old data property.
    """
    def __init__(
            self, data: Sequence[jraph.GraphsTuple],
            batch_size: int, repeat: Boolean, seed: int = None,
            dynamic_batch: bool = True,
            static_round_to_multiple: bool = True,
            static_constant_batch: bool = False,
            num_estimation_graphs: int = 1000,
            compile_batching=False,
            ):
        self.data = data[:]  # Pass a copy of the list.
        self.batch_size = batch_size
        self.repeat = repeat
        self.total_num_graphs = len(data)
        self.seed = seed
        self._generator = self._make_generator()
        self.compile_batching = compile_batching
        self._timing_measurements_batching = []
        self._update_measurements = []

        self._num_nodes_per_batch_before_batching = []
        self._num_edges_per_batch_before_batching = []
        self._num_nodes_per_batch_after_batching = []
        self._num_edges_per_batch_after_batching = []

        self.static_round_to_multiple = static_round_to_multiple
        self.static_constant_batch = static_constant_batch
        
        self.budget = estimate_padding_budget_for_batch_size(
            self.data, batch_size,
            num_estimation_graphs=num_estimation_graphs)

        self.dynamic_batch = dynamic_batch
        # From outside of DataReader
        # we interface with this batch generator, but this batch_generator
        # needs an iterator itself which is also defined in this class.
        if self.dynamic_batch is True:
            self.batch_generator = jraph.dynamically_batch(
                self._generator,
                self.budget.n_node,
                self.budget.n_edge,
                self.budget.n_graph)

        elif self.static_constant_batch is True:
            # Get the padding limits:
            self.pad_nodes_to, self.pad_edges_to = get_static_budget_for_constant_size(self.data, batch_size)
            self.batch_generator = self.static_batch_constant(
                self._generator,
                self.batch_size,
                self.pad_nodes_to,
                self.pad_edges_to,
            )
        else:  # This works for static-2^N/static-64
            self.batch_generator = self.static_batch(
                self._generator,
                self.batch_size
            )

    def static_batch(
            self, graphs_tuple_iterator: Iterator[gn_graph.GraphsTuple],
            batch_size: int) -> Generator[gn_graph.GraphsTuple, None, None]:

        batch_size_minus_one = batch_size-1
        # Curious if we should initialize this. [None]*batch_size_minus_one
        # It seems like MCPCDF is not too worried about it.
        accumulated_graphs = []

        for graph in graphs_tuple_iterator:
            
            if len(accumulated_graphs) == batch_size_minus_one:
                # Call get number of nodes/edges in the list.
                # Append to the list. self._num_nodes_per_batch
                
                # sum_of_nodes_in_batch, sum_of_edges_in_batch = get_node_edge_distribution_for_batch(
                #     accumulated_graphs)

                # self._num_nodes_per_batch_before_batching.append(sum_of_nodes_in_batch)
                # self._num_edges_per_batch_before_batching.append(sum_of_edges_in_batch)

                accumulated_graphs = jraph.batch_np(accumulated_graphs)
                # Call get number of nodes/edges in the new list.
                # Append to the list self._num_nodes_per_batch_after_batching.

                # How do i get the data out?
                if self.static_round_to_multiple:
                    yield pad_graph_to_nearest_multiple_of_64(
                        accumulated_graphs)
                    accumulated_graphs = []
                else:
                    yield pad_graph_to_nearest_power_of_two(
                        accumulated_graphs)
                    accumulated_graphs = []

            else:
                accumulated_graphs.append(graph)


    def static_batch_constant(
            self, graphs_tuple_iterator: Iterator[gn_graph.GraphsTuple],
            batch_size: int, pad_nodes_to: int, pad_edges_to: int) -> Generator[gn_graph.GraphsTuple, None, None]:

        batch_size_minus_one = batch_size-1
        # Curious if we should initialize this. [None]*batch_size_minus_one
        # It seems like MCPCDF is not too worried about it.
        accumulated_graphs = []

        for graph in graphs_tuple_iterator:
            
            if len(accumulated_graphs) == batch_size_minus_one:
                # Call get number of nodes/edges in the list.
                # Append to the list. self._num_nodes_per_batch
                
                # sum_of_nodes_in_batch, sum_of_edges_in_batch = get_node_edge_distribution_for_batch(
                #     accumulated_graphs)

                # self._num_nodes_per_batch_before_batching.append(sum_of_nodes_in_batch)
                # self._num_edges_per_batch_before_batching.append(sum_of_edges_in_batch)

                accumulated_graphs = jraph.batch_np(accumulated_graphs)
                # Call get number of nodes/edges in the new list.
                # Append to the list self._num_nodes_per_batch_after_batching.

                # How do i get the data out?
                yield pad_graph_to_constant_size(accumulated_graphs, pad_nodes_to, pad_edges_to)
                accumulated_graphs = []

            else:
                accumulated_graphs.append(graph)




    def __iter__(self):
        return self

    def __next__(self):
        return next(self.batch_generator)
        # if self.dynamic_batch and self.compile_batching is True:
        #     return self.jax_next()
        # else:
        #     return self.uncompiled_next()

    # @functools.partial(jax.jit, static_argnums=0)
    # def jax_next(self):
    #     return next(self.batch_generator)

    # def uncompiled_next(self):
    #     return next(self.batch_generator)

    def _make_generator(self):
        random.seed(a=self.seed)
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if not self.repeat:
                # logging.info('Make gen: REPEAT IS NOT TRUE')

                if idx == self.total_num_graphs:
                    idx = 0
                    # Here's the problem. At the end of iterating through this
                    # dataset, this return is reached and when trying to
                    # iterate with this generator again, only one None is
                    # returned.
                    return
            else:
                # logging.info('Make gen: REPEAT TRUE')
                if idx == self.total_num_graphs:
                    random.shuffle(self.data)
                # This will reset the index to 0 if we are at the end of the
                # dataset.
                idx = idx % self.total_num_graphs
            graph = self.data[idx]
            idx += 1
            yield graph


def get_train_val_test_split_dict(
        id_list: list, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """Return the id_list split into train, validation and test indices."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-5:
        raise ValueError('Train, val and test fractions do not add up to one.')
    (
        train_set,
        val_and_test_set) = sklearn.model_selection.train_test_split(
            id_list,
            test_size=test_frac+val_frac,
            random_state=seed-42)
    # seed-42 as seed is 42 by default, but default random state should be 0

    (
        val_set,
        test_set) = sklearn.model_selection.train_test_split(
            val_and_test_set,
            test_size=test_frac/(test_frac+val_frac),
            random_state=1)
    # seed-41 as seed is 42 by default, but default random state should be 1
    split_dict = {'train':train_set, 'validation':val_set, 'test':test_set}
    return split_dict


def split_dict_to_lists(split_dict_in):
    """Convert split_dict to signature {'split1': [...], 'split2': [...], ...}.

    split_dict must have signature
    {1: 'split1', 2: 'split1',... 11: 'split2',...}.
    """
    split_lists = {}
    for id_single, split in split_dict_in.items():
        if split in split_lists.keys():
            split_lists[split].append(id_single)
        else:
            split_lists[split] = []
            split_lists[split].append(id_single)
    return split_lists


def lists_to_split_dict(split_lists):
    """Convert split_lists to signature {1: 'split1', 2: 'split1', ... }.

    split_lists must have signature {'split1': [...], 'split2': [...], ...}.
    """
    split_dict = {}
    for split, ids in split_lists.items():
        for id_single in ids:
            split_dict[id_single] = split
    return split_dict


def save_split_dict(split_lists, workdir):
    """Save the split_lists in json file at workdir.

    The split_lists has signature:
    {'split1': [1, 2,...], 'split2': [11, 21...]}, ... and this is turned into
    the signature {1: 'split1', 2: 'split1',... 11: 'split2',...}.
    This format is more practical when doing the inference after training."""
    split_dict = lists_to_split_dict(split_lists)

    with open(os.path.join(workdir, 'splits.json'), 'w') as splits_file:
        json.dump(split_dict, splits_file, indent=4, separators=(',', ': '))


def load_split_dict(workdir):
    """Load the split dict that saved ids and their split in workdir.

    The keys are integer ids and the values are splitnames as strings."""
    with open(os.path.join(workdir, 'splits.json'), 'r') as splits_file:
        splits_dict = json.load(splits_file, parse_int=True)
    return {int(k): v for k, v in splits_dict.items()}


def cut_egap(egap: float, threshold: float = 0.0):
    """Turn egap into metal or non-metal class by applying threshold."""
    if egap > threshold:
        return 1
    else:
        return 0


def get_datasets(config, workdir):
    """New version of dataset getter."""
    # TODO: put in real docstring.
    # Pull the data from the ase database. If there is no file with splits
    # present, pull the data using the parameters selection and limit.
    # If the file with splits is present, load it and pull the data using the
    # ids in the split file
    split_path = os.path.join(workdir, 'splits.json')
    if not os.path.exists(split_path):
        logging.info(f'Did not find split file at {split_path}. Pulling data.')
        graphs_list, labels_list, ids = asedb_to_graphslist(
            config.data_file,
            label_str=config.label_str,
            selection=config.selection,
            num_edges_max=config.num_edges_max,
            limit=config.limit_data)
        # transform graphs list into graphs dict, same for labels
        graphs_dict = {}
        labels_dict = {}
        for (graph, label, id_single) in zip(graphs_list, labels_list, ids):
            graphs_dict[id_single] = graph
            labels_dict[id_single] = label

    else:
        logging.info(f'Found split file. Connecting to ase.db at {config.data_file}')
        graphs_dict = {}
        labels_dict = {}
        split_dict = load_split_dict(workdir)
        ase_db = ase.db.connect(config.data_file)
        for id_single in split_dict.keys():
            row = ase_db.get(id_single)
            graph = ase_row_to_jraph(row)
            #graphs_list.append(graph)
            graphs_dict[id_single] = graph
            label = row.key_value_pairs[config.label_str]
            #labels_list.append(label)
            labels_dict[id_single] = label
    # In either path, the list ids has been created at this point. ids contains
    # the asedb row.id of each graph that has been pulled.

    # Convert the atomic numbers in nodes to classes and set number of classes.
    num_path = os.path.join(workdir, 'atomic_num_list.json')
    if not os.path.exists(num_path):
        num_list = get_atom_num_list(graphs_dict)
        # save num list here
        with open(num_path, 'w+') as num_file:
            json.dump(num_list, num_file)
    else:
        with open(num_path, 'r') as num_file:
            num_list = json.load(num_file)
    graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)

    num_classes = len(num_list)
    config.max_atomic_number = num_classes

    for (id_single, graph), label in zip(graphs_dict.items(), labels_dict.values()):
        graphs_dict[id_single] = graph._replace(globals=np.array([label]))

    if not os.path.exists(split_path):
        logging.debug('Generating splits and saving split file.')
        # If split file did not exist before, generate and save it
        split_lists = get_train_val_test_split_dict(
            ids, 1.0-(config.val_frac+config.test_frac), config.val_frac,
            config.test_frac, seed=config.seed
        )
        save_split_dict(split_lists, workdir)
    else:
        # If it did exist, convert split_dict to split_lists
        split_lists = split_dict_to_lists(split_dict)

    graphs_split = {}  # dict with the graphs list divided into splits
    for key, id_list in split_lists.items():
        graphs_split[key] = []  # init lists for every split
        for id_single in id_list:
            # append graph from graph_list using the id in split_dict
            graphs_split[key].append(graphs_dict[id_single])

    # get normalization metrics from train data
    if config.label_type == 'scalar':
        mean, std = get_normalization_metrics(
            graphs_split['train'], config.aggregation_readout_type)
        logging.info(f'Mean: {mean}, Std: {std}')
    elif config.label_type == 'class':
        mean, std = None, None
    else:
        raise ValueError(f'{config.label_type} not recognized as label type.')

    for split, graphs_list in graphs_split.items():
        if config.label_type == 'scalar':
            graphs_split[split] = normalize_graphs(
                graphs_list, mean, std, config.aggregation_readout_type)
        elif config.label_type == 'class':
            for i, graph in enumerate(graphs_list):
                label = cut_egap(graph.globals[0], config.egap_cutoff)
                graphs_list[i] = graph._replace(globals=np.array([label]))
    return graphs_split, mean, std
