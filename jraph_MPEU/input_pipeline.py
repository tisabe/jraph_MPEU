"""Functions to import data and engineer data into expected graph format.

The functions in this file interface with the ase.db to make lists of jraph
graphs, where the labels are standardized and the nodes have the right numbers
as input features.
"""

from xmlrpc.client import Boolean
import random
from typing import Dict, Iterable, Sequence, Tuple
import warnings

from absl import logging
import ml_collections
import jraph
import sklearn.model_selection
import numpy as np
import ase.db
import ase
from ase import Atoms
from ase.neighborlist import NeighborList

from jraph_MPEU.utils import (
    estimate_padding_budget_for_batch_size,
    normalize_targets,
    add_labels_to_graphs,
    load_config
)


def load_data(workdir):
    """Load datasets only using the working directory."""
    config = load_config(workdir)
    dataset, dataset_raw, mean, std, _ = get_datasets(config)  # might refactor
    return dataset, dataset_raw, mean, std


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

    for i in range(len(atoms)):
        neighbor_indices, offset = neighborhood.get_neighbors(i)
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
        list of jraph.GraphsTuple, list of labels as single scalars.
    """
    graphs = []
    labels = []
    ase_db = ase.db.connect(file)
    count = 0
    #print(f'Selection: {selection}')
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
        count += 1

    return graphs, labels

def atoms_to_nodes_list(graphs: Sequence[jraph.GraphsTuple]) -> Tuple[
        Sequence[jraph.GraphsTuple], list]:
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
    Also return the number of classes."""
    num_list = [] # List with atomic numbers in the graphs list.
    # Generate full list first.
    for graph in graphs:  # Loop over all graphs.
        nodes = graph.nodes  # Grab information about nodes in the graph.
        for num in nodes:  # Loop over nodes in a graph.
            if not num in num_list:
                # Append unseen atomic numbers to num_list.
                num_list.append(int(num))
    # Transform atomic numbers into classes. Meaning relabel the atomic number
    # compactly with a new compact numbering system.
    for graph in graphs:
        nodes = graph.nodes
        for i, num in enumerate(nodes):
            nodes[i] = num_list.index(num)
        graph._replace(nodes=nodes)

    return graphs, num_list


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
            batch_size: int, repeat: Boolean, seed: int = None):
        self.data = data[:]  # Pass a copy of the list.
        self.batch_size = batch_size
        self.repeat = repeat
        self.total_num_graphs = len(data)
        self.seed = seed
        self._generator = self._make_generator()

        self.budget = estimate_padding_budget_for_batch_size(
            self.data, batch_size,
            num_estimation_graphs=1000)

        # This makes this thing complicated. From outside of DataReader
        # we interface with this batch generator, but this batch_generator
        # needs an iterator itself which is also defined in this class.
        self.batch_generator = jraph.dynamically_batch(
            self._generator,
            self.budget.n_node,
            self.budget.n_edge,
            self.budget.n_graph)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.batch_generator)

    def _make_generator(self):
        random.seed(a=self.seed)
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if not self.repeat:
                if idx == self.total_num_graphs:
                    idx = 0
                    # Here's the problem. At the end of iterating through this
                    # dataset, this return is reached and when trying to
                    # iterate with this generator again, only one None is
                    # returned.
                    return
            else:
                if idx == self.total_num_graphs:
                    random.shuffle(self.data)
                # This will reset the index to 0 if we are at the end of the
                # dataset.
                idx = idx % self.total_num_graphs
            graph = self.data[idx]
            idx += 1
            yield graph


def get_datasets(config: ml_collections.ConfigDict) -> Tuple[
        Dict[str, Iterable[jraph.GraphsTuple]],
        Dict[str, Sequence[jraph.GraphsTuple]],
        float, float, list]:
    """Return a dict with a dataset for each split (train, val, test).

    Return in normalized and in raw form/labels. Also return the mean and
    standard deviation.

    Each dataset is an iterator that yields batches of graphs.
    The training dataset will reshuffle each time the end of the list has been
    reached, while the validation and test sets are only iterated over once.

    The raw dataset is just a dict with lists of graphs.

    The graphs have their regression label as a global feature attached.
    TODO: refactor this to make it single responsiblity. Idea: split this
    function into getting raw data and a normalizing/processing function.
    """
    # Data will be split into normalized data for regression and raw data for
    # analyzing later
    graphs_list, labels_list = asedb_to_graphslist(
        config.data_file,
        label_str=config.label_str,
        selection=config.selection,
        num_edges_max=config.num_edges_max,
        limit=config.limit_data)
    # Convert the atomic numbers in nodes to classes and set number of classes.
    graphs_list, num_list = atoms_to_nodes_list(graphs_list)

    num_classes = len(num_list)
    config.max_atomic_number = num_classes
    labels_raw = labels_list

    labels_list, mean, std = normalize_targets(
        graphs_list, labels_list, config)
    logging.info(f'Mean: {mean}, Std: {std}')
    graphs_list = add_labels_to_graphs(graphs_list, labels_list)
    graphs_raw = add_labels_to_graphs(graphs_list, labels_raw)

    # Split the graphs into three splits using the fractions defined in config.
    (
        train_set,
        val_and_test_set,
        train_raw,
        val_and_test_raw) = sklearn.model_selection.train_test_split(
            graphs_list, graphs_raw,
            test_size=config.test_frac+config.val_frac,
            random_state=0)

    (
        val_set,
        test_set,
        val_raw,
        test_raw) = sklearn.model_selection.train_test_split(
            val_and_test_set, val_and_test_raw,
            test_size=config.test_frac/(config.test_frac+config.val_frac),
            random_state=1)

    # Define iterators and generators.
    reader_train = DataReader(
        data=train_set,
        batch_size=config.batch_size,
        repeat=True,
        seed=config.seed)
    reader_val = DataReader(
        data=val_set,
        batch_size=config.batch_size,
        repeat=False)
    reader_test = DataReader(
        data=test_set,
        batch_size=config.batch_size,
        repeat=False)

    dataset = {
        'train': reader_train,
        'validation': reader_val,
        'test': reader_test}

    dataset_raw = {
        'train': train_raw,
        'validation': val_raw,
        'test': test_raw}

    return dataset, dataset_raw, mean, std, num_list


def get_train_val_test_split_dict(
        id_list: list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """Return the id_list split into train, validation and test indices."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-5:
        raise ValueError('Train, val and test fractions do not add up to one.')
    (
        train_set,
        val_and_test_set) = sklearn.model_selection.train_test_split(
            id_list,
            test_size=test_frac+val_frac,
            random_state=0)

    (
        val_set,
        test_set) = sklearn.model_selection.train_test_split(
            val_and_test_set,
            test_size=test_frac/(test_frac+val_frac),
            random_state=1)
    split_dict = {'train':train_set, 'validation':val_set, 'test':test_set}
    return split_dict

def add_splits_to_database(config):
    """Modify the database given in the config.data_file with split value for
    each row."""
    num_rows = 1000  # TODO: get actual number of rows in database
    ids_all = range(1, num_rows+1)  # row ids in ase.db start at 1
    split_ids_dict = get_train_val_test_split_dict(
        ids_all, config.train_frac, config.val_frac, config.test_frac)
    # connect to database and update rows
    with ase.db.connect(config.data_file) as db:
        for split, ids in split_ids_dict.items():
            for id_single in ids:
                db.update(id_single, split=split)
