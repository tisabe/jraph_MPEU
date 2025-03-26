"""Test the functions of the input_pipeline_test.py module."""
import tempfile
import os
import json
import random

import unittest
import jraph
import numpy as np
from ase import Atoms
import ase.db
import ml_collections

from jraph_MPEU.input_pipeline import (
    DataReader,
    ase_row_to_jraph,
    asedb_to_graphslist,
    atoms_to_nodes_list,
    get_train_val_test_split_dict,
    save_split_dict,
    load_split_dict,
    get_datasets,
    get_atom_num_list,
    label_list_to_class_dict,
    label_list_to_int_class_list,
    shuffle_train_val_data,
    get_graph_fc,
    get_graph_knearest,
    get_graph_cutoff
)
from jraph_MPEU.utils import add_labels_to_graphs, dist_matrix

class TestPipelineFunctions(unittest.TestCase):
    """Testing class."""
    def setUp(self):
        """Prepare test cases."""
        # ase databases with graph features (in the "data" sections)
        self.graphs_dbs = [
            'databases/QM9/graphs_fc_vec.db',
            'databases/aflow/graphs_12knn_vec.db',
            'databases/matproj/mp2018_graphs.db']
        # ase databases without graph features
        self.raw_dbs = ['databases/QM9/qm9.db']
        self.test_db_path = 'databases/QM9/graphs_fc_vec.db'
        # aflow database to test Egap classification inputs
        self.aflow_db = 'databases/aflow/graphs_12knn_vec.db'
        self.rng = np.random.default_rng(seed=7)

    def test_graph_fc(self):
        """Test if fully connected graphs are correctly generated."""
        atoms = Atoms('H5')
        num_nodes = 5
        dimensions = 3
        position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
        atoms.set_positions(position_matrix)

        nodes, pos, edges, senders, receivers = get_graph_fc(atoms)

        expected_edges = []
        expected_senders = []
        expected_receivers = []

        for receiver in range(num_nodes):
            for sender in range(num_nodes):
                if sender != receiver:
                    expected_edges.append(
                        position_matrix[receiver] - position_matrix[sender])
                    expected_senders.append(sender)
                    expected_receivers.append(receiver)

        np.testing.assert_array_equal(np.array([1]*num_nodes), nodes)
        np.testing.assert_array_equal(pos, position_matrix)
        np.testing.assert_array_almost_equal(np.array(expected_edges), edges)
        np.testing.assert_array_equal(np.array(expected_senders), senders)
        np.testing.assert_array_equal(np.array(expected_receivers), receivers)

    def test_catch_fc_with_pbc(self):
        """Test that trying to make a fully connected graph from atoms with
        periodic boundary conditions raises an exception."""
        atoms = Atoms('H5', pbc=True)
        num_nodes = 5
        dimensions = 3
        position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
        atoms.set_positions(position_matrix)

        with self.assertRaises(ValueError, msg='PBC not allowed for fully connected graph.'):
            get_graph_fc(atoms)

    def test_k_nn_random(self):
        """Test generating a k-nearest neighbor graph from random atomic
        positions."""
        num_nodes = 5
        atoms = Atoms(f'H{num_nodes}')
        dimensions = 3
        k = 3
        position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
        distances = dist_matrix(position_matrix)
        atoms.set_positions(position_matrix)
        nodes, pos, edges, senders, receivers = get_graph_knearest(atoms, k)

        expected_senders = []
        expected_receivers = []

        for row in range(num_nodes):
            idx_list = []
            last_idx = 0
            for _ in range(k):
                # temporary last saved minimum value, initialized to high value
                min_val_last = 9999.9
                for col in range(num_nodes):
                    if col == row or (col in idx_list):
                        # do nothing on the diagonal,
                        # or if column has already been included
                        continue
                    else:
                        val = distances[row, col]
                        if val < min_val_last:
                            min_val_last = val
                            last_idx = col
                idx_list.append(last_idx)
                expected_senders.append(last_idx)
                expected_receivers.append(row)

        expected_edges = position_matrix[expected_receivers] - position_matrix[expected_senders]
        # we only check distances exactly, since senders have an arbitrary
        # ordering because of the way neighborlists are built in ase
        dists = np.sqrt(np.sum(edges**2, axis=1))
        dists_expected = np.sqrt(np.sum(expected_edges**2, axis=1))
        np.testing.assert_array_equal(np.array(dists_expected), dists)
        self.assertTupleEqual(np.shape(nodes), (num_nodes,))
        self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
        self.assertTupleEqual(np.shape(edges), (num_nodes*k, dimensions))
        self.assertTupleEqual(np.shape(senders), (num_nodes*k,))
        self.assertTupleEqual(np.shape(receivers), (num_nodes*k,))

    def test_k_nn_pbc(self):
        """Test generating a k-nearest neighbor graph from random atomic
        positions with periodic boundary conditions."""
        cell_l = 2
        num_nodes = 5
        atoms = Atoms(f'H{num_nodes}', cell=[cell_l]*3, pbc=[1, 1, 1])
        dimensions = 3
        k = 3
        position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
        atoms.set_positions(position_matrix)
        nodes, pos, edges, senders, receivers = get_graph_knearest(atoms, k)
        self.assertTupleEqual(np.shape(nodes), (num_nodes,))
        self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
        self.assertTupleEqual(np.shape(edges), (num_nodes*k, dimensions))
        self.assertTupleEqual(np.shape(senders), (num_nodes*k,))
        self.assertTupleEqual(np.shape(receivers), (num_nodes*k,))
        # check that coordinates of pos have been wrapped to inside the cell
        for coordinate in pos.flatten():
            self.assertLessEqual(coordinate, cell_l)

    def test_k_nn_too_far(self):
        """Test generating a k-nearest neighbor graph, but an exception is
        raised because the atoms are too far apart."""
        atoms = Atoms('H2')
        dimensions = 3
        scale = 10
        position_matrix = [[0]*dimensions, [scale]*dimensions]
        k = 1
        atoms.set_positions(position_matrix)
        with self.assertRaises(RuntimeError):
            _ = get_graph_knearest(atoms, k, initial_radius=scale/20)

    def test_get_cutoff_adj_from_dist_random(self):
        """Test generating a graph with constant cutoff from random
        atomic positions."""
        num_nodes = 4
        atoms = Atoms(f'H{num_nodes}')
        dimensions = 3
        cutoff = 0.7
        position_matrix = self.rng.random((num_nodes, dimensions))
        distances = dist_matrix(position_matrix)

        atoms.set_positions(position_matrix)
        nodes, pos, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)

        expected_senders = []
        expected_receivers = []

        for receiver in range(num_nodes):
            for sender in range(num_nodes):
                if not sender == receiver:
                    if distances[sender, receiver] < cutoff:
                        expected_senders.append(sender)
                        expected_receivers.append(receiver)

        expected_edges = position_matrix[expected_receivers] - position_matrix[expected_senders]

        # edges might be arranged differently
        edges = np.sort(edges)
        expected_edges = np.sort(expected_edges)

        np.testing.assert_array_equal(np.array([1]*num_nodes), nodes)
        np.testing.assert_array_equal(pos, position_matrix)
        np.testing.assert_array_almost_equal(np.array(expected_edges), edges)
        self.assertCountEqual(np.array(expected_senders), senders)
        self.assertCountEqual(np.array(expected_receivers), receivers)

    def test_cutoff_pbc(self):
        """Test generating a constant cutoff graph from random atomic
        positions with periodic boundary conditions."""
        cell_l = 10
        num_nodes = 5
        atoms = Atoms(f'H{num_nodes}', cell=[cell_l]*3, pbc=[1, 1, 1])
        dimensions = 3
        position_matrix = self.rng.integers(0, 10, size=(num_nodes, dimensions))
        atoms.set_positions(position_matrix)
        nodes, pos, _, _, _ = get_graph_cutoff(atoms, 5)
        np.testing.assert_array_equal(nodes, [1]*5)
        self.assertTupleEqual(np.shape(nodes), (num_nodes,))
        self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
        # check that coordinates of pos have been wrapped to inside the cell
        for coordinate in pos.flatten():
            self.assertLessEqual(coordinate, cell_l)

    def test_cutoff_no_edges(self):
        """Test generating a constant cutoff graph from random atomic
        positions with periodic boundary conditions."""
        num_nodes = 2
        atoms = Atoms('H2')
        dimensions = 3
        position_matrix = [[0]*dimensions, [1]*dimensions]
        atoms.set_positions(position_matrix)
        with self.assertWarns(RuntimeWarning):
            nodes, pos, edges, senders, receivers = get_graph_cutoff(atoms, 1)
        np.testing.assert_array_equal(nodes, [1]*num_nodes)
        self.assertTupleEqual(np.shape(nodes), (num_nodes,))
        self.assertTupleEqual(np.shape(pos), (num_nodes, dimensions))
        self.assertEqual(len(senders), 0)
        self.assertEqual(len(receivers), 0)
        np.testing.assert_array_equal(edges, np.zeros((0, 1)))

    def test_get_datasets_split(self):
        """Test that the same reproducible splits are returned by get_datasets."""
        config = ml_collections.ConfigDict()
        config.train_frac = 0.5
        config.val_frac = 0.3
        config.test_frac = 0.2
        config.label_str = 'test_label'
        config.selection = None
        config.limit_data = None
        config.num_edges_max = None
        config.seed_splits = 42
        config.normalization_types = {config.label_str: 'mean'}
        config.shuffle_val_seed = -1
        num_rows = 10  # number of rows to write
        label_values = np.arange(num_rows)*1.0
        compound_list = ['H', 'He2', 'Li3', 'Be4', 'B5', 'C6', 'N7', 'O8']

        with tempfile.TemporaryDirectory() as test_dir:  # directory for database
            config.data_file = test_dir + 'test.db'
            path_split = os.path.join(test_dir, 'splits.json')
            path_num = os.path.join(test_dir, 'atomic_num_list.json')
            # check that there is no file with splits or atomic numbers yet
            self.assertFalse(os.path.exists(path_split))
            self.assertFalse(os.path.exists(path_num))
            # create and connect to temporary database
            database = ase.db.connect(config.data_file)
            for label_value in label_values:
                # example structure
                h2_atom = Atoms(
                    random.choice(compound_list))
                key_value_pairs = {config.label_str: label_value}
                data = {
                    'senders': [0],
                    'receivers': [0],
                    'edges': [5.0]
                }
                database.write(h2_atom, key_value_pairs=key_value_pairs, data=data)
            graphs_split, norm_dict = get_datasets(
                config, test_dir
            )
            mean = norm_dict[config.label_str]['mean']
            std = norm_dict[config.label_str]['std']
            train_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['train']]
            val_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['validation']]
            test_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['test']]
            self.assertListEqual(train_labels, [6., 7., 3., 0., 5.])
            self.assertListEqual(val_labels, [1., 2., 9.])
            self.assertListEqual(test_labels, [4., 8.])

            # now with shuffled val and train set
            # split dict will be loaded, test stays the same but val and train
            # are shuffled
            config.shuffle_val_seed = 1

            graphs_split, norm_dict = get_datasets(
                config, test_dir
            )
            mean = norm_dict[config.label_str]['mean']
            std = norm_dict[config.label_str]['std']
            train_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['train']]
            val_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['validation']]
            test_labels = [graph.globals[config.label_str]*std + mean \
                for graph in graphs_split['test']]
            self.assertListEqual(train_labels, [2., 6., 5., 0., 1.])
            self.assertListEqual(val_labels, [9., 3., 7.])
            self.assertListEqual(test_labels, [4., 8.])

    def test_shuffle_train_val_data(self):
        """Test shuffle function. Same seed should produce the same result."""
        train_data = [1, 2, 3, 4, 5]
        val_data = [6, 7, 8]
        seed = 42
        train_new, val_new = shuffle_train_val_data(train_data, val_data, seed)
        self.assertListEqual(train_new, [8, 3, 5, 4, 7])
        self.assertListEqual(val_new, [2, 6, 1])

    def test_dbs_not_empty(self):
        for db_name in self.graphs_dbs + self.raw_dbs:
            if not os.path.isfile(db_name):
                raise FileNotFoundError(f'{db_name} does not exist')
            ase_db = ase.db.connect(db_name)
            self.assertGreater(ase_db.count(), 0, f"{db_name} is empty")

    def test_class_conversion(self):
        """Test converting label list with string classes to int classes."""
        label_list = ['A', 'A', 'B', 'A', 'C', 'C', 'B']
        class_dict_expected = {'A': 0, 'B': 1, 'C': 2}
        class_dict = label_list_to_class_dict(label_list)
        self.assertTrue(class_dict_expected == class_dict)

        class_list_expected = [0, 0, 1, 0, 2, 2, 1]
        class_list = label_list_to_int_class_list(label_list, class_dict)
        self.assertTrue(class_list_expected == class_list)

    def test_get_datasets(self):
        """Test new version of get_datasets.

        For this an ase database is generated with atoms objects and graphs
        attributes. Also a config with parameters is generated."""
        config = ml_collections.ConfigDict()
        config.train_frac = 0.5
        config.val_frac = 0.3
        config.test_frac = 0.2
        config.label_str = 'test_label'
        config.selection = None
        config.limit_data = None
        config.num_edges_max = None
        config.seed_splits = 42
        config.normalization_types = {config.label_str: 'mean'}
        num_rows = 10  # number of rows to write
        label_values = np.arange(num_rows)*1.0
        compound_list = ['H', 'He2', 'Li3', 'Be4', 'B5', 'C6', 'N7', 'O8']

        with tempfile.TemporaryDirectory() as test_dir:  # directory for database
            config.data_file = test_dir + 'test.db'
            path_split = os.path.join(test_dir, 'splits.json')
            path_num = os.path.join(test_dir, 'atomic_num_list.json')
            path_norm = os.path.join(test_dir, 'normalization.json')
            # check that there is no file with splits or atomic numbers yet
            self.assertFalse(os.path.exists(path_split))
            self.assertFalse(os.path.exists(path_num))
            # create and connect to temporary database
            database = ase.db.connect(config.data_file)
            test_split_dict = {
                'train': [1, 2, 3, 4, 5],
                'validation': [6, 7, 8],
                'test': [9, 10]}
            save_split_dict(test_split_dict, test_dir)
            for label_value in label_values:
                # example structure
                h2_atom = Atoms(
                    random.choice(compound_list))
                key_value_pairs = {config.label_str: label_value}
                data = {
                    'senders': [0],
                    'receivers': [0],
                    'edges': [5.0]
                }
                database.write(h2_atom, key_value_pairs=key_value_pairs, data=data)
            graphs_split, norm_dict = get_datasets(
                config, test_dir
            )
            mean = norm_dict[config.label_str]['mean']
            std = norm_dict[config.label_str]['std']
            # calculate expected metrics using only training labels
            mean_expected = np.mean(
                label_values[np.array(test_split_dict['train'])-1])
            std_expected = np.std(
                label_values[np.array(test_split_dict['train'])-1])
            self.assertAlmostEqual(mean, mean_expected)
            self.assertAlmostEqual(std, std_expected)

            # check that the splits.json and atomic_num_list.json file was created
            self.assertTrue(os.path.exists(path_split))
            self.assertTrue(os.path.exists(path_num))
            self.assertTrue(os.path.exists(path_norm))
            # check that the atomic num list has at least one entry
            with open(path_num, 'r', encoding="utf-8") as list_file:
                num_list = json.load(list_file)
            self.assertTrue(len(num_list) > 0)

            globals_expected = {
                'train': [0, 1, 2, 3, 4],
                'validation': [5, 6, 7],
                'test': [8, 9]
            }
            graphs_split_old = graphs_split.copy() # copy for comparison later
            for split, graph_list in graphs_split.items():
                labels = [(graph.globals[config.label_str]*std)+mean for graph in graph_list]
                np.testing.assert_array_equal(labels, globals_expected[split])

            # load the dataset again to check if generated jsons work
            graphs_split, _ = get_datasets(
                config, test_dir
            )
            for split, graph_list in graphs_split.items():
                nodes = [graph.nodes for graph in graph_list]
                graphs_list_old = graphs_split_old[split]
                nodes_old = [graph.nodes for graph in graphs_list_old]
                for node, node_old in zip(nodes, nodes_old):
                    np.testing.assert_array_equal(node['atomic_numbers'], node_old['atomic_numbers'])
                    np.testing.assert_array_equal(node['node_info'], node_old['node_info'])

    def test_get_datasets_class(self):
        """Test get_dataset function using classification label.
        
        For this test to work, an aflow dataset has to be created using
        scripts/data/get_aflow_csv.py and scripts/data/aflow_to_graphs.py."""

        if not os.path.isfile(self.aflow_db):
            raise FileNotFoundError(f'{self.aflow_db} does not exist')
        config = ml_collections.ConfigDict()
        config.train_frac = 0.5
        config.val_frac = 0.3
        config.test_frac = 0.2
        config.label_str = 'Egap_type'
        config.egap_cutoff = 0.0
        config.selection = None
        config.limit_data = 10000
        config.num_edges_max = None
        config.seed_splits = 42
        config.normalization_types = {config.label_str: 'class'}
        config.data_file = self.aflow_db

        with tempfile.TemporaryDirectory() as workdir:
            graphs_split, norm_dict = get_datasets(config, workdir)
            globals_list = []
            for graph in graphs_split['train']:
                globals_list.append(graph.globals[config.label_str])
            self.assertListEqual(sorted(set(globals_list)), [0, 1, 2, 3, 4, 5])

    def test_save_load_split_dict(self):
        """Test the saving and loading of a split dict by generating, saving
        and loading a split dict in a temporary file."""
        test_dict = {
            'train': [1, 2, 3, 4, 5],
            'validation': [6, 7, 8],
            'test': [9]}
        with tempfile.TemporaryDirectory() as test_dir:
            save_split_dict(test_dict, test_dir, database_path='test')
            with open(os.path.join(test_dir, 'splits.json'), 'r', encoding="utf-8") as splits_file:
                splits_dict = json.load(splits_file)
                self.assertEqual(splits_dict['database_path'], 'test')
            dict_loaded = load_split_dict(test_dir)
        # the loaded dict now has signature
        # {1: 'split1', 2: 'split1',... 11: 'split2',...}.
        for key, ids in test_dict.items():
            for i in ids:
                self.assertEqual(
                    key, dict_loaded[i], f"Failed dict: {dict_loaded[i]}")

    def test_get_splits_fail(self):
        """Test getting the indices for train/test/validation splits.

        This test should fail because the fractions don't add up to one."""
        id_list = range(100)
        train_frac = 0.85
        val_frac = 0.15
        test_frac = 0.1
        # function should fail, because fractions do not add up to one
        self.assertRaises(
            ValueError,
            get_train_val_test_split_dict,
            id_list, train_frac, val_frac, test_frac
        )

    def test_get_splits_length(self):
        """Test getting the indices for train/test/validation splits."""
        id_list = range(100)
        train_frac = 0.75
        val_frac = 0.15
        test_frac = 0.1
        split_dict = get_train_val_test_split_dict(
            id_list, train_frac, val_frac, test_frac)
        self.assertEqual(len(split_dict['train']), 75)
        self.assertEqual(len(split_dict['validation']), 15)
        self.assertEqual(len(split_dict['test']), 10)

    def test_get_splits(self):
        """Test getting the exact indices for reproducibility in later versions."""
        id_list = range(1, 21)
        train_frac = 0.5
        val_frac = 0.3
        test_frac = 0.2
        split_dict = get_train_val_test_split_dict(
            id_list, train_frac, val_frac, test_frac)
        np.testing.assert_array_equal(
            split_dict['train'], [6, 15, 10, 8, 17, 12, 4, 1, 16, 13])
        np.testing.assert_array_equal(
            split_dict['validation'], [19, 9, 2, 14, 5, 18])
        np.testing.assert_array_equal(
            split_dict['test'], [20, 3, 7, 11])

    def test_get_cutoff_val(self):
        """Test getting the cutoff types and values from the datasets."""
        for db_name in self.graphs_dbs:
            if not os.path.isfile(db_name):
                raise FileNotFoundError(f'{db_name} does not exist')
            first_row = None
            database = ase.db.connect(db_name)
            for i, row in enumerate(database.select(limit=10)):
                if i == 0:
                    first_row = row
            _ = first_row['cutoff_type']
            _ = first_row['cutoff_val']


    def test_asedb_to_graphslist(self):
        """Test converting an asedb to a list of jraph.GraphsTuple.

        Note: for this test, the QM9 data has to be downloaded
        beforehand, using the scipt get_qm9.py, and converted to graphs
        using asedb_to_graphs.py. The database has to be located at
        QM9/qm9_graphs.db relative to the path of this test.

        Test case:
            file: QM9/qm9_graphs.db
            label_str: U0
            selection: None
            limit: 100
        """

        file_str = self.test_db_path
        if not os.path.isfile(file_str):
            raise FileNotFoundError(f'{file_str} does not exist')
        label_str = 'U0'
        selection = None
        limit = 100
        graphs, ids = asedb_to_graphslist(
            file=file_str, selection=selection, limit=limit)
        _ = [self.assertIsInstance(graph, jraph.GraphsTuple) for graph in graphs]
        _ = [self.assertIsInstance(i, int) for i in ids]
        self.assertEqual(limit, len(graphs))

    def test_DataReader_no_repeat(self):
        """Test the DataReader without repeating."""

        num_graphs = 20 # number of graphs
        graph = jraph.GraphsTuple(
            nodes=np.asarray([0, 1, 2, 3, 4]),
            edges=np.ones((6, 2)),
            senders=np.array([0, 1]),
            receivers=np.array([2, 2]),
            n_node=np.asarray([5]),
            n_edge=np.asarray([6]),
            globals=None)
        graphs = [graph] * num_graphs
        labels = range(num_graphs)
        graphs = add_labels_to_graphs(graphs, labels)
        batch_size = 10

        reader = DataReader(graphs, batch_size, False, 42)
        expected_batches = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 0],
            [18, 19, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        for batch, expected_batch in zip(reader, expected_batches):
            np.testing.assert_array_equal(batch.globals, expected_batch)

    def test_DataReader_repeat(self):
        """Test the DataReader repeating and shuffling."""

        num_graphs = 20 # number of graphs
        graph = jraph.GraphsTuple(
            nodes=np.asarray([0, 1, 2, 3, 4]),
            edges=np.ones((6, 2)),
            senders=np.array([0, 1]),
            receivers=np.array([2, 2]),
            n_node=np.asarray([5]),
            n_edge=np.asarray([6]),
            globals=None)
        graphs = [graph] * num_graphs
        labels = range(num_graphs)
        graphs = add_labels_to_graphs(graphs, labels)
        batch_size = 10
        num_batches = 3  # Number of batches to query and test
        reader = DataReader(graphs, batch_size, True, 42)
        labels_repeat_sum = 0

        for _ in range(num_batches):
            graphs = next(reader)
            labels_batch = graphs.globals
            labels_repeat_sum += np.sum(labels_batch)

        # Check that the accumulated sum of labels is larger than the sum of
        # original labels. This can only be true if the reader is looping.
        self.assertTrue(labels_repeat_sum > np.sum(labels))

    def test_ase_row_to_jraph(self):
        """Test conversion from ase.db.Row to jraph.GraphsTuple."""
        row = None
        with tempfile.TemporaryDirectory() as test_dir:  # directory for database
            db_path = test_dir + 'test.db'
            db = ase.db.connect(db_path)
            h2 = Atoms('H2', [(0, 0, 0), (0, 0, 0.7)])
            data ={'senders': [0, 1], 'receivers': [1, 0],
                'edges': [[0, 0, 0.7], [0, 0, -0.7]], 'node_info': ['node0', 'node1']}
            key_val = {'key1': 'val1', 'key2': 'val2'}
            db.write(h2, key_value_pairs=key_val, data=data)
            row = db.get(1)
        atomic_numbers = row.toatoms().get_atomic_numbers()
        graph = ase_row_to_jraph(row)
        nodes = graph.nodes
        self.assertIsInstance(
            graph, jraph.GraphsTuple, f"{graph} is not a graph")
        np.testing.assert_array_equal(
            atomic_numbers, nodes['atomic_numbers'], "Atomic numbers are not equal")
        self.assertEqual(row.data['node_info'], nodes['node_info'])
        self.assertEqual(row.key_value_pairs, graph.globals)

    def test_dbs_raw(self):
        """Test the raw ase databases without graph features."""
        limit = 100 # maximum number of entries that are read
        for db_name in self.raw_dbs:
            with ase.db.connect(db_name) as asedb:
                keys_list0 = None
                for i, row in enumerate(asedb.select(limit=limit)):
                    key_value_pairs = row.key_value_pairs
                    self.assertIsInstance(key_value_pairs, dict)
                    # check that all the keys are the same
                    if i == 0:
                        keys_list0 = key_value_pairs.keys()
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
        return 0

    def test_dbs_graphs(self):
        """Test the ase databases with graph features."""
        limit = 100 # maximum number of entries that are read
        for db_name in self.graphs_dbs:
            if not os.path.isfile(db_name):
                raise FileNotFoundError(f'{db_name} does not exist')
            with ase.db.connect(db_name) as asedb:
                keys_list0 = None
                data_keys_expected = ['senders', 'receivers', 'edges']
                count_no_edges = 0 # count how many graphs have not edges
                for i, row in enumerate(asedb.select(limit=limit)):
                    key_value_pairs = row.key_value_pairs
                    data = row.data
                    self.assertIsInstance(key_value_pairs, dict)
                    self.assertIsInstance(data, dict)
                    # check that all the keys are the same
                    if i == 0:
                        keys_list0 = key_value_pairs.keys()
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
                        self.assertCountEqual(data.keys(), data_keys_expected)
                    if len(data.edges) == 0:
                        count_no_edges += 1
                self.assertEqual(count_no_edges, 0,
                    f'Number of graphs with zero edges: {count_no_edges}')
        return 0

    def test_atoms_to_nodes_list(self):
        """Example: atomic numbers as nodes before:
        [1 1 1 6] Methane
        [1 1 1 1 1 1 6 6] Ethane
        [1 1 1 1 6 8] Carbon Monoxide
        Will be turned into:
        [0 0 0 1]
        [0 0 0 0 0 0 1 1]
        [0 0 0 0 1 2]"""
        graph0 = jraph.GraphsTuple(
            nodes={'atomic_numbers': np.array([1, 1, 1, 6]), 'node_info': 'test'},
            n_node=[4], n_edge=None, edges=None,
            senders=None, receivers=None, globals=None)
        graph1 = jraph.GraphsTuple(
            nodes={'atomic_numbers': np.array([1, 1, 1, 1, 1, 1, 6, 6]), 'node_info': 'test'},
            n_node=[8], n_edge=None,
            edges=None, senders=None, receivers=None, globals=None)
        graph2 = jraph.GraphsTuple(
            nodes={'atomic_numbers': np.array([1, 1, 1, 1, 6, 8]), 'node_info': 'test'},
            n_node=[6], n_edge=None,
            edges=None, senders=None, receivers=None, globals=None)
        graphs_dict = {1: graph0, 3: graph1, 4:graph2}
        num_list = get_atom_num_list(graphs_dict)
        graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)

        nodes0_expected = np.array([0, 0, 0, 1])
        nodes1_expected = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        nodes2_expected = np.array([0, 0, 0, 0, 1, 2])
        nodes0 = graphs_dict[1].nodes['atomic_numbers']
        nodes1 = graphs_dict[3].nodes['atomic_numbers']
        nodes2 = graphs_dict[4].nodes['atomic_numbers']
        np.testing.assert_array_equal(nodes0_expected, nodes0)
        np.testing.assert_array_equal(nodes1_expected, nodes1)
        np.testing.assert_array_equal(nodes2_expected, nodes2)
        # also check that the list of atomic numbers has three different entries
        self.assertEqual(len(num_list), 3)
        self.assertTrue(len(num_list) == len(set(num_list)))
        np.testing.assert_array_equal(num_list, [1, 6, 8])
        self.assertIsInstance(num_list[0], int)
        self.assertIsInstance(num_list, list)


if __name__ == '__main__':
    unittest.main()
