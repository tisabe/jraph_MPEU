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
    label_list_to_int_class_list
)
from jraph_MPEU.utils import add_labels_to_graphs

class TestPipelineFunctions(unittest.TestCase):
    """Testing class."""
    def setUp(self):
        """Prepare test cases."""
        # ase databases with graph features (in the "data" sections)
        self.graphs_dbs = [
            'QM9/qm9_graphs.db',
            'aflow/graphs_all_12knn.db',
            'matproj/mp2018_graphs.db']
        # ase databases without graph features
        self.raw_dbs = ['QM9/qm9.db']
        self.test_db = 'QM9/qm9_graphs.db'
        # aflow database to test Egap classification inputs
        self.aflow_db = 'aflow/graphs_all_12knn.db'

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
        config.seed = 42
        config.aggregation_readout_type = 'mean'
        config.label_type = 'scalar'
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
            graphs_split, mean, std = get_datasets(
                config, test_dir
            )
            # calculate expected metrics using only training labels
            mean_expected = np.mean(
                label_values[np.array(test_split_dict['train'])-1])
            std_expected = np.std(
                label_values[np.array(test_split_dict['train'])-1])
            self.assertAlmostEqual(mean, mean_expected)
            self.assertAlmostEqual(std, std_expected)
            self.assertTrue(mean is not None)
            self.assertTrue(std is not None)

            # check that the splits.json and atomic_num_list.json file was created
            self.assertTrue(os.path.exists(path_split))
            self.assertTrue(os.path.exists(path_num))
            # check that the atomic num list has at least one entry
            with open(path_num) as list_file:
                num_list = json.load(list_file)
            self.assertTrue(len(num_list) > 0)

            globals_expected = {
                'train': [0, 1, 2, 3, 4],
                'validation': [5, 6, 7],
                'test': [8, 9]
            }
            graphs_split_old = graphs_split.copy() # copy for comparison later
            for split, graph_list in graphs_split.items():
                labels = [(graph.globals[0]*std)+mean for graph in graph_list]
                np.testing.assert_array_equal(labels, globals_expected[split])

            # load the dataset again to check if generated jsons work
            graphs_split, mean, std = get_datasets(
                config, test_dir
            )
            # TODO: test that the nodes are still transformed in the same way
            for split, graph_list in graphs_split.items():
                nodes = [np.array(graph.nodes) for graph in graph_list]
                graphs_list_old = graphs_split_old[split]
                nodes_old = [np.array(graph.nodes) for graph in graphs_list_old]
                for node, node_old in zip(nodes, nodes_old):
                    np.testing.assert_array_equal(node, node_old)

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
        config.label_str = 'Egap'
        config.egap_cutoff = 0.0
        config.selection = None
        config.limit_data = 10000
        config.num_edges_max = None
        config.seed = 42
        config.aggregation_readout_type = 'mean'
        config.label_type = 'class'
        config.data_file = self.aflow_db

        with tempfile.TemporaryDirectory() as workdir:
            graphs_split, mean, std = get_datasets(config, workdir)
            globals_list = []
            for graph in graphs_split['train']:
                globals_list.append(graph.globals[0])
            # check that there are only zeros and ones in the list
            self.assertTrue(sorted(set(globals_list)) == [0, 1])
            # TODO: check that threshold is evaluated correctly

    def test_save_load_split_dict(self):
        """Test the saving and loading of a split dict by generating, saving
        and loading a split dict in a temporary file."""
        test_dict = {
            'train': [1, 2, 3, 4, 5],
            'validation': [6, 7, 8],
            'test': [9]}
        with tempfile.TemporaryDirectory() as test_dir:
            save_split_dict(test_dict, test_dir)
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

        file_str = self.test_db
        if not os.path.isfile(file_str):
            raise FileNotFoundError(f'{file_str} does not exist')
        label_str = 'U0'
        selection = None
        limit = 100
        graphs, labels, ids = asedb_to_graphslist(
            file=file_str, label_str=label_str, selection=selection, limit=limit)
        _ = [self.assertIsInstance(graph, jraph.GraphsTuple) for graph in graphs]
        _ = [self.assertIsInstance(i, int) for i in ids]
        # assert that the selection worked
        _ = [self.assertFalse(label == 0) for label in labels]
        _ = [self.assertTrue(label < 0) for label in labels]
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

    def test_DataReader_repeat_dynamic_batch(self):
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
        reader = DataReader(
            data=graphs, batch_size=batch_size, repeat=True,
            seed=42, dynamic_batch=True)
        labels_repeat_sum = 0

        for _ in range(num_batches):
            graphs = next(reader)
            labels_batch = graphs.globals
            labels_repeat_sum += np.sum(labels_batch)

        # Check that the accumulated sum of labels is larger than the sum of
        # original labels. This can only be true if the reader is looping.
        self.assertTrue(labels_repeat_sum > np.sum(labels))
        self.assertEqual(len(graphs.n_node), batch_size)

        print(graphs)

        for i, n_node in enumerate(reversed(graphs.n_node)):
            print(f'{i}, {n_node} are the features here')
            if i == 0:
                # last graph in the batch is for padding. We have 5 nodes
                # times 10-1 graphs in a batch is 9*5 = 45 nodes in total. So
                # the last graph should have 14 nodes to batch us to a multiple
                # of 64.
                print('here')
                self.assertTrue(n_node == 19)
            else:
                self.assertTrue(n_node == 5)  # Each graph has 5 nodes.


    def test_DataReader_repeat_static_batch(self):
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
        batch_size = 30
        num_batches = 5  # Number of batches to query and test
        reader = DataReader(
            data=graphs, batch_size=batch_size, repeat=True,
            seed=42, dynamic_batch=False, static_round_to_multiple=False)
        labels_repeat_sum = 0

        for _ in range(num_batches):
            graphs = next(reader)
            labels_batch = graphs.globals
            labels_repeat_sum += np.sum(labels_batch)

        # Check that the accumulated sum of labels is larger than the sum of
        # original labels. This can only be true if the reader is looping.
        self.assertTrue(labels_repeat_sum > np.sum(labels))
        total_nodes = 0
        total_edges = 0
        num_graphs = len(graphs.n_node)
        print(num_graphs)
        self.assertEqual(num_graphs, batch_size)
        print(graphs.n_node)

        # Ok here we have 29 graphs and 1 padded graph in our batch size 20.
        # 29*5 nodes = 145. 256 (power of 2) - 145 (multiple of 64). 
        for i in range(num_graphs):
            if (num_graphs - i) == 1:  # Last index of array:
                self.assertEqual(graphs.n_node[i], 111)
            else:
                self.assertEqual(graphs.n_node[i], 5)
                self.assertEqual(graphs.n_edge[i], 6)

            total_nodes += graphs.n_node[i]
            total_edges += graphs.n_edge[i]
        # Check that the sum of the n_nodes is a power of 2.
        # We perform a bitwise AND on the integer and the integer -1
        # to check if in binary. Ex. 4 is 100 and 3 is 011 so 4 & 3 is 111.
        print(total_nodes)
        self.assertEqual(total_nodes & (total_nodes - 1), 0)
        self.assertEqual(total_edges & (total_edges - 1), 0)


    def test_DataReader_repeat_static_batch_round(self):
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
        batch_size = 30
        num_batches = 5  # Number of batches to query and test
        reader = DataReader(
            data=graphs, batch_size=batch_size, repeat=True,
            seed=42, dynamic_batch=False, static_round_to_multiple=True)
        labels_repeat_sum = 0

        for _ in range(num_batches):
            graphs = next(reader)
            labels_batch = graphs.globals
            labels_repeat_sum += np.sum(labels_batch)

        # Check that the accumulated sum of labels is larger than the sum of
        # original labels. This can only be true if the reader is looping.
        self.assertTrue(labels_repeat_sum > np.sum(labels))
        total_nodes = 0
        total_edges = 0
        num_graphs = len(graphs.n_node)
        print(num_graphs)
        self.assertEqual(num_graphs, batch_size)
        print(graphs.n_node)

        # Ok here we have 29 graphs and 1 padded graph in our batch size 20.
        # 29*5 nodes = 145.  192 (multiple of 64) - 145 (multiple of 64). 
        for i in range(num_graphs):
            if (num_graphs - i) == 1:  # Last index of array:
                self.assertEqual(graphs.n_node[i], 47)
            else:
                self.assertEqual(graphs.n_node[i], 5)
                self.assertEqual(graphs.n_edge[i], 6)

            total_nodes += graphs.n_node[i]
            total_edges += graphs.n_edge[i]
        # Check that the sum of the n_nodes is a power of 2.
        # We perform a bitwise AND on the integer and the integer -1
        # to check if in binary. Ex. 4 is 100 and 3 is 011 so 4 & 3 is 111.
        print(total_nodes)
        self.assertFalse(total_nodes & (total_nodes - 1) == 0)
        self.assertFalse(total_edges & (total_edges - 1) == 0)


    def test_ase_row_to_jraph(self):
        """Test conversion from ase.db.Row to jraph.GraphsTuple."""
        database = ase.db.connect(self.test_db)
        row = database.get(1)
        atomic_numbers = row.toatoms().get_atomic_numbers()
        graph = ase_row_to_jraph(row)
        nodes = graph.nodes
        self.assertIsInstance(
            graph, jraph.GraphsTuple, f"{graph} is not a graph")
        np.testing.assert_array_equal(
            atomic_numbers, nodes, "Atomic numbers are not equal")

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
            n_node=[4], nodes=np.array([1, 1, 1, 6]), n_edge=None, edges=None,
            senders=None, receivers=None, globals=None)
        graph1 = jraph.GraphsTuple(
            n_node=[8], nodes=np.array([1, 1, 1, 1, 1, 1, 6, 6]), n_edge=None,
            edges=None, senders=None, receivers=None, globals=None)
        graph2 = jraph.GraphsTuple(
            n_node=[6], nodes=np.array([1, 1, 1, 1, 6, 8]), n_edge=None,
            edges=None, senders=None, receivers=None, globals=None)
        graphs_dict = {1: graph0, 3: graph1, 4:graph2}
        num_list = get_atom_num_list(graphs_dict)
        graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)

        nodes0_expected = np.array([0, 0, 0, 1])
        nodes1_expected = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        nodes2_expected = np.array([0, 0, 0, 0, 1, 2])
        nodes0 = graphs_dict[1].nodes
        nodes1 = graphs_dict[3].nodes
        nodes2 = graphs_dict[4].nodes
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
