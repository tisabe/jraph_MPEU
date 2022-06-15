"""Test the functions of the input_pipeline_test.py module."""
import tempfile

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
    add_splits_to_database,
    save_split_dict,
    load_split_dict
)
from jraph_MPEU.utils import add_labels_to_graphs

class TestPipelineFunctions(unittest.TestCase):
    """Testing class."""
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
        print(dict_loaded)
        for key, ids in test_dict.items():
            for i in ids:
                self.assertEqual(key, dict_loaded[i])

    def test_add_splits_to_database(self):
        """Note: might get rid of this function"""
        config = ml_collections.ConfigDict()
        config.train_frac = 0.5
        config.val_frac = 0.3
        config.test_frac = 0.2
        num_rows = 20  # number of rows to write
        with tempfile.TemporaryDirectory() as test_dir:  # directory for database
            config.data_file = test_dir + 'test.db'
            # create and connect to temporary database
            db = ase.db.connect(config.data_file)
            for _ in range(num_rows):
                h2 = Atoms('H2', [(0, 0, 0), (0, 0, 0.7)])  # example structure
                db.write(h2)
            add_splits_to_database(config, num_rows)
            split_dict = {'train': [], 'validation': [], 'test': []}
            for row in db.select():
                split_dict[row.split].append(row.id)
            np.testing.assert_array_equal(
                split_dict['train'], sorted([6, 15, 10, 8, 17, 12, 4, 1, 16, 13]))
            np.testing.assert_array_equal(
                split_dict['validation'], sorted([19, 9, 2, 14, 5, 18]))
            np.testing.assert_array_equal(
                split_dict['test'], sorted([20, 3, 7, 11]))

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
        db_names = [
            'matproj/mp_graphs.db',
            'matproj/mp_graphs_knn.db',
            'aflow/graphs_cutoff_6A.db',
            'QM9/qm9_graphs.db'
        ]
        for db_name in db_names:
            first_row = None
            db = ase.db.connect(db_name)
            for i, row in enumerate(db.select(limit=10)):
                if i == 0:
                    first_row = row
            cutoff_type = first_row['cutoff_type']
            cutoff_val = first_row['cutoff_val']
            print(f'db name: {db_name}, cutoff type: {cutoff_type}, \
                cutoff val: {cutoff_val}')

    def test_asedb_to_graphslist(self):
        """Test converting an asedb to a list of jraph.GraphsTuple.

        Note: for this test, the Materials Project data has to be downloaded
        beforehand, using the scipt get_matproj.py, and converted to graphs
        using asedb_to_graphs.py. The database has to be located at
        matproj/mp_graphs.db relative to the path of this test.

        Test case:
            file: matroj/mp_graphs.db
            label_str: delta_e
            selection: None
            limit: 100
        """

        file_str = 'matproj/mp_graphs.db'
        label_str = 'delta_e'
        selection = 'delta_e<0'
        limit = 100
        graphs, labels = asedb_to_graphslist(
            file=file_str, label_str=label_str, selection=selection, limit=limit)
        _ = [self.assertIsInstance(graph, jraph.GraphsTuple) for graph in graphs]
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
        db = ase.db.connect('matproj/mp_graphs.db')
        row = db.get('mp_id=mp-1001')
        atomic_numbers = row.toatoms().get_atomic_numbers()
        graph = ase_row_to_jraph(row)
        nodes = graph.nodes
        self.assertIsInstance(graph, jraph.GraphsTuple)
        np.testing.assert_array_equal(atomic_numbers, nodes)

    '''
    def test_asedb_to_graphslist_filter(self):
        selection = 'fold>=0'
        graphs, labels = asedb_to_graphslist('matproj/mp_graphs.db', 
            label_str='delta_e', 
            selection=selection, limit=2000)
        print(labels[0:10])
        ndim0 = len(np.shape(graphs[0].edges))
        for graph in graphs:
            ndim = len(np.shape(graph.edges))
            if ndim != ndim0:
                raise ValueError('Dimension of edges not 1D')
                break
    '''

    def test_dbs_raw(self):
        '''Test the raw ase databases without graph features.'''
        files = ['matproj/matproj.db', 'QM9/qm9.db', ]
        limit = 100 # maximum number of entries that are read
        if not limit is None:
            print(f'Testing {limit} graphs. To test all graphs, change limit to None.')
        for file in files:
            print(f'Testing data in {file}')
            with ase.db.connect(file) as asedb:
                keys_list0 = None
                for i, row in enumerate(asedb.select(limit=limit)):
                    key_value_pairs = row.key_value_pairs
                    self.assertIsInstance(key_value_pairs, dict)
                    # check that all the keys are the same
                    if i == 0:
                        keys_list0 = key_value_pairs.keys()
                        #print(keys_list0)
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
        return 0

    def test_dbs_graphs(self):
        '''Test the ase databases with graph features.'''
        files = ['matproj/mp_graphs.db', 'QM9/qm9_graphs.db', 'aflow/graphs_cutoff_6A.db']
        limit = 100 # maximum number of entries that are read
        if not limit is None:
            print(f'Testing {limit} graphs. To test all graphs, change limit to None.')
        for file in files:
            print(f'Testing data in {file}')
            with ase.db.connect(file) as asedb:
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
                        print(keys_list0)
                        #print(data)
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
                        self.assertCountEqual(data.keys(), data_keys_expected)
                    if len(data.edges) == 0:
                        count_no_edges += 1
                        print(row.toatoms())
                print(f'Number of graphs with zero edges: {count_no_edges}')
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
        graphs = [graph0, graph1, graph2]
        graphs, num_list = atoms_to_nodes_list(graphs)

        nodes0_expected = np.array([0, 0, 0, 1])
        nodes1_expected = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        nodes2_expected = np.array([0, 0, 0, 0, 1, 2])
        nodes0 = graphs[0].nodes
        nodes1 = graphs[1].nodes
        nodes2 = graphs[2].nodes
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
