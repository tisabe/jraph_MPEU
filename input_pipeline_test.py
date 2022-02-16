import unittest
import jax
import jraph
import numpy as np
import ase.db
import pandas as pd

from input_pipeline import DataReader
from input_pipeline import ase_row_to_jraph
from input_pipeline import asedb_to_graphslist
from input_pipeline import atoms_to_nodes_list
from utils import add_labels_to_graphs

class TestHelperFunctions(unittest.TestCase):
    def test_DataReader_no_repeat(self):
        n = 20 # number of graphs
        graph = jraph.GraphsTuple(nodes=np.asarray([0,1,2,3,4]),
                      edges=np.ones((6,2)),
                      senders=np.array([0, 1]), 
                      receivers=np.array([2, 2]),
                      n_node=np.asarray([5]),
                      n_edge=np.asarray([6]),
                      globals=None)
        graphs = [graph] * n
        labels = [i for i in range(n)]
        graphs = add_labels_to_graphs(graphs, labels)
        batch_size = 10

        key = jax.random.PRNGKey(42)
        reader = DataReader(graphs, batch_size, False, key)
        print('This should not loop')
        for batch in reader:
            print(batch.globals)
        # TODO: quantify test

    def test_DataReader_repeat(self):
        n = 20 # number of graphs
        graph = jraph.GraphsTuple(nodes=np.asarray([0,1,2,3,4]),
                      edges=np.ones((6,2)),
                      senders=np.array([0, 1]), 
                      receivers=np.array([2, 2]),
                      n_node=np.asarray([5]),
                      n_edge=np.asarray([6]),
                      globals=None)
        graphs = [graph] * n
        labels = [i for i in range(n)]
        graphs = add_labels_to_graphs(graphs, labels)
        batch_size = 10

        key = jax.random.PRNGKey(42)
        reader = DataReader(graphs, batch_size, True, key)
        print('This should loop and shuffle')
        print(next(reader).globals)
        print(next(reader).globals)
        print(next(reader).globals)
        # TODO: quantify test

    def test_ase_row_to_jraph(self):
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
        files = ['matproj/matproj.db', 'QM9/qm9.db']
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
                    if i==0:
                        keys_list0 = key_value_pairs.keys()
                        #print(keys_list0)
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
        return 0 

    def test_dbs_graphs(self):
        '''Test the ase databases with graph features.'''
        files = ['matproj/mp_graphs_knn.db', 'QM9/qm9_graphs.db']
        limit = None # maximum number of entries that are read
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
                    if i==0:
                        keys_list0 = key_value_pairs.keys()
                        print(keys_list0)
                        #print(data)
                    else:
                        self.assertCountEqual(key_value_pairs.keys(), keys_list0)
                        self.assertCountEqual(data.keys(), data_keys_expected)
                    if len(data.edges)==0:
                        count_no_edges += 1
                        print(row.toatoms())
                print(f'Number of graphs with zero edges: {count_no_edges}')
        return 0
    
    def test_atoms_to_nodes_list(self):
        '''Example: atomic numbers as nodes before:
        [1 1 1 6] Methane
        [1 1 1 1 1 1 6 6] Ethane
        [1 1 1 1 6 8] Carbon Monoxide
        Will be turned into:
        [0 0 0 1]
        [0 0 0 0 0 0 1 1]
        [0 0 0 0 1 2]'''
        graph0 = jraph.GraphsTuple(n_node=[4], nodes=np.array([1,1,1,6]),
            n_edge=None, edges=None, senders=None, receivers=None, globals=None)
        graph1 = jraph.GraphsTuple(n_node=[8], nodes=np.array([1,1,1,1,1,1,6,6]),
            n_edge=None, edges=None, senders=None, receivers=None, globals=None)
        graph2 = jraph.GraphsTuple(n_node=[6], nodes=np.array([1,1,1,1,6,8]),
            n_edge=None, edges=None, senders=None, receivers=None, globals=None)
        graphs = [graph0, graph1, graph2]
        graphs, num_classes = atoms_to_nodes_list(graphs)
        print(graphs)
        nodes0_expected = np.array([0,0,0,1])
        nodes1_expected = np.array([0,0,0,0,0,0,1,1])
        nodes2_expected = np.array([0,0,0,0,1,2])
        nodes0 = graphs[0].nodes
        nodes1 = graphs[1].nodes
        nodes2 = graphs[2].nodes
        np.testing.assert_array_equal(nodes0_expected, nodes0)
        np.testing.assert_array_equal(nodes1_expected, nodes1)
        np.testing.assert_array_equal(nodes2_expected, nodes2)
        self.assertEqual(num_classes, 3)


if __name__ == '__main__':
    unittest.main()

