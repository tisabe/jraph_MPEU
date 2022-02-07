import unittest
import jax
import jraph
import numpy as np
import ase.db

from input_pipeline import DataReader
from input_pipeline import ase_row_to_jraph
from input_pipeline import asedb_to_graphslist
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

    def test_asedb_to_graphslist_nofilter(self):
        graphs, labels = asedb_to_graphslist('matproj/mp_graphs.db', 
            label_str='delta_e', 
            limit=2000)
        ndim0 = len(np.shape(graphs[0].edges))
        for graph in graphs:
            ndim = len(np.shape(graph.edges))
            if ndim != ndim0:
                raise ValueError('Dimension of edges not 1D')
                break

    
    def test_asedb_to_graphslist_filter(self):
        selection = 'fold>=0'
        graphs, labels = asedb_to_graphslist('matproj/mp_graphs.db', 
            label_str='delta_e', 
            selection=selection, limit=2000)
        print(labels[0:10])
    
if __name__ == '__main__':
    unittest.main()

