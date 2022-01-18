import unittest
import jraph
import numpy as np

from input_pipeline import DataReader
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

        reader = DataReader(graphs, batch_size, False)
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

        reader = DataReader(graphs, batch_size, True)
        print('This should loop and shuffle')
        print(next(reader).globals)
        print(next(reader).globals)
        print(next(reader).globals)
        # TODO: quantify test


if __name__ == '__main__':
    unittest.main()

