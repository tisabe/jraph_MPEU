import numpy as np
from ase import Atoms

import warnings
import unittest

from datahandler import *

class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms('N4',
            [(0,0,0), (2,2,0), (2,1,0), (3,1,0)])

    def test_knearest(self):
        k = 2
        nodes, pos, edges, senders, receivers = get_graph_knearest(
            self.atoms, k)
        np.testing.assert_array_equal(nodes, [7,7,7,7])
        np.testing.assert_array_equal(pos, self.atoms.get_positions())
        np.testing.assert_array_equal(edges, np.sqrt([5, 8, 1, 2, 1, 1, 1, 2]))
        np.testing.assert_array_equal(receivers, [0,0,1,1,2,2,3,3])
        np.testing.assert_array_equal(senders, [2,1,2,3,3,1,2,1])
        
    def test_cutoff(self):
        cutoff = 2.5
        nodes, pos, edges, senders, receivers = get_graph_cutoff(
            self.atoms, cutoff)
        np.testing.assert_array_equal(nodes, [7,7,7,7])
        np.testing.assert_array_equal(pos, self.atoms.get_positions())
        np.testing.assert_array_equal(edges, np.sqrt([5,1,2,1,5,1,2,1]))
        np.testing.assert_array_equal(receivers, [0,1,1,2,2,2,3,3])
        np.testing.assert_array_equal(senders, [2,2,3,3,0,1,1,2])
        

if __name__ == '__main__':
    unittest.main()
