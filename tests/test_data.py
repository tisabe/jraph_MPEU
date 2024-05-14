"""Test class(es) in the data module."""

import unittest

from jraph_MPEU.data import AsedbDataset


class Unittests(unittest.TestCase):
    """Class for data module unittests."""
    def test_asedbdataset(self):
        """Unittest for AsedbDataset class."""
        dataset = AsedbDataset(
            db_dir='databases/QM9/graphs_fc_vec.db',
            target='U0',
            selection=None,
            limit=10,
            workdir=None,
            globals_strs='lumo'
        )
        print(dataset._data)


if __name__ == '__main__':
    unittest.main()
