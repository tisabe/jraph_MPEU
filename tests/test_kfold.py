"""Work in progress."""

import unittest
import tempfile

import ase.db
from ase import Atoms

from scripts.crossval.train_kfold import (
    get_db_ids,
    get_data_indices_ith_fold
)

class TestKfold(unittest.TestCase):
    """Unittests for kfold functions."""
    def test_get_ids(self):
        """Test that the right ids are returned when applied on and artificial
        database."""
        compound_energy_dict = {
            'H': 0, 'He2': 1, 'Li3': 2, 'Be4': 3, 'B5': 4, 'C6': 5, 'N7': 6,
            'O8': 7}
        with tempfile.TemporaryDirectory() as test_dir:
            test_db = test_dir + '/test.db'
            db = ase.db.connect(test_db)
            for compound, value in compound_energy_dict.items():
                db.write(Atoms(compound), key_value_pairs={'test': value})
            ids = get_db_ids(test_db, selection='test>1', limit=4)
            ids_expected = [3, 4, 5, 6]
            self.assertSequenceEqual(ids, ids_expected)

    def test_get_data_indices_ith_fold(self):
        """Test that from a simple collection of indices, all indices get
        selected once for the test set in one of the folds."""
        n_splits = 5
        seed = 0
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_ids_all = []
        for i_fold in range(n_splits):
            _, test_ids = get_data_indices_ith_fold(
                ids, n_splits, i_fold, seed)
            test_ids_all.extend(test_ids)
        # check that all test ids have been generated and they are unique
        self.assertEqual(len(ids), len(test_ids_all))
        self.assertEqual(len(test_ids_all), len(set(test_ids_all)))


if __name__ == '__main__':
    unittest.main()
