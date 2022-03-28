import unittest

import numpy as np

from train_crossval import (
    get_data_indices_ith_fold
)

class TestHelperFunctions(unittest.TestCase):
    def test_get_data_indices_ith_fold(self):
        N = 11
        n_splits = 5
        val_indices = [] # collect validation indices
        for i_fold in range(n_splits):
            train_index, val_index = get_data_indices_ith_fold(N, n_splits, i_fold, 0)
            self.assertIsInstance(train_index, list)
            self.assertIsInstance(val_index, list)
            for index in val_index:
                val_indices.append(index)
        # test that each validation index appears once in val_indices
        self.assertCountEqual(val_indices, range(N))


if __name__ == '__main__':
    unittest.main()