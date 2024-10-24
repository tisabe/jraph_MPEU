"""Test suite for parsing profiling experiments."""
import csv
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.plotting import plot_longer_batching_experiments as pble


class ParseProfileExperiments(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        pass

    def test_get_column_list(self):
        # Write the sample text to file:
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
        #     with open(temp_file_name, 'w') as fd:
        #         fd.write(sample_err_file)
        data_split = 'train'
        column_list = pble.get_column_list(data_split)
        self.assertEqual(column_list[0], 'step_100_000_train_rmse')
        self.assertEqual(column_list[-1], 'step_2_000_000_train_rmse')
        self.assertEqual(len(column_list), 20)

    def test_get_avg_std_rmse(self):
        # column_list = pble.get_column_list(data_split)
        test_df = pd.DataFrame({
            'model': ['schnet', 'schnet', 'schnet', 'schnet'],
            'batching_type': ['dynamic', 'dynamic', 'dynamic', 'dynamic'],
            'dataset': ['aflow', 'aflow', 'aflow', 'aflow'],
            'batch_size': [32, 32, 32, 32],
            'computing_type': ['gpu', 'gpu', 'gpu', 'gpu'],
            'step_100_000_train_rmse': [2, 3, 4, 5],
            'step_200_000_train_rmse': [2, 3, 4, 5],
            'step_300_000_train_rmse': [2, 3, 4, 5],
            'step_400_000_train_rmse': [2, 3, 4, 5],
            'step_500_000_train_rmse': [2, 3, 4, 5],
            'step_600_000_train_rmse': [2, 3, 4, 5],
            'step_700_000_train_rmse': [2, 3, 4, 5],
            'step_800_000_train_rmse': [2, 3, 4, 5],
            'step_900_000_train_rmse': [2, 3, 4, 5],
            'step_1_000_000_train_rmse': [2, 3, 4, 5],
            'step_1_100_000_train_rmse': [2, 3, 4, 5],
            'step_1_200_000_train_rmse': [2, 3, 4, 5],
            'step_1_300_000_train_rmse': [2, 3, 4, 5],
            'step_1_400_000_train_rmse': [2, 3, 4, 5],
            'step_1_500_000_train_rmse': [2, 3, 4, 5],
            'step_1_600_000_train_rmse': [2, 3, 4, 5],
            'step_1_700_000_train_rmse': [2, 3, 4, 5],
            'step_1_800_000_train_rmse': [2, 3, 4, 5],
            'step_1_900_000_train_rmse': [2, 3, 4, 5],
            'step_2_000_000_train_rmse': [2, 3, 4, 5],
        })
        df_mean, df_std = pble.get_avg_std_rmse(
            df=test_df, model='schnet',
            batch_method='dynamic',
            compute_type='gpu',
            batch_size=32,
            dataset='aflow', data_split='train')
        
        print(df_mean)
        print(df_std)
        self.assertAlmostEqual(df_mean['step_1_900_000_train_rmse'], 3.5, 5)
        self.assertAlmostEqual(df_std['step_1_900_000_train_rmse'], 1.290994, 5)

if __name__ == '__main__':
    unittest.main()