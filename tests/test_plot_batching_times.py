"""Test suite for parsing profiling experiments."""
import csv
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.plotting import plot_batching_times as pbt

BASE_DIR = '/home/dts/Documents/hu'
COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_16_05__12_12_2024.csv'

class PlotProfileExperiments(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        self.test_csv = os.path.join(BASE_DIR, COMBINED_CSV)
        self.df = pd.read_csv(self.test_csv)

    def test_get_avg_std_of_profile_column(self):
        profile_column = 'batching'
        model = 'MPEU'
        batch_method = 'static'
        compute_type = 'gpu_a100'
        batch_size = 16
        dataset = 'aflow'
        batching_round_to_64 = True
        print(self.df)
        mean_result, std_result = pbt.get_avg_std_of_profile_column(
            self.df, profile_column, model, batch_method, compute_type,
            batch_size, dataset, batching_round_to_64)
        self.assertTrue(mean_result == 0.0005044880275725999)
        self.assertTrue(std_result == 7.667239738478865e-06)


if __name__ == '__main__':
    unittest.main()