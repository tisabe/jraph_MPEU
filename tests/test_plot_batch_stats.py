"""Test suite for parsing profiling experiments."""
import csv
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np

from scripts.plotting import plot_batch_stats as pbs


class TestPlotBatchStats(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        pass

    def test_get_csv_name_single_batch(self):
        base_dir = 'tests/data'
        model_type = 'MPEU'
        batching_method = 'static'
        dataset = 'qm9'
        batching_round_to_64 = False
        batch_size = 32
        csv_filename = pbs.get_csv_name_single_batch(
            base_dir, model_type, batching_method, dataset, batching_round_to_64,
            batch_size)
        self.assertEqual(
            csv_filename,
            'tests/data/profiling_experiments/MPEU/qm9/'
            'static/round_False/32/cpu/iteration_0/'
            'graph_distribution_batching_path.csv')
    
    def test_get_stats_for_single_batch_size(self):
        base_dir = 'tests/data'
        model_type = 'MPEU'
        batching_method = 'static'
        dataset = 'qm9'
        batch_stats_col = 'num_node_before_batching'
        batching_round_to_64 = False
        batch_size = 32
        num_nodes_list = pbs.get_stats_for_single_batch_size(
            base_dir, batch_stats_col, batching_method, dataset,
            batching_round_to_64, batch_size, model_type)
        print(num_nodes_list)
        self.assertEqual(num_nodes_list[0], 562)
        self.assertEqual(len(num_nodes_list), 20)


    def test_get_batch_stats_data(self):
        base_dir = 'tests/data'
        model_type = 'MPEU'
        batching_method = 'static'
        dataset = 'qm9'
        batch_stats_col = 'num_node_before_batching'
        batching_round_to_64 = False
        batch_stats_array = pbs.get_batch_stats_data(
                base_dir, batch_stats_col, batching_method, dataset,
                batching_round_to_64,
                model_type)
        print(np.shape(batch_stats_array))
        self.assertEqual(np.shape(batch_stats_array)[0], 4)
        self.assertEqual(np.shape(batch_stats_array)[1], 20)


if __name__ == '__main__':
    unittest.main()