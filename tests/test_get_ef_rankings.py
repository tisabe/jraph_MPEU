"""Test get formation energy rankings script."""
import os
import tempfile
import unittest
import pandas as pd
from scripts.data.get_ef_rankings import get_correct_min_ef_predicted


class TestModelPolymorphRankings(unittest.TestCase):
    """Test inference.py methods and classes."""

    def test_get_ef_ranking(self):
        """Test case: no dropout, scalar predictions"""
        expected_correct = 2
        df = pd.read_csv('tests/data/fake_ef.csv')
        correct_num = get_correct_min_ef_predicted(df)
        assert correct_num == expected_correct


if __name__ == '__main__':
    unittest.main()