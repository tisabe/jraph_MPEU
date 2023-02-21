"""Test data prep for the ElemNet neural network."""

from absl.testing import absltest
from absl import logging
import numpy as np

import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))

# import jraph_MPEU.inference as inf
import other_models.data_prep as dp


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""

    def setUp(self):
        self.data_prep_obj = dp.DataPrep(
            ase_db_path='/None',
            functional='pbe')

    def test_get_get_csv_val(self):
        """Test getting elemental data from csv works."""

        atomic_number = 1
        feature = 'EA_half'
        expected_ea_half = 1.05595894
        data_prep_ea_half = self.data_prep_obj.get_csv_val(
            atomic_number, feature)
        self.assertEqual(expected_ea_half, data_prep_ea_half)

    def test_get_feature_bar_val(self):
        atomic_number_list = np.array([26, 8])
        fraction_list = np.array([0.4, 0.6])
        feature = 'Atomic number'
        expected_average_atomic_number = 15.2
        average_atomic_number, feature_val_list = self.data_prep_obj.get_feature_bar_val(
            atomic_number_list=atomic_number_list,
            fraction_list=fraction_list, feature=feature)
        self.assertEqual(average_atomic_number, expected_average_atomic_number)


    def test_get_feature_hat_val(self):
        fraction_list = np.array([0.4, 0.6])
        f_bar = 15.2
        feature_val_list = np.array([26, 8])
        expected_deviation_atomic_number = 8.64
        average_deviation_atomic_number = self.data_prep_obj.get_feature_hat_val(
            fraction_list=fraction_list, feature_val_list=feature_val_list, f_bar=f_bar)
        self.assertEqual(
            average_deviation_atomic_number, expected_deviation_atomic_number)

if __name__ == "__main__":
    absltest.main()