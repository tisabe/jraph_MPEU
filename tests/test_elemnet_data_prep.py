"""Test data prep for the ElemNet neural network."""

from absl.testing import absltest
from absl import logging
import ase
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
        """Test getting the feature average value"""
        atomic_number_list = np.array([26, 8])
        fraction_list = np.array([0.4, 0.6])
        feature = 'Atomic number'
        expected_average_atomic_number = 15.2
        average_atomic_number, feature_val_list = self.data_prep_obj.get_feature_bar_val(
            atomic_number_list=atomic_number_list,
            fraction_list=fraction_list, feature=feature)
        self.assertEqual(average_atomic_number, expected_average_atomic_number)
        self.assertListEqual(list(feature_val_list), list(atomic_number_list))


    def test_get_atomic_number_and_fraction_list(self):
        """Test getting the unique atomic numbers and fraction list"""
        compound_name = 'Fe2O3'
        ase_atoms_obj = ase.Atoms(compound_name)
        atomic_number_list, fraction_list = self.data_prep_obj.get_atomic_number_and_fraction_list(
            ase_atoms_obj)
        expected_atomic_number_list = [26, 8]
        expected_fraction_list = [0.4, 0.6]
        self.assertListEqual(expected_atomic_number_list, atomic_number_list)
        self.assertListEqual(expected_fraction_list, fraction_list)

    def test_get_feature_hat_val(self):
        """Test getting the feature value deviation."""
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