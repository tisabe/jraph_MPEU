"""Test data prep for the ElemNet neural network."""

from absl.testing import absltest
from absl import logging

import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))

# import jraph_MPEU.inference as inf
import other_models.data_prep as dp


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""
    def test_get_elemental_data_of_atom(self):
        """Test get elemental data from csv works."""

        data_prep_obj = dp.DataPrep(
            ase_db_path='/None',
            functional='pbe')
        atomic_number = 1
        expected_ea_half = 1.05595894
        data_prep_ea_half = data_prep_obj.get_elemental_data_of_atom(
            atomic_number)
        self.assertEqual(expected_ea_half, data_prep_ea_half)


if __name__ == "__main__":
    absltest.main()