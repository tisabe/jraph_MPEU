"""Test suite for parsing profiling experiments."""
from pathlib import Path
import os
import tempfile

from jraph_MPEU import parse_profile_experiments as ppe

import unittest


sample_err_file = """
        I1107 20:07:21.992192 22949931718464 train.py:72] LOG Message: Recompiling!
        07 20:59:07.525845 22949931718464 train.py:605] Step 1000000 train loss: 1.8053530084216618e-06
        I1107 20:59:07.705219 22949931718464 train.py:349] RMSE/MAE train: [0.0019891  0.00134341]
        I1107 20:59:07.705431 22949931718464 train.py:349] RMSE/MAE validation: [0.06648617 0.01503589]
        I1107 20:59:07.705571 22949931718464 train.py:349] RMSE/MAE test: [0.6293168  0.05498861]
        I1107 20:59:07.709884 22949931718464 train.py:623] Reached maximum number of steps without early stopping.
        I1107 20:59:07.710278 22949931718464 train.py:630] Lowest validation loss: 0.0664861650789436
        I1107 20:59:07.765833 22949931718464 train.py:633] Mean batching time: 0.0005979892456531525
        I1107 20:59:07.805134 22949931718464 train.py:636] Mean update time: 0.0026955132026672364
"""

sample_err_file_cancelled = sample_err_file + '\nCANCELLED DUE TO TIME LIMIT'


class UnitTests(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        paths_text_file = '/home/dts/Documents/hu/jraph_MPEU/tests/data/fake_profile_paths.txt'
        csv_filename = None
        db_name = None
        paths_to_resubmit = None
        paths_misbehaving = None
        self.output_parser_object = ppe.OutputParser(
            paths_text_file,
            csv_filename,
            db_name,
            paths_to_resubmit,
            paths_misbehaving
        )

    #TODO(dts): figure out what happens to the CSV writer if some fields
    # are missing ideally, I would like to write nan to those fields.
    def test_get_recompile_and_timing_info(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            # Now test reading the file.
            data_dict = {}
            data_dict = self.output_parser_object.get_recompile_and_timing_info(
                temp_file_name, data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 1)
            self.assertEqual(data_dict['experiment_completed'], 1)
    
    def test_update_dict_with_batching_method_size(self):
        """Test updating the dict with a batching method."""
        parent_path = 'tests/data/mpnn/aflow/dynamic/64/gpu_a100/iteration_5'
        data_dict = {}
        data_dict = self.output_parser_object.update_dict_with_batching_method_size(
            parent_path, data_dict)
        self.assertEqual(data_dict['batching_type'], 'dynamic')
        self.assertEqual(data_dict['iteration'], 5)
        self.assertEqual(data_dict['computing_type'], 'gpu_a100')
        self.assertEqual(data_dict['batch_size'], 64)
        self.assertEqual(data_dict['model'], 'mpnn')
        self.assertEqual(data_dict['dataset'], 'aflow')
    

    def test_check_calc_finished(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            parent_path = Path(temp_file_name).parent.absolute()
            calc_ran_bool, most_recent_error_file = self.output_parser_object.check_experiment_ran(
                parent_path)
            self.assertEqual(calc_ran_bool, True)
            self.assertEqual(most_recent_error_file, temp_file_name)


    def test_check_sim_time_lim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name_cancelled = os.path.join(tmp_dir, 'sample_err_file_cancelled.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file_cancelled)
            calc_expired = self.output_parser_object.check_sim_time_lim(
                temp_file_name_cancelled)
            self.assertEqual(calc_expired, True)

            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            calc_expired = self.output_parser_object.check_sim_time_lim(
                temp_file_name)
            self.assertEqual(calc_expired, False)



if __name__ == '__main__':
    unittest.main()
