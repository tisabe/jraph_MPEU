"""Test suite for parsing profiling experiments."""
import csv
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from jraph_MPEU import parse_profile_experiments as ppe


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

sample_err_file_cancelled_nothing_else = '\nCANCELLED DUE TO TIME LIMIT'


PATHS_TEXT_FILE = '/home/dts/Documents/hu/jraph_MPEU/tests/data/fake_profile_paths.txt'

class UnitTests(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        paths_text_file = PATHS_TEXT_FILE
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


    def test_gather_all_path_data_expired(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file_cancelled_nothing_else.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file_cancelled_nothing_else)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            output_parser_object = ppe.OutputParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                db_name=None,
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)

            data_dict = output_parser_object.gather_all_path_data(
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['submission_path'],
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['iteration'], 5)

    def test_create_header(self):
        """Test writing the header to the CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            csv_file_name = os.path.join(
                temp_dir_for_err_file,
                'output.csv')
            output_parser_object = ppe.OutputParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=csv_file_name,
                db_name=None,
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)
            output_parser_object.create_header()

            # Now open the CSV and count how many lines are there.
            df = pd.read_csv(csv_file_name)

            self.assertEqual(len(df.columns), 36)

    def test_write_all_path_data(self):
        """Test writing data for a submission path as a row to csv and db."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            csv_file_name = os.path.join(
                tmp_dir,
                'output.csv')

            paths_text_file = os.path.join(tmp_dir, 'path_text_file.csv')
            # Create a paths text file:
            with open(paths_text_file, 'w') as txt_file:
                txt_file.write(temp_file_name_cancelled)

            output_parser_object = ppe.OutputParser(
                paths_txt_file=paths_text_file,
                csv_filename=csv_file_name,
                db_name=None,
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)
        
            if not os.path.isfile(csv_file_name):
                output_parser_object.create_header()
            with open(csv_file_name, 'a') as csv_file:
                dict_writer = csv.DictWriter(
                    csv_file, fieldnames=output_parser_object.csv_columns)
                output_parser_object.write_all_path_data(dict_writer)
            df = pd.read_csv(csv_file_name)
            self.assertEqual(
                df['path'].values[0],
                temp_file_name_cancelled)
            self.assertEqual(
                df['experiment_completed'].values[0],
                True)
            self.assertEqual(
                df['recompilation_counter'].values[0],
                1)
            # self.assertEqual(
            #     df['step_400000_train_rmse'].values[0],
            #     np.nan)
            self.assertEqual(
                df['step_1000000_train_rmse'].values[0],
                0.0019891)
            self.assertAlmostEqual(
                df['step_1000000_batching_time'].values[0],
                0.000597989245653152, 10
            )
            self.assertAlmostEqual(
                df['step_1000000_update_time'].values[0],
                0.0026955132026672364, 10
            )

if __name__ == '__main__':
    unittest.main()
