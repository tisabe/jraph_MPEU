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

sample_full_err_file = """
2023-11-08 09:05:43.807428: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Error.  nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64)I1108 09:05:48.392958 22597638805312 main.py:51] JAX host: 0 / 1
I1108 09:05:48.393078 22597638805312 main.py:52] JAX local devices: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]
I1108 09:05:48.393294 22597638805312 train.py:538] Loading datasets.
I1108 09:05:48.395299 22597638805312 input_pipeline.py:268] Number of entries selected: 2000
I1108 09:05:49.484347 22597638805312 input_pipeline.py:616] Mean: -3.1217995700000003, Std: 0.5050800478821303
I1108 09:05:49.492709 22597638805312 train.py:540] Number of node classes: 46
I1108 09:05:49.665660 22597638805312 train.py:464] Initializing network.
I1108 09:05:51.900247 22597638805312 train.py:564] 175776 params, size: 0.70MB
I1108 09:05:51.901073 22597638805312 train.py:580] Starting training.
I1108 09:05:51.904885 22597638805312 train.py:72] LOG Message: Recompiling!
2023-11-08 09:05:54.592684: W external/xla/xla/service/gpu/buffer_comparator.cc:1054] INTERNAL: ptxas exited with non-zero error code 65280, output: ptxas /tmp/tempfile-ravg1069-c170fb0-41501-6099f8e722b30, line 10; fatal   : Unsupported .version 7.8; current version is '7.4'
ptxas fatal   : Ptx assembly aborted due to errors

Relying on driver to perform ptx compilation. 
Setting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda  or modifying $PATH can be used to set the location of ptxas
This message will only be logged once.
I1108 09:18:47.202389 22597638805312 train.py:605] Step 100000 train loss: 4.598790474119596e-05
I1108 09:18:50.715722 22597638805312 train.py:349] RMSE/MAE train: [0.0042509  0.00319617]
I1108 09:18:50.715964 22597638805312 train.py:349] RMSE/MAE validation: [0.08130376 0.02019022]
I1108 09:18:50.716107 22597638805312 train.py:349] RMSE/MAE test: [0.6269935  0.05686472]
I1108 09:31:42.567523 22597638805312 train.py:605] Step 400000 train loss: 3.301641118014231e-05
I1108 09:31:42.750351 22597638805312 train.py:349] RMSE/MAE train: [0.00317097 0.00237973]
I1108 09:31:42.750577 22597638805312 train.py:349] RMSE/MAE validation: [0.06572255 0.01539369]
I1108 09:31:42.750722 22597638805312 train.py:349] RMSE/MAE test: [0.62795682 0.05516521]
I1108 09:44:31.191949 22597638805312 train.py:605] Step 600000 train loss: 5.473248165799305e-06
I1108 09:44:31.373494 22597638805312 train.py:349] RMSE/MAE train: [0.00145899 0.00080222]
I1108 09:44:31.373725 22597638805312 train.py:349] RMSE/MAE validation: [0.06393591 0.01392511]
I1108 09:44:31.373871 22597638805312 train.py:349] RMSE/MAE test: [0.62778498 0.0542569 ]
I1108 09:57:19.871214 22597638805312 train.py:605] Step 800000 train loss: 5.320032869349234e-06
I1108 09:57:20.042230 22597638805312 train.py:349] RMSE/MAE train: [0.00119142 0.0008522 ]
I1108 09:57:20.042458 22597638805312 train.py:349] RMSE/MAE validation: [0.06556004 0.01431762]
I1108 09:57:20.042606 22597638805312 train.py:349] RMSE/MAE test: [0.62869476 0.05487633]
I1108 10:10:05.896673 22597638805312 train.py:145] Serializing experiment state to /u/dansp/batching/profiling_experiments/mpnn/aflow/dynamic/32/gpu_a100/iteration_3/checkpoints/checkpoint_1000000.pkl
I1108 10:10:05.901463 22597638805312 train.py:605] Step 1000000 train loss: 1.3091990922475816e-06
I1108 10:10:06.075805 22597638805312 train.py:349] RMSE/MAE train: [0.00045792 0.00033444]
I1108 10:10:06.076016 22597638805312 train.py:349] RMSE/MAE validation: [0.06635462 0.01384054]
I1108 10:10:06.076157 22597638805312 train.py:349] RMSE/MAE test: [0.62898028 0.05483346]
I1108 10:10:06.080812 22597638805312 train.py:623] Reached maximum number of steps without early stopping.
I1108 10:10:06.081199 22597638805312 train.py:630] Lowest validation loss: 0.06393590794230612
I1108 10:10:06.133434 22597638805312 train.py:633] Mean batching time: 0.0009820856218338012
I1108 10:10:06.171946 22597638805312 train.py:636] Mean update time: 0.003449098623037338
"""

sample_err_file_cancelled = sample_err_file + '\nCANCELLED DUE TO TIME LIMIT'

sample_err_file_cancelled_nothing_else = '\nCANCELLED DUE TO TIME LIMIT'


PATHS_TEXT_FILE = '/home/dts/Documents/hu/jraph_MPEU/tests/data/fake_profile_paths.txt'

class ParseProfileExperiments(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')
            paths_txt_file = PATHS_TEXT_FILE
            csv_filename = None
            self.profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file,
                csv_filename,
                save_directory
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
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                temp_file_name, data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 1)
            self.assertEqual(data_dict['experiment_completed'], 1)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_1000000_train_rmse'], 0.0019891)
            self.assertEqual(data_dict['step_1000000_val_rmse'], 0.06648617)
            self.assertEqual(data_dict['step_1000000_test_rmse'], 0.6293168)
            self.assertEqual(data_dict['step_1000000_batching_time_mean'], 0.0005979892456531525)
            self.assertEqual(data_dict['step_1000000_update_time_mean'], 0.0026955132026672364)


    def test_get_recompile_and_timing_info_full(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file_full.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_full_err_file)
            # Now test reading the file.
            data_dict = {}
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                temp_file_name, data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 1)
            self.assertEqual(data_dict['experiment_completed'], 1)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_100000_train_rmse'], 0.0042509)
            self.assertEqual(data_dict['step_100000_val_rmse'], 0.08130376)
            self.assertEqual(data_dict['step_1000000_batching_time_mean'], 0.0009820856218338012)
            self.assertEqual(data_dict['step_1000000_update_time_mean'], 0.003449098623037338)

    def test_update_dict_with_batching_method_size(self):
        """Test updating the dict with a batching method.
        
        
        schnet/qm9/static/round_True/16/gpu_a100/iteration_5
        """
        parent_path = 'tests/data/mpnn/aflow/dynamic/round_True/64/gpu_a100/iteration_5'
        data_dict = {}
        data_dict = self.profiling_parser_object.update_dict_with_batching_method_size(
            parent_path, data_dict)
        self.assertEqual(data_dict['batching_type'], 'dynamic')
        self.assertEqual(data_dict['iteration'], 5)
        self.assertEqual(data_dict['batching_round_to_64'], 'True')
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
            calc_ran_bool, most_recent_error_file = self.profiling_parser_object.check_experiment_ran(
                temp_file_name, parent_path)
            self.assertEqual(calc_ran_bool, True)
            self.assertEqual(most_recent_error_file, temp_file_name)


    def test_check_sim_time_lim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name_cancelled = os.path.join(tmp_dir, 'sample_err_file_cancelled.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file_cancelled)
            calc_expired = self.profiling_parser_object.check_sim_time_lim(
                temp_file_name_cancelled)
            self.assertEqual(calc_expired, True)

            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            calc_expired = self.profiling_parser_object.check_sim_time_lim(
                temp_file_name)
            self.assertEqual(calc_expired, False)


    def test_gather_all_path_data_expired(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_False/64/gpu_a100/'
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

            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=tmp_dir)

            data_dict = profiling_parser_object.gather_all_path_data(
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['submission_path'],
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['iteration'], 5)
            self.assertEqual(
                data_dict['batching_round_to_64'], 'False')

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
            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=csv_file_name,
                save_directory=tmp_dir)
            profiling_parser_object.create_header()

            # Now open the CSV and count how many lines are there.
            df = pd.read_csv(csv_file_name)

            self.assertEqual(len(df.columns), 19)

    def test_write_all_path_data(self):
        """Test writing data for a submission path as a row to csv and db."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_False/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_full_err_file)
        

            csv_file_name = os.path.join(
                tmp_dir,
                'output.csv')

            paths_text_file = os.path.join(tmp_dir, 'path_text_file.csv')
            # Create a paths text file:
            with open(paths_text_file, 'w') as txt_file:
                txt_file.write(temp_file_name_cancelled)

            with open(paths_text_file, 'w') as txt_file:
                txt_file.write(temp_file_name_cancelled)

            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=paths_text_file,
                csv_filename=csv_file_name,
                save_directory=tmp_dir)

            # paths_to_resubmit = os.path.join(
            #     tmp_dir,
            #     'paths_to_resubmit.txt')

            if not os.path.isfile(csv_file_name):
                profiling_parser_object.create_header()
            with open(csv_file_name, 'a') as csv_file:
                dict_writer = csv.DictWriter(
                    csv_file, fieldnames=profiling_parser_object.csv_columns, extrasaction='ignore')
                profiling_parser_object.write_all_path_data(dict_writer)
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

            self.assertEqual(
                df['step_100000_train_rmse'].values[0],
                0.0042509)


    def test_get_recompile_and_timing_info_failed(self):
        # Write the sample text to file:
        failing_err_file = 'tests/data/profiling_err_file_failing.err'

        # Now test reading the file.
        data_dict = {}
        data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
            failing_err_file, data_dict)
        self.assertEqual(data_dict['recompilation_counter'], 60)
        self.assertEqual(data_dict['experiment_completed'], True)
        # Test that we were able to get MAE/RMSE info
        self.assertEqual(data_dict['step_100000_train_rmse'], 0.47112376)
        self.assertEqual(data_dict['step_100000_val_rmse'], 0.466972)

        self.assertEqual(data_dict['step_100000_test_rmse'], 0.46532637)
        self.assertEqual(data_dict['step_100000_batching_time_mean'], 0.0005266756558418273)
        self.assertEqual(data_dict['step_100000_update_time_mean'], 0.0031542533135414125)


    def test_check_experiment_ran_empty_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_Talse/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')

            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=save_directory)
            # Now do not create a file in the folder and see what happens when we parse it.
            # temp_file_name_cancelled = os.path.join(
            #     temp_dir_for_err_file,
            #     'sample_err_file.err')
            # with open(temp_file_name_cancelled, 'w') as fd:
            #     fd.write(sample_full_err_file)
            with self.assertRaises(ValueError):
                profiling_parser_object.check_experiment_ran(
                    os.path.join(temp_dir_for_err_file, 'submission_MgO.sh'), temp_dir_for_err_file)
            with open(
                    profiling_parser_object.paths_resubmit_from_scratch, 'r') as fo:
                    parsed_resubmit_paths = fo.readlines()
                    print(parsed_resubmit_paths)
                    self.assertEqual(len(parsed_resubmit_paths), 1)
                    self.assertEqual(parsed_resubmit_paths[0], os.path.join(temp_dir_for_err_file, 'submission_MgO.sh\n'))

    def test_parsing_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_Talse/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')

            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=save_directory)
            # Now do not create a file in the folder and see what happens when we parse it.
            # temp_file_name_cancelled = os.path.join(
            #     temp_dir_for_err_file,
            #     'sample_err_file.err')
            # with open(temp_file_name_cancelled, 'w') as fd:
            #     fd.write(sample_full_err_file)
            with self.assertRaises(ValueError):
                profiling_parser_object.gather_all_path_data(
                    os.path.join(temp_dir_for_err_file, 'submission_MgO.sh'))

            with open(
                    profiling_parser_object.paths_resubmit_from_scratch, 'r') as fo:
                parsed_resubmit_paths = fo.readlines()
                print(parsed_resubmit_paths)
                self.assertEqual(len(parsed_resubmit_paths), 1)
                self.assertEqual(parsed_resubmit_paths[0], os.path.join(temp_dir_for_err_file, 'submission_MgO.sh\n'))

if __name__ == '__main__':
    unittest.main()
