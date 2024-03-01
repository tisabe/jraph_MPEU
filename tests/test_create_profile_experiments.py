"""Test creating experiments for profiling."""

import os

from jraph_MPEU import create_profile_experiments as cpe


def test_get_settings_list(
        network_type_list=['mpnn'],
        dataset_list=['aflow'],
        batch_size_list=[16, 32],
        batching_method_list=['static', 'dynamic'],
        static_round_to_multiple_list=['True', 'False'],
        computing_type_list=['cpu']):
    settings_list = cpe.get_settings_list(
        network_type_list, dataset_list,
        batch_size_list, batching_method_list,
        static_round_to_multiple_list,
        computing_type_list
    )
    assert len(settings_list) == 80


def test_create_folder_for_setting(tmp_path):
    folder_base_path = tmp_path / "test_dir"
    folder_base_path.mkdir()
    setting = {
        'network_type': 'mpnn',
        'dataset': 'aflow',
        'batch_size': 32,
        'batching_method': 'static',
        'static_round_to_multiple': True,
        'computing_type': 'cpu',
        'iteration': 1,
    }
    folder_name = cpe.create_folder_for_setting(folder_base_path, setting)
    expected_path = folder_base_path / 'profiling_experiments/mpnn/aflow/static/round_True/32/cpu/iteration_1'
    assert folder_name == str(expected_path)
    assert os.path.isdir(expected_path)


def test_create_config_file_path(tmp_path):
    folder_base_path = tmp_path / "test_dir"
    folder_base_path.mkdir()
    setting = {
        'network_type': 'mpnn',
        'dataset': 'aflow',
        'batch_size': 32,
        'batching_method': 'static',
        'static_round_to_multiple': False,
        'computing_type': 'gpu:a100',
        'iteration': 1,
    }
    config_file_path = cpe.create_config_file_path(setting, folder_base_path)
    assert config_file_path == folder_base_path / 'config_aflow_static_32.py'
    assert os.path.isfile(config_file_path)
    with open(config_file_path, 'r') as fd:
        config = fd.readlines()
        print(config)
        assert '    config.batch_size = 32\n' in config
        assert '    config.dynamic_batch = False\n' in config
        assert '    config.static_round_to_multiple = False\n' in config
        assert "    config.data_file = 'aflow/graphs_knn.db'\n" in config
        assert "    config.label_str = 'enthalpy_formation_atom'\n"
        assert "    config.compute_device = 'gpu_a100'\n"


def test_create_job_script(tmp_path):
    folder_base_path = tmp_path / "test_dir"
    folder_base_path.mkdir()
    setting = {
        'network_type': 'mpnn',
        'dataset': 'aflow',
        'batch_size': 32,
        'batching_method': 'static',
        'static_round_to_multiple': True,
        'computing_type': 'gpu:a100',
        'iteration': 1,
    }
    config_name = 'test_config.py'
    job_script_path_name = cpe.create_job_script(
        setting, str(folder_base_path / config_name), folder_base_path)
    assert job_script_path_name == folder_base_path / 'profiling_job.sh'
    assert os.path.isfile(job_script_path_name)
    with open(job_script_path_name, 'r') as fd:
        job_script = fd.readlines()
        print(job_script)
        assert ('#SBATCH -o ' + str(folder_base_path) + '/%j.out\n') in job_script
        assert '<gres>' not in job_script
        assert '#SBATCH --gres=gpu:a100:4\n' in job_script
        assert ('srun python3.9 scripts/main.py' + ' --workdir=' + str(folder_base_path) +
            ' --config=' + str(folder_base_path) + '/' + 'test_config.py\n') in job_script


def test_create_folder_and_files_for_setting(tmp_path):
    folder_base_path = tmp_path / "test_dir"
    folder_base_path.mkdir()

    # Now create the jobs:
    job_script_folder = tmp_path / 'job_dir'
    job_script_folder.mkdir()
    job_list_path  = job_script_folder / 'jobs_list.txt'

    settings_list = [
        {'network_type': 'mpnn',
        'dataset': 'aflow',
        'batch_size': 32,
        'batching_method': 'static',
        'static_round_to_multiple': True,
        'computing_type': 'gpu:a100',
        'iteration': 1},
        {'network_type': 'mpnn',
        'dataset': 'aflow',
        'batch_size': 64,
        'batching_method': 'static',
        'static_round_to_multiple': False,
        'computing_type': 'gpu:a100',
        'iteration': 2}]

    cpe.create_folder_and_files_for_setting(
        settings_list, folder_base_path, job_list_path)
    assert os.path.isfile(job_list_path)
    with open(job_list_path, 'r') as fd:
        job_files = fd.readlines()
        print(job_files)
        assert len(job_files) == 2
        assert (
            str(folder_base_path) + '/profiling_experiments/mpnn/aflow/' + \
            'static/round_True/32/gpu_a100/iteration_1/profiling_job.sh\n') in job_files
        assert (
            str(folder_base_path) + '/profiling_experiments/mpnn/aflow/' + \
            'static/round_False/64/gpu_a100/iteration_2/profiling_job.sh') in job_files
    # Open one of the job paths and check the job script is good.
    with open(
        str(folder_base_path) + \
        '/profiling_experiments/mpnn/aflow/' + \
        'static/round_True/32/gpu_a100/iteration_1/profiling_job.sh', 'r') as fd:
        job_script = fd.readlines()
        print(job_script)
        assert '#SBATCH --gres=gpu:a100:4\n' in job_script
        expected_srun_statement = (
            'srun python3.9 scripts/main.py' + ' --workdir=' + str(folder_base_path) + \
            '/profiling_experiments/mpnn/aflow/static/round_True/32/gpu_a100/iteration_1' + \
            ' --config=' + str(folder_base_path) + '/' + \
            'profiling_experiments/mpnn/aflow/static/round_True/32/gpu_a100/iteration_1/config_aflow_static_32.py\n')
        print(expected_srun_statement)
        assert expected_srun_statement in job_script