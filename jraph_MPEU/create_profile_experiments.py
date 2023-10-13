"""Creates folders, job scripts and config files for profiling tests.

We need to create a seperate folder, config and job script to run each setting
for the profiling tests.

We create a folder with the following structure:

profiling_experiments/network_type/dataset/batch_size/batching_method/compute_type
"""
from absl import flags
from absl import app
import os
from pathlib import Path


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'network_type', 'None',
    'Network types to create and test. Right now either "mpnn" or "schnet".')
flags.DEFINE_list(
    'dataset', 'None',
    'Dataset type to create.')
flags.DEFINE_list(
    'batch_size', 'None',
    'Batch sizes as ints to use for training.')
flags.DEFINE_list(
    'batching_method', 'None',
    'Can be either "static" or "dynamic".')
flags.DEFINE_list(
    'computing_type', 'None',
    'Can be either "gpu:v100", "gpu:a100" or "cpu".')
flags.DEFINE_string(
    'experiment_dir', 'None',
    'Directory for experiments.')


JOB_SCRIPT = """#!/bin/bash -l
#SBATCH -o <folder_name>/%j.out
#SBATCH -e <folder_name>/%j.err
#SBATCH -D <folder_name>/
#SBATCH -J <job_name>
#SBATCH --nodes=1
<constraint>
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32000  # Request 32 GB of main memory per node in MB units.
#SBATCH --mail-type=none
#SBATCH --mail-user=speckhard@fhi.mpg.de
#SBATCH --time=12:00:00
<gres>
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU

# Load the environment with modules and python packages.
source /u/dansp/envs/activate_jax.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python3 scripts/main.py --workdir=<folder_name> --config=<folder_name>/<config_name>
# --config.label_str=<label_str>
"""

TEMPLATE_CONFIG = """
import ml_collections
from jraph_MPEU_configs.default_mp_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    config = get_config_super() # inherit from default mp config
    config.eval_every_steps = 100_000
    config.num_train_steps_max = 1_000_000
    config.log_every_steps = 100_000
    config.checkpoint_every_steps = 1_000_000
    config.selection = None
    config.data_file = <data_file>
    config.label_str = <label_str>
    config.num_edges_max = None
    config.dynamic_batch = <dynamic_batch>
    config.batch_size = <batch_size>
    return config
"""

def create_config_file_path(
        setting, folder_name):
    if setting['batching_method'] == 'dynamic':
        dynamic_batch = True
    else:
        dynamic_batch = False
    config = TEMPLATE_CONFIG.replace(
        '<dynamic_batch>', str(dynamic_batch))
    config = config.replace(
        '<batch_size>', str(setting['batch_size']))
    if setting['dataset'] == 'aflow':
        data_file = 'aflow/graphs_knn24_ICSD_bandgaps_and_fe_11_28.db'
        label_str = 'enthalpy_formation_atom'
        config = config.replace('<data_file>', data_file)
        config = config.replace('<label_str>', label_str)

    folder_name = Path(folder_name)
    config_sub_string = (
        'config_' + setting['dataset'] + '_' + \
            setting['batching_method'] + '_' + \
            str(setting['batch_size']))
    config_file_path = folder_name / (config_sub_string + '.py')
    with open(config_file_path, 'w') as fd:
        fd.write(config)
    return config_file_path

def create_job_script(
        setting,
        config_name,
        folder_name):
    job_script = JOB_SCRIPT.replace(
        '<config_name>', str(config_name))
    job_script = job_script.replace(
        '<folder_name>', str(folder_name))
    
    if setting['computing_type'] in ['gpu:a100', 'gpu:v100']:
        constraint = '#SBATCH --constraint="gpu"\n'
        job_script = job_script.replace(
            '<constraint>', str(constraint))
        gres = '#SBATCH --gres=' + setting['computing_type'] + ':1'
        job_script = job_script.replace(
            '<gres>', str(gres))
    job_script_path_name = Path(folder_name) / 'profiling_job.sh'
    with open(job_script_path_name, 'w') as fd:
        fd.write(job_script)
    return job_script_path_name


def create_folder_for_setting(base_dir, setting):
    """Create a folder for the profiling setting
    
    profiling_experiments/network_type/dataset/batching_method/batch_size/compute_type
    """
    folder_name = (
        str(base_dir) + '/' +
        'profiling_experiments' + '/' +
        setting['network_type'] + '/' +
        setting['dataset'] + '/' +
        setting['batching_method'] + '/' +
        str(setting['batch_size']) + '/' +
        setting['computing_type'] + '/' +
        'iteration_' + str(setting['iteration']))
    if not os.path.isdir(folder_name):
        # Make a directory for this setting.
        os.makedirs(folder_name)
    return folder_name



def get_settings_list(
        network_type_list, dataset_list,
        batch_size_list, batching_method_list,
        computing_type_list):
    """Get a list of n-tuples of settings to use in profiling experiments.

    a list e.g. [("mpnn", "aflow", 32, "static", "gpu:v100"), ...]
    """

    settings_list = []

    for network_type in network_type_list:
        for dataset in dataset_list:
            for batch_size in batch_size_list:
                for batching_method in batching_method_list:
                    for computing_type in computing_type_list:
                        for iteration in range(10):
                            settings_dict = {
                                'network_type': network_type,
                                'dataset': dataset,
                                'batch_size': batch_size,
                                'batching_method': batching_method,
                                'computing_type': computing_type,
                                'iteration': iteration}
                                
                            settings_list.append(settings_dict)
    return settings_list


def create_folder_and_files_for_setting(
            settings_list, base_dir, job_list_path):
    job_path_list = []

    for setting in settings_list:
        folder_name = create_folder_for_setting(
            base_dir, setting)
        config_file_path = create_config_file_path(
            setting, folder_name)
        job_script_path = create_job_script(
            setting, config_file_path, folder_name)
        job_path_list.append(str(job_script_path))


    with open(job_list_path, 'w') as fd:
        fd.write('\n'.join(job_path_list))


def main(argv):
    network_type_list = FLAGS.network_type
    dataset_list = FLAGS.dataset
    batch_size_list = FLAGS.batch_size
    batching_method_list = FLAGS.batching_method
    computing_type_list = FLAGS.computing_type

    settings_list = get_settings_list(
        network_type_list, dataset_list,
        batch_size_list, batching_method_list,
        computing_type_list)
    
    experiment_dir = FLAGS.experiment_dir
    job_list_path = Path(experiment_dir) / "profiling_jobs_list.txt"
    create_folder_and_files_for_setting(
        settings_list, experiment_dir, job_list_path)


if __name__ == '__main__':
    app.run(main)
