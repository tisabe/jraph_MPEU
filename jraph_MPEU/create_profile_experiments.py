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
    'computing_method', 'None',
    'Can be either "gpu:v100", "gpu:a100" or "cpu".')
flags.DEFINE_string(
    'experiment_dir', 'None',
    'Directory for experiments.')


def get_job_names(submitted_jobs_txt_path):
    """Get names of jobs into an list."""
    with open(submitted_jobs_txt_path, 'r') as fd:
        jod_dir_list = []
        for line in fd:
            job_dir = Path(line.rstrip('\n')).resolve().parents[0]
            jod_dir_list += [str(job_dir)]
    return jod_dir_list


def create_zip_command(job_dir_list, target_zip_file):
    """Get linux command to zip all jod dirs in the list."""
    command = "zip -r"
    command += " " + str(target_zip_file)
    for job_dir in job_dir_list:
        command += " " + str(job_dir)
    return command


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
                        settings_dict = {
                            'network_type': network_type,
                            'dataset': dataset,
                            'batch_size': batch_size,
                            'batching_method': batching_method,
                            'computing_type': computing_type}
                            
                        settings_list.append(settings_dict)
    return settings_list



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
    job_path = Path(experiment_dir) / "profiling_jobs_list.txt"

    job_path_list = []

    for setting in settings_list:
        folder_name = create_folder_for_setting(setting)
        job_script_path = create_job_script(setting, folder_name)
        job_path_list.append(job_script_path)
        config_file_path = create_config_file_path(setting, folder_name)

    with open(job_path, 'w') as fd:
        fd.write(job_path_list)

if __name__ == '__main__':
    app.run(main)
