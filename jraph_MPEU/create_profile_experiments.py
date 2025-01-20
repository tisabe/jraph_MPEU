"""Creates folders, job scripts and config files for profiling tests.

We need to create a seperate folder, config and job script to run each setting
for the profiling tests.

We create a folder with the following structure:

profiling_experiments/network_type/dataset/batch_size/batching_method/compute_type
"""
from absl import flags
from absl import app
import ast
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
    'static_round_to_multiple', 'False',
    'Round static batching to multiple or power.')
flags.DEFINE_list(
    'static_constant_batch', 'False',
    'Static batch to a constant padding size?'
)
flags.DEFINE_list(
    'batching_method', 'None',
    'Can be either "static" or "dynamic".')
flags.DEFINE_list(
    'computing_type', 'None',
    'Can be either "gpu:v100", "gpu:a100" or "cpu".')
flags.DEFINE_string(
    'experiment_dir', 'None',
    'Directory for experiments.')
flags.DEFINE_string(
    'number_of_training_steps', 'None',
    'How many training steps to run.')
flags.DEFINE_integer(
    'num_estimation_graphs', 1000,
    'How many graphs to analyze to determine dynamic batching budget')


JOB_SCRIPT = """#!/bin/bash -l
#SBATCH -o <folder_name>/%j.out
#SBATCH -e <folder_name>/%j.err
#SBATCH -D <folder_name>/
#SBATCH -J <job_name>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-core=1
#SBATCH --mem=<mem>  # In MB, when we set to 0, we reserve node.
#SBATCH --mail-type=none
#SBATCH --mail-user=speckhard@fhi.mpg.de
#SBATCH --time=04:00:00
<gres>
<constraint>

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# cd /u/dansp/jraph_MPEU
# Load the environment with modules and python packages.
source /u/dansp/combined_profiling/combined_venv/bin/activate
srun python3 /u/dansp/combined_profiling/jraph_MPEU/scripts/main.py --workdir=<folder_name> --config=<config_name>
"""

TEMPLATE_SCHNET_CONFIG = """
import ml_collections
from jraph_MPEU_configs.default_mp_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    config = get_config_super() # inherit from default mp config
    config.eval_every_steps = 100_000
    config.num_train_steps_max = <number_of_training_steps>
    config.log_every_steps = 100_000
    config.checkpoint_every_steps = 100_000
    config.limit_data = None
    config.selection = None
    config.data_file = <data_file>
    config.label_str = <label_str>
    config.num_edges_max = None
    config.dynamic_batch = <dynamic_batch>
    config.compute_device = <compute_device>
    config.num_estimation_graphs = <num_estimation_graphs>
    config.batch_size = <batch_size>
    config.static_round_to_multiple = <static_round_to_multiple>
    config.static_constant_batch = <static_constant_batch>
    # MPNN hyperparameters
    config.model_str = 'SchNet'
    config.message_passing_steps = 3
    config.latent_size = 64
    config.hk_init = None
    config.max_input_feature_size = 100
    config.aggregation_message_type = 'mean'
    config.aggregation_readout_type = 'mean'
    config.global_readout_mlp_layers = 0
    config.mlp_depth = 2
    config.activation_name = 'shifted_softplus'
    # Edge embedding parameters
    config.k_max = 150
    config.delta = 0.1
    config.mu_min = 0.0
    # Node embedding parameters
    config.max_atomic_number = 90
    config.use_layer_norm = False
    config.dropout_rate = 0.0

    return config
"""

TEMPLATE_MPEU_CONFIG = """
import ml_collections
from jraph_MPEU_configs.default_mp_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    config = get_config_super() # inherit from default mp config
    config.eval_every_steps = 100_000
    config.num_train_steps_max = <number_of_training_steps>
    config.log_every_steps = 100_000
    config.checkpoint_every_steps = 100_000
    config.limit_data = None
    config.selection = None
    config.data_file = <data_file>
    config.label_str = <label_str>
    config.num_edges_max = None
    config.dynamic_batch = <dynamic_batch>
    config.compute_device = <compute_device>
    config.num_estimation_graphs = <num_estimation_graphs>
    config.batch_size = <batch_size>
    config.static_round_to_multiple = <static_round_to_multiple>
    config.static_constant_batch = <static_constant_batch>
    # MPNN hyperparameters we use the defaults.

    return config
"""

def create_config_file_path(
        setting, folder_name, number_of_training_steps):
    static_constant_batch = False
    if setting['batching_method'] == 'dynamic':
        dynamic_batch = True
    elif setting['batching_method'] == 'static_constant':
        static_constant_batch = True
        dynamic_batch = False
    else:
        dynamic_batch = False
    if setting['network_type'] == 'schnet':
        config = TEMPLATE_SCHNET_CONFIG
    elif setting['network_type'] == 'MPEU':
        config = TEMPLATE_MPEU_CONFIG
    else:
        raise ValueError(f'wrong value for network type {setting["network_type"]}')
    config = config.replace(
        '<dynamic_batch>', str(dynamic_batch))
    config = config.replace(
        '<batch_size>', str(setting['batch_size']))
    config = config.replace(
        '<static_round_to_multiple>',
        str(setting['static_round_to_multiple'])
    )
    config = config.replace(
        '<static_constant_batch>',
        str(static_constant_batch)
    )
    config = config.replace(
        '<compute_device>',
        "\'" + str(setting['computing_type'].replace(':', '_') + "\'")
    )
    config = config.replace(
        '<num_estimation_graphs>',
        str(setting['num_estimation_graphs'])
    )
    config = config.replace(
        '<number_of_training_steps>', str(number_of_training_steps))
    if setting['dataset'] == 'aflow':
        data_file = "\'/u/dansp/jraph_MPEU/aflow/graphs_knn.db\'"
        label_str = "\'enthalpy_formation_atom\'"
        config = config.replace('<data_file>', data_file)
        config = config.replace('<label_str>', label_str)
    elif setting['dataset'] == 'qm9':
        data_file = "\'/u/dansp/jraph_MPEU/qm9/qm9_graphs_fc.db\'"
        label_str = "\'U0\'"
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
    job_script = job_script.replace(
        '<job_name>',
        setting['batching_method'] + '_' + str(setting['batch_size']))   
    if setting['computing_type'] in ['gpu:a100', 'gpu:v100']:
        constraint = '#SBATCH --constraint="gpu"\n'
        job_script = job_script.replace(
            '<constraint>', str(constraint))
        gres = '#SBATCH --gres=' + setting['computing_type'] + ':4'
        job_script = job_script.replace(
            '<gres>', str(gres))
        job_script = job_script.replace('<mem>', '0')
    else:
        job_script = job_script.replace(
            '<constraint>', '')
        job_script = job_script.replace(
            '<gres>', '')
        job_script = job_script.replace('<mem>', '256000')        
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
        "round_" + str(setting['static_round_to_multiple']) + '/' +
        str(setting['batch_size']) + '/' +
        setting['computing_type'].replace(':', '_') + '/' +
        'iteration_' + str(setting['iteration']))
    if not os.path.isdir(folder_name):
        # Make a directory for this setting.
        os.makedirs(folder_name)
    return folder_name



def get_settings_list(
        network_type_list, dataset_list,
        batch_size_list, batching_method_list, static_round_to_multiple_list,
        computing_type_list, num_estimation_graphs=1000):
    """Get a list of n-tuples of settings to use in profiling experiments.

    a list e.g. [("mpnn", "aflow", 32, "static", "gpu:v100"), ...]
    """

    settings_list = []

    for network_type in network_type_list:
        for dataset in dataset_list:
            for batch_size in batch_size_list:
                for batching_method in batching_method_list:
                    for static_round_to_multiple in static_round_to_multiple_list:
                        for computing_type in computing_type_list:
                            for iteration in range(10):
                                settings_dict = {
                                    'network_type': network_type,
                                    'dataset': dataset,
                                    'batch_size': batch_size,
                                    'batching_method': batching_method,
                                    'static_round_to_multiple': ast.literal_eval(static_round_to_multiple),
                                    'computing_type': computing_type,
                                    'iteration': iteration,
                                    'num_estimation_graphs': num_estimation_graphs}
                                    
                                settings_list.append(settings_dict)
    return settings_list


def create_folder_and_files_for_setting(
            settings_list, base_dir, job_list_path, number_of_training_steps):
    job_path_list = []

    for setting in settings_list:
        folder_name = create_folder_for_setting(
            base_dir, setting)
        config_file_path = create_config_file_path(
            setting, folder_name, number_of_training_steps)
        job_script_path = create_job_script(
            setting, config_file_path, folder_name)
        job_path_list.append(str(job_script_path))


    with open(job_list_path, 'w') as fd:
        fd.write('\n'.join(job_path_list))


def main(argv):
    network_type_list = FLAGS.network_type
    print(f'network types to use: {network_type_list}')
    dataset_list = FLAGS.dataset
    print(f'Dataset to use: {dataset_list}')

    static_round_to_multiple_list = FLAGS.static_round_to_multiple
    print(f'Round to multiple boolean list: {static_round_to_multiple_list}')

    batch_size_list = FLAGS.batch_size
    print(f'Batch size list to use: {batch_size_list}')

    batching_method_list = FLAGS.batching_method
    print(f'batching methods to use: {batching_method_list}')

    computing_type_list = FLAGS.computing_type
    print(f'Computing types to use: {computing_type_list}')

    number_of_training_steps = FLAGS.number_of_training_steps
    print(f'Number of training steps to run: {number_of_training_steps}')

    num_estimation_graphs = FLAGS.num_estimation_graphs
    print(f'Number of graphs to determine dynamic batching budget. {num_estimation_graphs}')

    settings_list = get_settings_list(
        network_type_list, dataset_list,
        batch_size_list, batching_method_list,
        static_round_to_multiple_list,
        computing_type_list, num_estimation_graphs)
    
    experiment_dir = FLAGS.experiment_dir
    job_list_path = Path(experiment_dir) / "profiling_jobs_list.txt"
    create_folder_and_files_for_setting(
        settings_list, experiment_dir, job_list_path, number_of_training_steps)


if __name__ == '__main__':
    app.run(main)
