import os
import pickle
import json

from absl import app
from absl import flags
import numpy as np
import pandas as pd


FLAGS = flags.FLAGS
flags.DEFINE_string('directory', 'results/aflow/crossval_grid',
    'input directory name')
flags.DEFINE_integer('max_step', 100000000,
    'maximum number of steps to take the mse/mae minimum from')
flags.DEFINE_boolean('plot_num_params', False, 'If number of params vs. error \
    should be plotted')


def split_list(list_a, chunk_size):
    """Split list_a into even chunks of chunk_size elements"""
    if isinstance(chunk_size, int):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]
    else:
        for i in chunk_size:
            yield [list_a[j] for j in i]


def id_list_to_int_list(ids_list):
    """Convert id list to int"""
    return [int(ids.removeprefix('id')) for ids in ids_list]


def get_loss_from_metrics(metrics):
    """From the saved loss curve in metrics, get minimum loss values."""
    loss_rmse = [row[1][0] for row in metrics if int(row[0]) < FLAGS.max_step]
    loss_mae = [row[1][1] for row in metrics if int(row[0]) < FLAGS.max_step]

    step = [int(row[0]) for row in metrics if int(row[0]) < FLAGS.max_step]
    min_step_rmse = step[np.argmin(loss_rmse)]
    min_step_mae = step[np.argmin(loss_mae)]
    min_rmse = min(loss_rmse)
    min_mae = min(loss_mae)

    result = {}

    if min_rmse > 1e4 or min_mae > 1e4:
        result['rmse'] = None
        result['mae'] = None
    else:
        result['rmse'] = min_rmse
        result['mae'] = min_mae

    result['min_step_rmse'] = min_step_rmse
    result['min_step_mae'] = min_step_mae

    return result


def append_key(dict_in, key_append):
    """Append a string key_append to the end of the keys of dict_in"""
    dict_out = {}
    for key, value in dict_in.items():
        dict_out[key+'_'+key_append] = value
    return dict_out


def main(_):
    """Main body where files are opened and plots plotted."""
    # plot learning curves
    df = pd.DataFrame({})
    for dirname in os.listdir(FLAGS.directory):
        workdir = FLAGS.directory + '/' + dirname
        try:
            metrics_path = FLAGS.directory+'/'+dirname+'/checkpoints/metrics.pkl'
            # open the file with evaluation metrics
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            config_path = FLAGS.directory + '/' + dirname + '/config.json'
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config_dict = json.load(config_file)

            if os.path.exists(workdir + '/STOPPED_EARLY'):
                finish_condition = 'stopped_early'
            elif os.path.exists(workdir + '/ABORTED_EARLY'):
                finish_condition = 'aborted_early'
            elif os.path.exists(workdir + '/REACHED_MAX_STEPS'):
                finish_condition = 'reached_max_steps'
            else:
                finish_condition = 'time_limit_reached'

            activation_name_convert = {
                'shifted_softplus': 'SSP', 'relu': 'relu', 'swish': 'swish'}
            row_dict = {
                'batch_size': int(config_dict['batch_size']),
                'mp_steps': int(config_dict['message_passing_steps']),
                'latent_size': int(config_dict['latent_size']),
                'init_lr': config_dict['init_lr'],
                'decay_rate': config_dict['decay_rate'],
                'dropout_rate': config_dict['dropout_rate'],
                'global_readout_mlp_layers': int(config_dict['global_readout_mlp_layers']),
                'mlp_depth': int(config_dict['mlp_depth']),
                'activation_fn': activation_name_convert[
                    config_dict['activation_name']],
                #'seed': config_dict['seed_weights'],
                'layer_norm': config_dict['use_layer_norm'],
                #'mae': loss_dict['min_mae'],
                #'rmse': loss_dict['min_rmse'],
                #'min_step_mae': loss_dict['min_step_mae'],
                #'min_step_rmse': loss_dict['min_step_rmse'],
                'directory': dirname
            }
            for split in ['train', 'validation', 'test']:
                metrics = metrics_dict[split]
                loss_rmse = [row[1][0] for row in metrics if int(row[0]) < FLAGS.max_step]
                loss_mae = [row[1][1] for row in metrics if int(row[0]) < FLAGS.max_step]
                step = [int(row[0]) for row in metrics]
                row_dict[split+'_'+'rmse_curve'] = loss_rmse
                row_dict[split+'_'+'mae_curve'] = loss_mae
                row_dict['curve_step_number'] = step

            state_dir = workdir+'/checkpoints/best_state.pkl'
            if FLAGS.plot_num_params:
                import haiku as hk
                with open(state_dir, 'rb') as state_file:
                    best_state = pickle.load(state_file)
                params = best_state['state']['params']
                num_params = hk.data_structures.tree_size(params)
                row_dict['num_params'] = num_params

            df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

        except OSError:
            if os.path.exists(workdir + '/ABORTED_EARLY'):
                # in this case, the training was aborted before the first
                # checkpoint
                finish_condition = 'aborted_early'
            else:
                finish_condition = 'unknown'

    col_to_label = {
        'latent_size': 'Latent size', 'mp_steps': 'MP steps',
        'init_lr': 'Learning rate', 'decay_rate': 'LR decay rate',
        'dropout_rate': 'Dropout rate',
        'batch_size': 'Batch size', 'layer_norm': 'Layer norm',
        'global_readout_mlp_layers': 'Readout layers',
        'mlp_depth': 'MLP depth', 'activation_fn': 'Activation'}
    df = df.astype({'mlp_depth': 'int32'})
    df = df.astype({'global_readout_mlp_layers': 'int32'})
    df = df.astype({'batch_size': 'int32'})
    df = df.astype({'latent_size': 'int32'})
    df = df.astype({'mp_steps': 'int32'})
    df = df.astype({'layer_norm': 'bool'})

    print(df.describe())
    print(df.info(verbose=True))
    df.to_csv(FLAGS.directory+'/df_curves.csv', index=False)

    return 0


if __name__ == "__main__":
    app.run(main)
