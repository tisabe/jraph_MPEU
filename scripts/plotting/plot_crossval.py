import argparse
import os
import pickle
import json

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import haiku as hk


FLAGS = flags.FLAGS
flags.DEFINE_string('directory', 'results/aflow/crossval_grid',
    'input directory name')
flags.DEFINE_integer('max_step', 100000000,
    'maximum number of steps to take the mse/mae minimum from')
flags.DEFINE_integer('drop_n', 0,
    'Number of worst values to drop, for clearer visualization')
flags.DEFINE_integer('n_plots', 5,
    'Number of subplots in a single box plot frame.')
flags.DEFINE_integer('fontsize', 18, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "eV/atom" or "eV"')
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
    # make a dict to list ids depending on how the model training was stopped
    finish_condition = {
        "stopped_early": [], "aborted_early": [], "time_elapsed": [],
        "unknown": [], "reached_max_steps": []}
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
                finish_condition["stopped_early"].append(dirname)
            elif os.path.exists(workdir + '/ABORTED_EARLY'):
                finish_condition["aborted_early"].append(dirname)
            elif os.path.exists(workdir + '/REACHED_MAX_STEPS'):
                finish_condition["reached_max_steps"].append(dirname)
            else:
                finish_condition["time_elapsed"].append(dirname)

            #split = 'validation'
            #metrics = metrics_dict[split]
            #loss_dict = get_loss_from_metrics(metrics)

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
                'seed': config_dict['seed_weights'],
                'layer_norm': config_dict['use_layer_norm'],
                #'mae': loss_dict['min_mae'],
                #'rmse': loss_dict['min_rmse'],
                #'min_step_mae': loss_dict['min_step_mae'],
                #'min_step_rmse': loss_dict['min_step_rmse'],
                'directory': dirname
            }
            for split in ['validation', 'test']:
                metrics = metrics_dict[split]
                loss_dict = get_loss_from_metrics(metrics)
                for key, value in loss_dict.items():
                    row_dict[key+"_"+split] = value

            state_dir = workdir+'/checkpoints/best_state.pkl'
            if FLAGS.plot_num_params:
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
                finish_condition["aborted_early"].append(dirname)
            else:
                finish_condition["unknown"].append(dirname)

    for key, dir_list in finish_condition.items():
        print(f"# {key}: {len(dir_list)}")
    print(f"Aborted early: {finish_condition['aborted_early']}")
    print(f"Time elapsed: {finish_condition['time_elapsed']}")
    print(f"Unkown: {finish_condition['unknown']}")

    fig, ax = plt.subplots()
    sns.scatterplot(df, x='rmse_validation', y='rmse_test', ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.show()
    fig.savefig(
        FLAGS.directory + '/val_test_rmse.png', bbox_inches='tight', dpi=600)
    fig, ax = plt.subplots()
    sns.scatterplot(df, x='mae_validation', y='mae_test', ax=ax)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.show()
    fig.savefig(
        FLAGS.directory + '/val_test_mae.png', bbox_inches='tight', dpi=600)
    exit()

    # print list of best 10 configs
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='rmse_validation', axis='index')
    id_list_best = []
    n_ids = 50
    for i in range(n_ids):
        #print(f'{i}. minimum rmse configuration: \n', df_copy.iloc[i])
        id_list_best.append(df_copy.iloc[i]['directory'])
    id_list_best = id_list_to_int_list(id_list_best)
    print(f'Top {n_ids} models: ')
    print(id_list_best)

    # drop the worst n configs
    for i in range(FLAGS.drop_n):
        i_max = df['rmse_validation'].idxmax()
        df = df.drop([i_max])

    # plot rmse for main hyperparameters with logscale
    #box_xnames = ['latent_size', 'mp_steps', 'init_lr', 'decay_rate']
    #box_xnames = ['seed', 'dropout_rate']
    n_unique = df.nunique()
    n_dropped = n_unique.drop(n_unique[n_unique < 2].index)
    #n_dropped = n_dropped.drop(
    #    labels=['mae_validation', 'rmse_validation', 'min_step_mae', 'min_step_rmse', 'directory'])
    print(n_dropped)
    box_xnames = list(n_dropped.keys())
    col_to_label = {
        'latent_size': 'Latent size', 'mp_steps': 'MP steps',
        'init_lr': 'Learning rate', 'decay_rate': 'LR decay rate',
        'dropout_rate': 'Dropout rate', 'seed': 'Split seed',
        'batch_size': 'Batch size', 'layer_norm': 'Layer norm',
        'global_readout_mlp_layers': 'Readout layers',
        'mlp_depth': 'MLP depth', 'activation_fn': 'Activation'}
    df = df.astype({'mlp_depth': 'int32'})
    df = df.astype({'global_readout_mlp_layers': 'int32'})
    df = df.astype({'batch_size': 'int32'})
    df = df.astype({'latent_size': 'int32'})
    df = df.astype({'mp_steps': 'int32'})
    df = df.astype({'layer_norm': 'bool'})
    df = df.astype({'seed': 'int32'})
    #n_subplots_max = FLAGS.n_plots  # maximum number of subplots in a single large plot
    n_subplots_max = [[0,1,2,3],[4,5,6],[7,8,9]]
    count = 0  # count up plots for saving them in different files
    for box_xnames_split in split_list(box_xnames, n_subplots_max):
        fig, ax = plt.subplots(
            1, len(box_xnames_split), figsize=(len(box_xnames_split)*4, 8),
            sharey=True)
        for i, name in enumerate(box_xnames_split):
            sns.boxplot(ax=ax[i], x=name, y='rmse', data=df, color='lightblue')
            sns.swarmplot(ax=ax[i], x=name, y='rmse', data=df, color='.25')
            ax[i].set_xlabel(col_to_label[name], fontsize=FLAGS.fontsize)
            if i == 0:
                ax[i].set_ylabel(f'RMSE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
            else:
                ax[i].set_ylabel('')
            ax[i].tick_params(
                axis='both', which='both', labelsize=FLAGS.fontsize-4)
            ax[i].xaxis.labelpad = 15
        #plt.yscale('log')
        plt.rc('font', size=16)
        plt.tight_layout()
        plt.show()
        fig.savefig(
            FLAGS.directory+f'/grid_search_{count}.png', bbox_inches='tight',
            dpi=600)
        count += 1

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='rmse', y='mae', ax=ax)
    ax.set_xlabel(f'RMSE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax.set_ylabel(f'MAE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax.set_title('Bandgap', loc='center', y=1.0, pad=-30)
    ax.tick_params(which='both', labelsize=16)
    #plt.rc('font', size=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(
        FLAGS.directory + '/rmse_mae.png', bbox_inches='tight', dpi=600)

    if FLAGS.plot_num_params:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='num_params', y='mae', ax=ax)
        ax.set_xlabel('# of parameters', fontsize=FLAGS.fontsize)
        ax.set_ylabel(f'MAE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
        plt.rc('font', size=16)
        plt.tight_layout()
        plt.show()
        fig.savefig(
            FLAGS.directory + '/params_mae.png', bbox_inches='tight', dpi=600)

    return 0


if __name__ == "__main__":
    app.run(main)
