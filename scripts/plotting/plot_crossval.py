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
flags.DEFINE_integer('n_plots', 4,
    'Number of subplots in a single box plot frame.')
flags.DEFINE_integer('fontsize', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "eV/atom" or "eV"')
flags.DEFINE_boolean('plot_num_params', False, 'If number of params vs. error \
    should be plotted')

plt.set_loglevel('WARNING')


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
    df_metrics = pd.DataFrame({})
    df_configs = pd.DataFrame({})
    # make a dict to list ids depending on how the model training was stopped
    activation_name_convert = {
        'shifted_softplus': 'SSP', 'relu': 'relu', 'swish': 'swish'}
    finish_condition = {
        "stopped_early": [], "aborted_early": [], "time_elapsed": [],
        "unknown": [], "reached_max_steps": []}
    if FLAGS.plot_num_params:
        import haiku as hk
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

            loss_dict = {}
            for split in ['validation', 'test']:
                metrics = metrics_dict[split]
                loss_dict_split = get_loss_from_metrics(metrics)
                for key, value in loss_dict_split.items():
                    loss_dict[key+"_"+split] = value
            loss_dict['directory'] = dirname

            state_dir = workdir+'/checkpoints/best_state.pkl'
            if FLAGS.plot_num_params:
                with open(state_dir, 'rb') as state_file:
                    best_state = pickle.load(state_file)
                params = best_state['state']['params']
                num_params = hk.data_structures.tree_size(params)
                config_dict['num_params'] = num_params

            df_metrics = pd.concat(
                [df_metrics, pd.DataFrame([loss_dict])], ignore_index=True)
            df_configs = pd.concat(
                [df_configs, pd.DataFrame([config_dict])], ignore_index=True)

        except OSError:
            if os.path.exists(workdir + '/ABORTED_EARLY'):
                # in this case, the training was aborted before the first
                # checkpoint
                finish_condition["aborted_early"].append(dirname)
            else:
                finish_condition["unknown"].append(dirname)

    # get the columns with more than one unique value
    cols_configs = list(df_configs.columns.values)
    cols_variable = []
    for col in cols_configs:
        val_counts = df_configs[col].value_counts()
        if len(val_counts) > 1:
            cols_variable.append(col)
            print(val_counts)
    # hard-coded order of columns to reproduce figs in manuscript
    if FLAGS.directory == 'results/aflow/egap/mpeu/rand_search':
        cols_variable = ['batch_size', 'message_passing_steps', 'latent_size',
            'init_lr', 'decay_rate', 'dropout_rate', 'global_readout_mlp_layers',
            'mlp_depth', 'activation_name', 'use_layer_norm']
    elif FLAGS.directory == 'results/aflow/ef/mpeu/rand_search':
        cols_variable = ['batch_size', 'message_passing_steps', 'latent_size',
            'init_lr', 'decay_rate', 'dropout_rate', 'global_readout_mlp_layers',
            'mlp_depth', 'activation_name', 'use_layer_norm']
    print("Cols to plot: ", cols_variable)
    for key, dir_list in finish_condition.items():
        print(f"# {key}: {len(dir_list)}")
    print(f"Aborted early: {finish_condition['aborted_early']}")
    print(f"Time elapsed: {finish_condition['time_elapsed']}")
    print(f"Reached max steps: {finish_condition['reached_max_steps']}")
    print(f"Unkown: {finish_condition['unknown']}")
    df_path = FLAGS.directory + '/result_crossval.csv'
    df = pd.concat([df_metrics, df_configs], axis=1)
    df.to_csv(df_path, index=False)

    # sort by validation rmse and add a label for best 10 models
    df = df.sort_values(by='rmse_validation', axis='index')
    rmse_cut = df['rmse_validation'].iloc[10]
    print('Best RMSE: ', df['rmse_validation'].iloc[0])
    print('Best parameters: ', df[cols_variable].iloc[0])
    print('Tenth best RMSE: ', rmse_cut)
    df['in_ensemble'] = df['rmse_validation'] < rmse_cut
    # sort descencing, to put better points in front
    df = df.sort_values(by='rmse_validation', axis='index', ascending=False)
    df_best = df[df['rmse_validation'] < rmse_cut]
    df_other = df[df['rmse_validation'] >= rmse_cut]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.8, 4.8))
    ax1.scatter(
        x=df_other['rmse_validation'], y=df_other['rmse_test'],
        alpha=0.3, label='NAS models')
    ax1.scatter(
        x=df_best['rmse_validation'], y=df_best['rmse_test'],
        label='NAS models, top 10')
    ax1.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax1.set_ylabel(f'Test RMSE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax1.set_xlabel(f'Validation RMSE ({FLAGS.unit})', fontsize=FLAGS.fontsize)

    ax2.scatter(
        x=df_other['mae_validation'], y=df_other['mae_test'],
        alpha=0.3, label='NAS models')
    ax2.scatter(
        x=df_best['mae_validation'], y=df_best['mae_test'],
        label='NAS models, top 10')

    # custom settings for specific publication plots
    match FLAGS.directory:
        case 'results/aflow/egap/mpeu/rand_search':
            ax1.set_ylim([.3, .7])
            ax1.set_xlim([.3, .7])
            ax1.set_xticks([.3, .4, .5, .6, .7])
            ax1.set_yticks([.3, .4, .5, .6, .7])
            ax1.scatter(x=0.434, y=0.379, s=200, marker='*', label='Ensemble')
            ax1.scatter(x=0.506, y=0.399, label='Ref. MPEU')

            ax2.set_ylim([.15, .35])
            ax2.set_xlim([.15, .35])
            ax2.set_xticks([.15, .2, .25, .3, .35])
            ax2.set_yticks([.15, .2, .25, .3, .35])
            ax2.scatter(x=0.183, y=0.168, s=200, marker='*', label='Ensemble')
            ax2.scatter(x=0.209, y=0.180, label='Ref. MPEU')

        case 'results/aflow/egap/painn/rand_search':
            ax1.set_ylim([.3, .7])
            ax1.set_xlim([.3, .7])
            ax1.set_xticks([.3, .4, .5, .6, .7])
            ax1.set_yticks([.3, .4, .5, .6, .7])
            ax1.scatter(x=0.442, y=0.368, s=200, marker='*', label='Ensemble PaiNN')
            ax1.scatter(x=0.474, y=0.381, label='Ref. PaiNN')

            ax2.set_ylim([.15, .35])
            ax2.set_xlim([.15, .35])
            ax2.set_xticks([.15, .2, .25, .3, .35])
            ax2.set_yticks([.15, .2, .25, .3, .35])
            ax2.scatter(x=0.180, y=0.158, s=200, marker='*', label='Ensemble PaiNN')
            ax2.scatter(x=0.195, y=0.171, label='Ref. PaiNN')

        case _:
            pass

    ax2.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax2.set_ylabel(f'Test MAE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax2.set_xlabel(f'Validation MAE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    x_ref = np.linspace(*ax1.get_xlim())
    ax1.plot(x_ref, x_ref, '--', alpha=0.2, color='black', linewidth=4)
    x_ref = np.linspace(*ax2.get_xlim())
    ax2.plot(x_ref, x_ref, '--', alpha=0.2, color='black', linewidth=4)
    ax1.legend(fontsize=FLAGS.fontsize-5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    plt.show()
    fig.savefig(
        FLAGS.directory + '/val_test_both.png', bbox_inches='tight', dpi=600)

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
    box_xnames = cols_variable
    col_to_label_type = {
        'latent_size': {'label': 'Latent size', 'dtype': 'int32'},
        'batch_size': {'label': 'Batch size', 'dtype': 'int32'},
        'mp_steps': {'label': 'MP steps', 'dtype': 'int32'},
        'message_passing_steps': {'label': 'MP steps', 'dtype': 'int32'},
        'init_lr': {'label': 'Learning rate', 'dtype': 'float32'},
        'decay_rate': {'label': 'LR decay rate', 'dtype': 'float32'},
        'dropout_rate': {'label': 'Dropout rate', 'dtype': 'float32'},
        'layer_norm': {'label': 'Layer norm', 'dtype': 'bool'},
        'use_layer_norm': {'label': 'Layer norm', 'dtype': 'bool'},
        'global_readout_mlp_layers': {'label': 'Readout layers', 'dtype': 'int32'},
        'mlp_depth': {'label': 'MLP depth', 'dtype': 'int32'},
        'activation_name': {'label': 'Activation', 'dtype': '|S'},
        'aggregation_message_type': {'label': 'Message aggregation', 'dtype': '|S'},
    }
    for col in cols_variable:
        df[col].astype(col_to_label_type[col]['dtype'])
    if 'activation_name' in cols_variable:
        df['activation_name'] = df['activation_name'].map(activation_name_convert)
    n_subplots_max = FLAGS.n_plots  # maximum number of subplots in a single large plot
    #n_subplots_max = [[0,1,2,3],[4,5,6],[7,8,9]]
    count = 0  # count up plots for saving them in different files
    for box_xnames_split in split_list(box_xnames, n_subplots_max):
        fig, ax = plt.subplots(
            1, len(box_xnames_split), figsize=(len(box_xnames_split)*4, 8),
            sharey=True)
        for i, name in enumerate(box_xnames_split):
            print(name)
            sns.boxplot(ax=ax[i], x=name, y='rmse_validation', data=df, color='lightblue')
            sns.swarmplot(ax=ax[i], x=name, y='rmse_validation', data=df, color='.25')
            ax[i].set_xlabel(col_to_label_type[name]['label'], fontsize=FLAGS.fontsize)
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
    sns.scatterplot(data=df, x='rmse_validation', y='mae_validation', ax=ax)
    ax.set_xlabel(f'RMSE ({FLAGS.unit})', fontsize=FLAGS.fontsize)
    ax.set_ylabel(f'MAE ({FLAGS.unit})', fontsize=FLAGS.fontsize)

    match FLAGS.directory:
        case 'results/aflow/egap_rand_search':
            ax.set_title('Bandgap', loc='center', y=1.0, pad=-30)
        case _:
            pass
    ax.tick_params(which='both', labelsize=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(
        FLAGS.directory + '/rmse_mae.png', bbox_inches='tight', dpi=600)

    if FLAGS.plot_num_params:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='num_params', y='mae_validation', ax=ax)
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
