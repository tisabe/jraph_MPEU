"""This script aims to pool the best models from a grid search / random search,
and using their predictions in ensemble to get better predictions and
uncertainty estimates."""

import os
import pickle
import json

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('directory', 'results/aflow/egap_rand_search',
    'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV/atom', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "eV/atom" or "eV"')

PREDICT_LABEL = ''
CALCULATE_LABEL = ''
ABS_ERROR_LABEL = ''


def id_list_to_int_list(ids_list):
    return [int(ids.removeprefix('id')) for ids in ids_list]


def plot_prediction(df_ensemble):
    for split in ['train', 'validation', 'test']:
        df_split = df_ensemble.loc[lambda df_temp: df_temp['split'] == split]
        stdev = np.std(df_split['target'])
        print(f'STDEV of {split} set: {stdev}')
        mean_abs_err = df_split.mean(0, numeric_only=True)['abs. error']
        print(f'MAE on {split} set: {mean_abs_err}')
        rmse = (df_split['abs. error'] ** 2).mean() ** .5
        print(f'RMSE on {split} set: {rmse}')
        r2_split = sklearn.metrics.r2_score(
            df_split['target'], df_split['ensemble_mean']
        )
        print(f'R^2 on {split} set: {r2_split}')
        median_err = df_split.median(0, numeric_only=True)['abs. error']
        print(f'Median error on {split} set: {median_err}')

    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='target',
        y='ensemble_mean',
        data=df_ensemble,
        hue='split'
    )
    ax.set_xlabel(f'Target ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'Mean prediction ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.legend(title='', fontsize=FLAGS.font_size-3)  # disable 'split' title
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory + '/ensemble_pred.png', bbox_inches='tight', dpi=600)


def plot_stdev(df_ensemble):
    for split in ['train', 'validation', 'test']:
        df_split = df_ensemble.loc[lambda df_temp: df_temp['split'] == split]
        r2_split = sklearn.metrics.r2_score(
            df_split['abs. error'], df_split['ensemble_std']
        )
        print(f'Uncertainty R^2 on {split} set: {r2_split}')


    df_test = df_ensemble.loc[lambda df_temp: df_temp['split'] == 'test']
    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='abs. error',
        y='ensemble_std',
        data=df_ensemble,
        hue='split'
    )
    ax.set_xlabel(f'Absolute error ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.legend(title='', fontsize=FLAGS.font_size-3)  # disable 'split' title
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory + '/ensemble_err.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.histplot(
        x='abs. error', y='ensemble_std', data=df_test, ax=ax,
        cbar=True, cbar_kws={'label': 'Count'}, bins=(100, 100),
        log_scale=True)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.set_xlabel(f'Absolute error ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory+'/hist_simple.png', bbox_inches='tight', dpi=600)


def main(_):
    """Main function where all the data is gathered and put into dataframes
    and evaluated."""

    df = pd.DataFrame({})
    dict_minima = {}
    df_result_list = []
    label_str = None

    for dirname in os.listdir(FLAGS.directory):
        workdir = FLAGS.directory + '/' + dirname
        try:
            metrics_path = workdir+'/checkpoints/metrics.pkl'
            # open the file with evaluation metrics
            with open(metrics_path, 'rb') as metrics_file:
                metrics_dict = pickle.load(metrics_file)

            config_path = workdir + '/config.json'
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)
                label_str = config_dict['label_str']

            split = 'validation'
            metrics = metrics_dict[split]
            # get arrays with mae and rmse for this run
            loss_rmse = [row[1][0] for row in metrics]
            loss_mae = [row[1][1] for row in metrics]
            n_mean = 1 # number of points for running mean
            #  compute running mean using convolution
            loss_rmse = np.convolve(loss_rmse, np.ones(n_mean)/n_mean, mode='valid')
            loss_mae = np.convolve(loss_mae, np.ones(n_mean)/n_mean, mode='valid')
            min_mae = min(loss_mae)
            min_rmse = min(loss_rmse)
            if min_mae > 1e4 or min_rmse > 1e4:
                print(f'mae or rmse too high for path {dirname}')
                continue

            dict_minima[dirname] = list(loss_mae)
            step = [int(row[0]) for row in metrics]
            min_step_rmse = step[np.argmin(loss_rmse)]
            min_step_mae = step[np.argmin(loss_mae)]
            row_dict = {
                'mae': min_mae,
                'rmse': min_rmse,
                'min_step_mae': min_step_mae,
                'min_step_rmse': min_step_rmse,
                'directory': dirname
            }
            df = df.append(row_dict, ignore_index=True)

        except OSError:
            pass

    # get best 10 configs and put ids into a list
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='rmse', axis='index')
    id_list_best = []
    rmse_list_best = []
    mae_list_best = []
    n_ids = 10
    for i in range(n_ids):
        #print(f'{i}. minimum rmse configuration: \n', df_copy.iloc[i])
        id_list_best.append(df_copy.iloc[i]['directory'])
        rmse_list_best.append(df_copy.iloc[i]['rmse'])
        mae_list_best.append(df_copy.iloc[i]['mae'])
    print(f'Top {n_ids} models (id, rmse, mae): ')
    print(id_list_best)
    print(rmse_list_best)
    print(mae_list_best)

    df_ensemble = pd.DataFrame({})
    df_single = pd.DataFrame({})
    # loop over the best models and get their result dataframes
    for model_id in id_list_best:
        workdir = FLAGS.directory + model_id
        df_single = pd.read_csv(workdir + '/result.csv')
        if not 'prediction' in df_single.columns:
            df_single['prediction'] = df_single['prediction_mean']
        df_result_list.append(df_single)
        #print(df_single['prediction'])
        df_ensemble[model_id] = df_single['prediction']

    df_copy = df_ensemble.copy()
    df_copy['ensemble_mean'] = df_ensemble.apply(np.mean, axis='columns')
    df_copy['ensemble_std'] = df_ensemble.apply(np.std, axis='columns')
    df_copy['target'] = df_single[label_str]
    df_copy['abs. error'] = abs(df_copy['ensemble_mean'] - df_copy['target'])
    df_copy['split'] = df_single['split']
    print(df_copy)

    plot_prediction(df_copy)
    plot_stdev(df_copy)


if __name__ == "__main__":
    app.run(main)
