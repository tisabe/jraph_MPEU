"""This script aims to pool the best models from a grid search / random search,
and using their predictions in ensemble to get better predictions and
uncertainty estimates."""

import os
import pickle
import json

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.feature_selection


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


def plot_prediction(df_ensemble, label_str):
    df_ensemble['abs. error'] = abs(
        df_ensemble['prediction'] - df_ensemble[label_str])
    for split in ['train', 'validation', 'test']:
        df_split = df_ensemble.loc[lambda df_temp: df_temp['split'] == split]
        stdev = np.std(df_split[label_str])
        print(f'STDEV of {split} set: {stdev}')
        mean_abs_err = df_split.mean(0, numeric_only=True)['abs. error']
        print(f'MAE on {split} set: {mean_abs_err}')
        rmse = (df_split['abs. error'] ** 2).mean() ** .5
        print(f'RMSE on {split} set: {rmse}')
        r2_split = sklearn.metrics.r2_score(
            df_split[label_str], df_split['prediction']
        )
        print(f'R^2 on {split} set: {r2_split}')
        median_err = df_split.median(0, numeric_only=True)['abs. error']
        print(f'Median error on {split} set: {median_err}')

    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x=label_str,
        y='prediction',
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


def plot_stdev(df_ensemble, label_str):
    df_ensemble['abs. error'] = abs(
        df_ensemble['prediction'] - df_ensemble[label_str])
    for split in ['train', 'validation', 'test']:
        df_split = df_ensemble.loc[lambda df_temp: df_temp['split'] == split]
        r2_split = sklearn.metrics.r2_score(
            df_split['abs. error'], df_split['prediction_std']
        )
        print(f'Uncertainty R^2 on {split} set: {r2_split}')
        pearson = df_split[['abs. error', 'prediction_std']].corr()
        print(f'Pearson r^2 on {split} set: {pearson}')

    df_test = df_ensemble.loc[lambda df_temp: df_temp['split'] == 'test']
    # calculater cumulative distributions
    '''
    cum_dist_true = np.cumsum(np.histogram(
        df_test['abs. error'], bins=100, density=True)[0])

    cum_dist_obs = np.cumsum(np.histogram(
        df_test['ensemble_std'], bins=100, density=True)[0])
    plt.plot(cum_dist_true, cum_dist_obs)
    plt.show()
    '''

    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='abs. error',
        y='prediction_std',
        data=df_ensemble,
        hue='split'
    )
    ax.set_xlabel(f'Absolute error ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'Prediction STDEV ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.legend(title='', fontsize=FLAGS.font_size-3)  # disable 'split' title
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory + '/ensemble_err.png', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.histplot(
        x='abs. error', y='prediction_std', data=df_test, ax=ax,
        cbar=True, cbar_kws={'label': 'Count'}, bins=(100, 100),
        log_scale=False, binrange=((0,0.2),(0,0.2)))
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
    df_path = FLAGS.directory+'/result.csv'
    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')

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

        df_ensemble = pd.DataFrame({}) # used to calculate ensemble predictions
        df_single = pd.DataFrame({}) # save single result df in here
        df_result = pd.DataFrame({}) # save the complete results with predictions
        # loop over the best models and get their result dataframes
        for i, model_id in enumerate(id_list_best):
            workdir = FLAGS.directory + model_id
            df_single = pd.read_csv(workdir + '/result.csv')
            if i == 0:
                # save information about all structures from the first result.csv
                # in df_ensemble
                df_result = df_single.copy()
                # drop columns that might be confused with the ensemble prediction
                df_result = df_result.drop(errors='ignore',
                    columns=['prediction', 'prediction_mean', 'prediction_std'])
            else:
                # check that rows are in the same order
                if not np.all(df_result['auid'] == df_single['auid']):
                    print(f'auids in {model_id} not same order as in \
                        {id_list_best[0]}')
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

        # enter ensemble results into df_result
        df_result['prediction'] = df_copy['ensemble_mean']
        df_result['prediction_std'] = df_copy['ensemble_std']
        df_result.to_csv(df_path, index=False)

         # write config with only label string for error_analysis.py
        with open(os.path.join(FLAGS.directory, 'config.json'), 'w') as config_file:
            json.dump({'label_str': label_str},
                config_file, indent=4, separators=(',', ': '))
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df_result = pd.read_csv(df_path)
        with open(os.path.join(FLAGS.directory, 'config.json'), 'r') as config_file:
            config_dict = json.load(config_file)
            label_str = config_dict['label_str']

    #plot_prediction(df_result, label_str)
    plot_stdev(df_result, label_str)


if __name__ == "__main__":
    app.run(main)
