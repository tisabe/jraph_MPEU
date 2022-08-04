"""This script plots the error of predictions based on atomic number pairs
contained in the compound. E.g. a compound has atoms [1,1,1,1,4,4,5], then the
contained number pairs are [[1,1],[1,4],[1,5],[4,4],[4,5],[5,5]].
TODO: decide whether to include or exclude symmetric pairs.
"""
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from jraph_MPEU.utils import load_config, get_num_pairs, str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')


def get_pair_matrices(compounds, errors):
    """Return pair matrices by aggregating unique pairs from compounds and errors.

    Args:
        compound: list of lists with atomic numbers
        errors: absolute prediction error corresponding to compounds
    """
    max_num = 0
    compounds_pairs = []
    for compound in compounds:
        max_num = max(max_num, *compound)
        pairs = get_num_pairs(compound)
        compounds_pairs.append(pairs)
    print(f'Maximum atomic number: {max_num}')

    error_mat = np.zeros((max_num+1, max_num+1))  # matrix to store errors
    count_mat = np.zeros((max_num+1, max_num+1))  # matrix to store counts

    for compound_pairs, error in zip(compounds_pairs, errors):
        for pair in compound_pairs:
            i, j = int(pair[0]), int(pair[1])
            error_mat[i, j] += error
            count_mat[i, j] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mae_mat = np.divide(error_mat, count_mat)
    return mae_mat, count_mat

def main(argv):
    """Main function that gets data and creates plots."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    workdir = FLAGS.file
    df_path = workdir + '/result.csv'
    config = load_config(workdir)

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)
        df['numbers'] = df['numbers'].apply(str_to_list)

    df['abs. error'] = abs(df['prediction'] - df[config.label_str])
    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on validation set: {mean_abs_err_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    mean_abs_err_test = df_test.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')

    mae_mat = {}
    count_mat = {}
    # training data
    errors = list(df_train['abs. error'])
    compounds = []
    for compound in list(df_train['numbers']):
        #compounds.append(str_to_list(compound))
        compounds.append(compound)
    mae_mat['train'], count_mat['train'] = get_pair_matrices(compounds, errors)

    # validation data
    errors = list(df_val['abs. error'])
    compounds = []
    for compound in list(df_val['numbers']):
        #compounds.append(str_to_list(compound))
        compounds.append(compound)
    mae_mat['validation'], count_mat['validation'] = get_pair_matrices(
        compounds, errors)

    # testing data
    errors = list(df_test['abs. error'])
    compounds = []
    for compound in list(df_test['numbers']):
        #compounds.append(str_to_list(compound))
        compounds.append(compound)
    mae_mat['test'], count_mat['test'] = get_pair_matrices(compounds, errors)


    print(np.shape(mae_mat['train']))
    sns.heatmap(mae_mat['train'], norm=LogNorm())
    plt.show()

    sns.heatmap(count_mat['train'], norm=LogNorm())
    plt.show()

    splits = ['train', 'validation', 'test']
    fig, ax = plt.subplots()
    for split in splits:
        ax.scatter(
            count_mat[split].flatten(),
            mae_mat[split].flatten(),
            label=split)

    ax.set_ylabel('MAE (PUT UNITS)', fontsize=12)
    ax.set_xlabel('Number of samples', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    app.run(main)
