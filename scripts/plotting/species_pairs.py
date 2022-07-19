"""This script plots the error of predictions based on atomic number pairs
contained in the compound. E.g. a compound has atoms [1,1,1,1,4,4,5], then the
contained number pairs are [[1,1],[1,4],[1,5],[4,4],[4,5],[5,5]].
TODO: decide whether to include or exclude symmetric pairs.
"""
import os
from collections import Counter

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import numpy as np

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

    errors = list(df['abs. error'])
    compounds = []
    for compound in list(df['numbers']):
        compounds.append(str_to_list(compound))

    mae_mat, count_mat = get_pair_matrices(compounds, errors)
    print(mae_mat)
    print(count_mat)
    '''
    fig, (ax1, ax2) = plt.subplots(2)
    im1 = ax1.imshow(mae_mat)
    im2 = ax2.imshow(count_mat)
    ax1.set_title('MAE')
    ax2.set_title('Count')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    plt.imshow(mae_mat)
    plt.show()
    '''
    sns.heatmap(mae_mat, norm=LogNorm())
    plt.show()

    sns.heatmap(count_mat, norm=LogNorm())
    plt.show()

    return 0


if __name__ == "__main__":
    app.run(main)
