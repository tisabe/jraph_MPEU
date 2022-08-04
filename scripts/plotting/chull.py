"""Script to plot the proper and predicted convex hull using formation energies"""

import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.phasediagram import PhaseDiagram

from jraph_MPEU.utils import load_config, get_num_pairs, str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')


def get_refs(df: pd.DataFrame, label_str: str):
    """Return list of reference energies and formulae."""
    refs = []  # list of formulas and energies from aflow
    for _, row in df.iterrows():
        n_atoms = len(row['numbers'])
        #energy = row[config.label_str]*n_atoms
        energy = row[label_str]*n_atoms
        formula = row['formula']
        if isinstance(formula, str) and isinstance(energy, float):
            refs.append((formula, energy))
    return refs


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

    refs_target = get_refs(df, 'prediction')

    phases = PhaseDiagram(refs_target, filter='AlSi')

    phases.plot(show=True)


if __name__ == "__main__":
    app.run(main)
