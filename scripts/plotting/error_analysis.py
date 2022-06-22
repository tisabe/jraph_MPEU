"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
import argparse
import os

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from jraph_MPEU.input_pipeline import get_datasets
from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import get_results_df


def main(args):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder
    df_path = workdir + '/result.csv'
    
    if not os.path.exists(df_path):
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)
    sns.scatterplot(x='enthalpy_formation_atom', y='prediction', data=df, hue='split')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction errors for different distributions.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='tests/qm9_test_run',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
