"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
import argparse

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
    df = get_results_df(args.folder)
    df.describe()
    


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
