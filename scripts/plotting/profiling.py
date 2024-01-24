"""Script to plot profiling data"""

from absl import flags
from absl import app

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

BASE_DIR = '/home/dts/Documents/hu'
PROFILING_CSV = 'parsed_profiling_experiments_10k_steps_csv_12_41__18_1_2023.csv'

profiling_df = pd.read_csv(os.path.join(BASE_DIR, PROFILING_CSV))


def main(argv)

if __name__ == '__main__':
    app.run(main)