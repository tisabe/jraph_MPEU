import jax
import jax.numpy as jnp
from numpy.lib import type_check
import jraph
import numpy as np
import haiku as hk
import functools
import optax
import pandas
from tqdm import trange

# import custom functions
from graph_net_fn import *
from utils import *
import config
from model import *

import unittest

class TestHelperFunctions(unittest.TestCase):
    def test_import_data(self):
        file_str = 'aflow/graphs_test_cutoff3A.csv'
        #inputs, outputs = get_data_df_csv(file_str)
        array_string = '[0 1 2 3 4]'
        array_string = array_string.replace(' ', ',')
        array = str_to_array(array_string)
        print(array)

if __name__ == '__main__':
    unittest.main()