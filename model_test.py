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
import matplotlib.pyplot as plt

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

    def test_overfit_model(self):
        config.N_HIDDEN_C = 64
        print('N_HIDDEN_C: {}'.format(config.N_HIDDEN_C))
        config.AVG_MESSAGE = False
        config.AVG_READOUT = False
        lr = optax.exponential_decay(1e-4, 100, 0.9)
        batch_size = 1
        print('batch size: {}'.format(batch_size))
        model_test = Model(lr, batch_size, 5)
        file_str = 'QM9/graphs_U0K.csv'
        print('Loading data file')
        inputs, outputs, auids = get_data_df_csv(file_str)
        n = 4
        epochs = 2000
        inputs, outputs = inputs[:n], outputs[:n]
        outputs, mean_test, std_test = normalize_targets(inputs, outputs)
        print('Building model')
        model_test.build(inputs, outputs)
        model_test.train_and_test(inputs, outputs, epochs, 1, 0.5)

        train_loss = np.array(model_test.train_loss_arr)
        test_loss = np.array(model_test.test_loss_arr)

        print(np.shape(train_loss))
        print(np.shape(test_loss))

        fig, ax = plt.subplots()
        ax.plot(test_loss[:,0], test_loss[:,1], label='test data')
        ax.plot(train_loss[:,0], train_loss[:,1], label='train data')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss (MAE)')
        plt.yscale('log')
        plt.show()


if __name__ == '__main__':
    unittest.main()