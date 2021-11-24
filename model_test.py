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
    '''
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
    '''
    def test_zero_graph_apply(self):
        '''Test if the network being applied on a graph with all features 0 
        results on 0 prediction and gradient.'''
        n_node = 10
        n_edge = 4
        n_node_features = 1
        n_edge_features = 1
        graph_build = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0,0,1,2]),
            receivers=jnp.array([1,2,0,0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )
        label = np.array([1.0])
        #print(graph_build)
        # initialize network
        config.N_HIDDEN_C = 64
        config.AVG_MESSAGE = False
        config.AVG_READOUT = False
        config.HK_INIT = hk.initializers.Identity()
        lr = optax.exponential_decay(5*1e-4, 100000, 0.96)
        batch_size = 32
        model = Model(lr, batch_size, 5)
        model.build([graph_build], label)

        # graph with zeros as node features and a self edge for every node
        graph_zero = jraph.GraphsTuple(
            nodes=jnp.zeros((n_node)),
            edges=jnp.ones((n_node)),
            senders=jnp.arange(0,n_node),
            receivers=jnp.arange(0,n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        #print(graph_zero)
        prediction = model.predict([graph_zero])
        print('Prediction for zero graph: {}'.format(prediction))
        
        # from doing the propagation by hand:
        sp = shifted_softplus # shorten shifted softplus function
        h0p = sp(sp(sp(sp(1)))) + 1
        h0pp = h0p + sp(sp(sp(sp(h0p)))*h0p)
        h0ppp = h0pp + sp(sp(sp(sp(h0pp)))*h0pp)
        print('Expected label: {}'.format(n_node*sp(h0ppp)))
        self.assertAlmostEqual(n_node*sp(h0ppp), prediction[0,0])



    def test_padded_graph_apply(self):
        '''Test that the prediction of the network does not depend on the padding graphs.'''
        n_node = 10
        n_edge = 4
        n_node_features = 1
        n_edge_features = 1
        graph_build = jraph.GraphsTuple(
            nodes=jnp.ones((n_node))*5,
            edges=jnp.ones((n_edge)),
            senders=jnp.array([0,0,1,2]),
            receivers=jnp.array([1,2,0,0]),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_edge]),
            globals=None
        )
        label = np.array([1.0])
        #print(graph_build)
        # initialize network
        config.N_HIDDEN_C = 64
        config.AVG_MESSAGE = False
        config.AVG_READOUT = False
        lr = optax.exponential_decay(5*1e-4, 100000, 0.96)
        batch_size = 32
        model = Model(lr, batch_size, 5)
        model.build([graph_build], label)

        # graph with zeros as node features and a self edge for every node
        graph_zero = jraph.GraphsTuple(
            nodes=jnp.zeros((n_node)),
            edges=jnp.ones((n_node)),
            senders=jnp.arange(0,n_node),
            receivers=jnp.arange(0,n_node),
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([n_node]),
            globals=None
        )
        graph_padded = pad_graph_to_nearest_power_of_two(graph_zero)
        print(graph_padded)
        prediction = model.predict([graph_zero])
        prediction_padded = model.predict([graph_padded])
        print('Prediction for zero graph: {}'.format(prediction))
        print('Prediction for zero graph padded: {}'.format(prediction_padded))

        graph_batched = jraph.batch([graph_build, graph_zero])
        graph_batched_padded = pad_graph_to_nearest_power_of_two(graph_batched)
        prediction = model.predict([graph_batched])
        prediction_padded = model.predict([graph_batched_padded])
        print('Prediction for batch graph: {}'.format(prediction))
        print('Prediction for batch graph padded: {}'.format(prediction_padded))

        


if __name__ == '__main__':
    unittest.main()