import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk
import functools
import optax

# import custom functions
from graph_net_fn import *
from utils import *

class Model:
    '''Make a MPEU model.'''
    def __init__(self, learning_rate, batch_size, epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.built = False # set to false until model has been built
        self.set_train_logging()

    def build(self, input, output):
        '''Initialize optimiser and parameters'''
        self.net = hk.without_apply_rng(hk.transform(net_fn)) # initializing haiku MLP layers
        params = net.init(jax.random.PRNGKey(42), graph)
        opt_init, opt_update = optax.adam(lr_schedule)
        opt_state = opt_init(params)

        self.built = True

    def fit(self, train_inputs, train_outputs):
        '''Fit the model to training data.'''
        
    def train_and_validate(self, train_inputs, train_outputs):
        '''Train and validate the model using training data and cross validation.'''

    def predict(self, inputs):
        '''Predict outputs based on inputs.'''

    def set_train_logging(self, logging=True):
        self.logging = logging



