import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk
import functools
import optax
import pandas

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

    def get_data_df_csv(self, file_str):
        '''Import data from a pandas.DataFrame saved as csv. 
        Return as inputs, outputs (e.g. train_inputs, train_outputs)'''
        df = pandas.read_csv(file_str)
        inputs = []
        outputs = []
        for index, row in df.iterrows():
            nodes = row['nodes']
            senders = row['senders']
            receivers = row['receivers']
            graph = jraph.GraphsTuple(
                n_node=np.asarray([len(nodes)]),
                n_edge=np.asarray([len(senders)]),
                nodes=nodes, edges=None,
                globals=None,
                senders=np.asarray(senders), receivers=np.asarray(receivers))
            inputs.append(graph)
            outputs.append(row['label'])
        return inputs, outputs

    def compute_loss(self, params, graph, label, net):
        """Computes loss."""
        pred_graph = net.apply(params, graph)
        preds = pred_graph.globals

        # one graph was added to pad nodes and edges, so globals will also be padded by one
        # masking is not needed so long as the padded graph also has a zero global array after update
        label_padded = jnp.pad(label, ((0, 1), (0, 0)))

        loss = jnp.sum(jnp.abs(preds - label_padded))
        return loss

    def build(self, input, output):
        '''Initialize optimiser and model parameters'''
        self.net = hk.without_apply_rng(hk.transform(net_fn)) # initializing haiku MLP layers
        params = self.net.init(jax.random.PRNGKey(42), graph)
        opt_init, self.opt_update = optax.adam(self.learning_rate)
        self.opt_state = opt_init(params)

        self.compute_loss_fn = functools.partial(self.compute_loss, net=self.net)
        self.compute_loss_fn = jax.jit(jax.value_and_grad(
                                    self.compute_loss_fn))

        self.built = True

    @jax.jit
    def update(self,
            params: hk.Params,
            opt_state: optax.OptState,
            graph: jraph.GraphsTuple,
            label: jnp.ndarray,
        ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        loss, grad = self.compute_loss_fn(params, graph, label)
        updates, self.opt_state = self.opt_update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    def fit(self, train_inputs, train_outputs):
        '''Fit the model to training data.'''
        
    def train_and_validate(self, train_inputs, train_outputs):
        '''Train and validate the model using training data and cross validation.'''

    def predict(self, inputs):
        '''Predict outputs based on inputs.'''

    def set_train_logging(self, logging=True):
        self.logging = logging



