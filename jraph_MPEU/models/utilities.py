"""Define some useful functions for dealing with the models in this module."""

import haiku as hk
import jax.numpy as jnp

from jraph_MPEU.models.mpeu import MPEU

def load_model(workdir, is_training):
    """Load model to evaluate on."""
    state_dir = workdir+'/checkpoints/best_state.pkl'
    with open(state_dir, 'rb') as state_file:
        best_state = pickle.load(state_file)
    config = load_config(workdir)
    # load the model params
    params = best_state['state']['params']
    print(f'Loaded best state at step {best_state["state"]["step"]}')
    # TODO: make model choosable (between MPEU, GCN, MEGnet)
    net_fn = MPEU(config, is_training)
    # compatibility layer to load old models the were initialized without state
    try:
        hk_state = best_state['state']['hk_state']
        net = hk.transform_with_state(net_fn)
    except KeyError:
        print('Loaded old stateless function. Converting to stateful.')
        hk_state = {}
        net = hk.with_empty_state(hk.transform(net_fn))
    return net, params, hk_state


# Define the shifted softplus activation function.
# Compute log(2) outside of shifted_softplus so it only needs
# to be computed once.
LOG2 = jnp.log(2)

def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Continuous and shifted version of relu activation function.

    The function is: log(exp(x) + 1) - log(2)
    This implementation is from PB Jorgensen while SchNet uses
    log(0.5*exp(x)+0.5).

    The benefit compared to a ReLU is that x < 0 you don't have a dead
    zone. And the function is C(n) smooth, infinitely differentiable
    across whole domain.
    """
    return jnp.logaddexp(x, 0) - LOG2


class MLP(hk.Module):
    """Define a custom multi-layer perceptron module, which includes dropout,
    after every weight layer and layer normalization after the last layer.
    """
    def __init__(
            self,
            name: str,
            output_sizes: Sequence[int],
            use_layer_norm=False,
            dropout_rate=0.0,
            activation=shifted_softplus,
            with_bias=False,
            activate_final=True,
            w_init=None
    ):
        super().__init__(name=name)
        self.mlp = hk.nets.MLP(
            output_sizes=output_sizes,
            w_init=w_init,
            with_bias=with_bias,
            activation=activation,
            activate_final=activate_final,
            name=name
        )
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=name + "_layer_norm")

    def __call__(self, inputs):
        outputs = self.mlp(inputs, self.dropout_rate, hk.next_rng_key())
        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)
        return outputs
