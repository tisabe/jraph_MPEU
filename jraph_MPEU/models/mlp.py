"""Define some useful functions for dealing with the models in this module."""

from typing import Sequence
import haiku as hk
import jax.numpy as jnp


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
            use_batch_norm=False,
            dropout_rate=0.0,
            activation=shifted_softplus,
            with_bias=False,
            activate_final=True,
            w_init=None,
            batch_norm_decay=0.9,
            is_training=True
    ):
        super().__init__(name=name)
        layers = []
        for index, output_size in enumerate(output_sizes):
            layers.append(hk.Linear(
                output_size=output_size,
                w_init=w_init,
                with_bias=with_bias,
                name=f"{name}_linear_{index}"
            ))
        self.layers = tuple(layers)

        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.activate_final = activate_final
        self.is_training = is_training
        if self.use_layer_norm:
            self.layer_norm = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=name + "_layer_norm")
        if self.use_batch_norm:
            self.batch_norm = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=batch_norm_decay
            )

    def __call__(self, inputs):
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                # Only perform dropout if we are activating the output.
                if self.dropout_rate is not None:
                    out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)
                    out = self.activation(out)

        if self.use_layer_norm:
            out = self.layer_norm(out)
        if self.use_batch_norm:
            out = self.batch_norm(out, self.is_training)

        return out
