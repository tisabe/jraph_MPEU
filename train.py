import jax
import jax.numpy as jnp
import spektral
import jraph
import numpy as np
import haiku as hk
import functools
import optax

import logging
from tqdm import trange
import time
from typing import Generator, Mapping, Tuple

from spektral.datasets import QM9

# import custom functions
from graph_net_fn import *
from utils import *

# Download the dataset.
dataset = QM9(amount=8 * 1024)  # Set amount=None to train on whole dataset

graph_j, label = spektral_to_jraph(dataset[2])
# print(graph_j)
# print(label)
label_size = len(label)


# print(label_size)




def compute_loss(params, graph, label, net):
    """Computes loss."""
    pred_graph = net.apply(params, graph)
    preds = pred_graph.globals

    # one graph was added to pad nodes and edges, so globals will also be padded by one
    # masking is not needed so long as the padded graph also has a zero global array after update
    label_padded = jnp.pad(label, ((0, 1), (0, 0)))

    loss = jnp.sum(jnp.abs(preds - label_padded))
    return loss


@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        graph: jraph.GraphsTuple,
        label: jnp.ndarray,
) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss, grad = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


def evaluate(dataset_in, net, params, batch_size):
    reader = DataReader(dataset_in, batch_size)
    num_graphs = reader.total_num_graphs

    accumulated_loss = 0
    accumulated_label_MAE = jnp.zeros(label_size)

    for _ in np.arange(num_graphs // batch_size):
        graph, label = next(reader)

        graph = pad_graph_to_nearest_power_of_two(graph)

        pred_graph = net.apply(params, graph)
        preds = pred_graph.globals

        loss, grad = compute_loss_fn(params, graph, label)

        label_padded = jnp.pad(label, ((0, 1), (0, 0)))
        accumulated_loss += loss
        accumulated_label_MAE += jnp.sum(jnp.abs(preds - label_padded), axis=0)

    average_loss = accumulated_loss / num_graphs
    average_MAE = accumulated_label_MAE / num_graphs

    return average_loss, average_MAE


epochs = 10
batch_size = 32

reader = DataReader(dataset, batch_size)

reader.repeat()
net = hk.without_apply_rng(hk.transform(net_fn)) # initializing haiku MLP layers
graph, _ = reader.get_graph_by_idx(0)
# Initialize the network.
logging.info('Initializing network.')
params = net.init(jax.random.PRNGKey(42), graph)
# Initialize the optimizer.
lr_schedule = optax.exponential_decay(1e-3, 1000, 0.9)
#opt_init, opt_update = optax.adam(1e-4)
opt_init, opt_update = optax.adam(lr_schedule)
opt_state = opt_init(params)

compute_loss_fn = functools.partial(compute_loss, net=net)
compute_loss_fn = jax.jit(jax.value_and_grad(
    compute_loss_fn))

# res_pre = evaluate(dataset, net, params, batch_size)
# print(res_pre)

num_training_steps_per_epoch = reader.total_num_graphs // batch_size

print("starting training \n")

for idx_epoch in range(epochs):
    reader.shuffle()
    loss_sum = 0
    for i in trange(num_training_steps_per_epoch, desc=("epoch " + str(idx_epoch)), unit="gradient steps"):
        graph, label = next(reader)
        graph = pad_graph_to_nearest_power_of_two(graph)
        params, opt_state, loss = update(params, opt_state, graph, label)
        loss_sum += loss
    print(loss_sum / (num_training_steps_per_epoch * batch_size))  # print the average loss per graph

res_post = evaluate(dataset, net, params, batch_size)
print(res_post)
print(params)
