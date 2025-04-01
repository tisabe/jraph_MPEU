"""Loss functions for training MPNNs."""

from typing import Union, Iterable, Mapping, Any

import jax
import jax.numpy as jnp
import jraph


ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


def _safe_divide(x, y):
    return jnp.where(y == 0.0, 0.0, x / jnp.where(y == 0.0, 1.0, y))


def _safe_mask(graph):
    """Return the valid graph/node/edge mask, for padded or unpadded graph."""
    mask_dict = {}
    padding_functions = {
        'globals': jraph.get_graph_padding_mask,
        'nodes': jraph.get_node_padding_mask,
        'edges': jraph.get_edge_padding_mask
    }
    for key, fun in padding_functions.items():
        mask = fun(graph) # for unpadded graph, mask is all False
        mask_dict[key] = jnp.logical_not(jnp.logical_xor(mask, jnp.any(mask)))
    return mask_dict


def total_squared_error_pytree(targets: ArrayTree, predictions: ArrayTree) -> float:
    """Returns sum, for mean squared error, divide by number of globals/nodes/edges."""
    diff_sq = jax.tree.map(
        lambda x, y: jnp.sum((x - y)**2).astype(float), targets, predictions)
    return sum(jax.tree.flatten(diff_sq)[0]) # TODO: check if this should be jnp.sum


class WeightedGlobalsNodesEdgesLoss:
    # inspired by WeightedEnergyForcesStressLoss in mace-jax repository
    def __init__(self, globals_weight=1.0, nodes_weight=1.0, edges_weight=1.0) -> None:
        super().__init__()
        self.globals_weight = globals_weight
        self.nodes_weight = nodes_weight
        self.edges_weight = edges_weight

    def __call__(
        self,
        targets: jraph.GraphsTuple,
        predictions: jraph.GraphsTuple
    ) -> jnp.ndarray:
        loss = 0
        mask = _safe_mask(predictions)

        if (self.globals_weight > 0.0
            and targets.globals is not None
            and predictions.globals is not None):
            n_globals = jnp.sum(mask['globals'])
            loss += self.globals_weight * total_squared_error_pytree(
                targets.globals, predictions.globals) / n_globals

        if (self.nodes_weight > 0.0
            and targets.nodes is not None
            and predictions.nodes is not None):
            n_nodes = jnp.sum(mask['nodes'])
            loss += self.nodes_weight * total_squared_error_pytree(
                targets.nodes, predictions.nodes) / n_nodes

        if (self.edges_weight > 0.0
            and targets.edges is not None
            and predictions.edges is not None):
            n_edges = jnp.sum(mask['edges'])
            loss += self.edges_weight * total_squared_error_pytree(
                targets.edges, predictions.edges) / n_edges

        return loss  # [n_graphs, ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(globals_weight={self.globals_weight:.3f}, "
            f"nodes_weight={self.nodes_weight:.3f}, "
            f"edges_weight={self.edges_weight:.3f})"
        )


def graph_loss_mse(
    graph_pred: jraph.GraphsTuple,
    graph_target: jraph.GraphsTuple,
    weights=None, # TODO: figure out type and shape
    mask=None,
) -> float:
    """Compute the MSE (mean squared error) from the whole graph."""
    nodes_diff_sq = jax.tree.map(
        lambda x, y: (x - y)**2, graph_pred.nodes, graph_target.nodes)
    edges_diff_sq = jax.tree.map(
        lambda x, y: (x - y)**2, graph_pred.edges, graph_target.edges)
    globals_diff_sq = jax.tree.map(
        lambda x, y: (x - y)**2, graph_pred.globals, graph_target.globals)
    nodes_flat, _ = jax.tree.flatten(nodes_diff_sq)
    nodes_mse = jnp.mean(nodes_flat)
    edges_flat, _ = jax.tree.flatten(edges_diff_sq)
    edges_mse = jnp.mean(edges_flat)
    globals_flat, _ = jax.tree.flatten(globals_diff_sq)
    globals_mse = jnp.mean(globals_flat)
    if weights is None:
        return nodes_mse + edges_mse + globals_mse
