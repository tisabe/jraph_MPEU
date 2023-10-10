"""This module is used to pass the correct model function to the train or
inference script respectively. The model string in the passed config determines
which model is chosen."""

import pickle

import haiku as hk
import ml_collections

from jraph_MPEU.models.gcn import GCN
from jraph_MPEU.models.mpeu import MPEU
from jraph_MPEU.models.schnet import SchNet
from jraph_MPEU.utils import load_config


def load_model(workdir, is_training):
    """Load model to evaluate on."""
    state_dir = workdir+'/checkpoints/best_state.pkl'
    with open(state_dir, 'rb') as state_file:
        best_state = pickle.load(state_file)
    config = load_config(workdir)
    # load the model params
    params = best_state['state']['params']
    print(f'Loaded best state at step {best_state["state"]["step"]}')
    if config.model_str == 'GCN':
        net_fn = GCN(config, is_training)
    elif config.model_str == 'MPEU':
        net_fn = MPEU(config, is_training)
    elif config.model_str == 'SchNet':
        net_fn = SchNet(config, is_training)
    else:
        raise ValueError(
            f'Model string {config.model_str} not recognized')
    # compatibility layer to load old models the were initialized without state
    try:
        hk_state = best_state['state']['hk_state']
        net = hk.transform_with_state(net_fn)
    except KeyError:
        print('Loaded old stateless function. Converting to stateful.')
        hk_state = {}
        net = hk.with_empty_state(hk.transform(net_fn))
    return net, params, hk_state


def create_model(config: ml_collections.ConfigDict, is_training=True):
    """Return a function that applies the graph model."""
    if config.model_str == 'GCN':
        return GCN(config, is_training)
    elif config.model_str == 'MPEU':
        return MPEU(config, is_training)
    elif config.model_str == 'SchNet':
        return SchNet(config, is_training)
    else:
        raise ValueError(
            f'Model string {config.model_str} not recognized')
