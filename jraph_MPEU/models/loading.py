"""This module is used to pass the correct model function to the train or
inference script respectively. The model string in the passed config determines
which model is chosen."""

import pickle
import os

from absl import logging
import haiku as hk
import ml_collections

from jraph_MPEU.models.gcn import GCN
from jraph_MPEU.models.mpeu import MPEU
from jraph_MPEU.models.mpeu_uq import MPEU_uq
from jraph_MPEU.models.schnet import SchNet
from jraph_MPEU.models.painn import get_painn
from jraph_MPEU.utils import load_config


def load_model(workdir, is_training):
    """Load model to evaluate on."""
    state_dir = workdir+'/checkpoints/best_state.pkl'
    with open(state_dir, 'rb') as state_file:
        best_state = pickle.load(state_file)
    config = load_config(workdir)
    # load the model params
    params = best_state['state']['params']
    logging.info(f'Loaded best state at step {best_state["state"]["step"]}')
    match config.model_str:
        case 'GCN':
            net_fn = GCN(config, is_training)
        case 'MPEU':
            net_fn = MPEU(config, is_training)
        case 'SchNet':
            net_fn = SchNet(config, is_training)
        case 'PaiNN':
            net_fn = get_painn(config)
        case 'MPEU_uq':
            net_fn = MPEU_uq(config, is_training)
        case _:
            raise ValueError(
                f'Model string {config.model_str} not recognized')
    # compatibility layer to load old models the were initialized without state
    try:
        hk_state = best_state['state']['hk_state']
        net = hk.transform_with_state(net_fn)
    except KeyError:
        logging.info('Loaded old stateless function. Converting to stateful.')
        hk_state = {}
        net = hk.with_empty_state(hk.transform(net_fn))
    return net, params, hk_state


def load_ensemble(directory):
    """Load an ensemble of models."""
    models = []
    with os.scandir(directory) as dirs:
        for entry in dirs:
            if entry.is_dir():
                workdir = entry.path
                state_dir = workdir+'/checkpoints/best_state.pkl'
                if os.path.exists(state_dir):
                    net, params, hk_state = load_model(workdir, is_training=False)
                    models.append((net, params, hk_state))
    return models


def create_model(config: ml_collections.ConfigDict, is_training=True):
    """Return a function that applies the graph model."""
    match config.model_str:
        case 'GCN':
            return GCN(config, is_training)
        case 'MPEU':
            return MPEU(config, is_training)
        case 'SchNet':
            return SchNet(config, is_training)
        case 'PaiNN':
            return get_painn(config)
        case 'MPEU_uq':
            return MPEU_uq(config, is_training)
        case _:
            raise ValueError(
                f'Model string {config.model_str} not recognized')
