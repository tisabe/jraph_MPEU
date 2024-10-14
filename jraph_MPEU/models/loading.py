"""This module is used to pass the correct model function to the train or
inference script respectively. The model string in the passed config determines
which model is chosen."""

import pickle
import os
import json

from absl import logging
import haiku as hk
import ml_collections

from jraph_MPEU.models.gcn import GCN
from jraph_MPEU.models.mpeu import MPEU
from jraph_MPEU.models.mpeu_uq import MPEU_uq
from jraph_MPEU.models.schnet import SchNet
from jraph_MPEU.models.painn import get_painn
from jraph_MPEU.utils import load_config, load_norm_dict


def load_model(workdir, is_training):
    """Load model to evaluate on."""
    config = load_config(workdir)

    # get the correct max_atomic_number, it was changed from input config in 
    # input_pipeline
    num_path = os.path.join(workdir, 'atomic_num_list.json')
    with open(num_path, 'r', encoding="utf-8") as num_file:
        num_list = json.load(num_file)
        config.max_atomic_number = len(num_list)

    # load the model params
    state_dir = workdir+'/checkpoints/best_state.pkl'
    with open(state_dir, 'rb') as state_file:
        best_state = pickle.load(state_file)
    params = best_state['state']['params']
    logging.info(f'Loaded best state at step {best_state["state"]["step"]}')
    net_fn = create_model(config, is_training)

    # compatibility layer to load old models the were initialized without state
    try:
        hk_state = best_state['state']['hk_state']
        net = hk.transform_with_state(net_fn)
    except KeyError:
        logging.info('Loaded old stateless function. Converting to stateful.')
        hk_state = {}
        net = hk.with_empty_state(hk.transform(net_fn))

    # load target normalization dict
    norm_path = os.path.join(workdir, 'normalization.json')
    norm_dict = load_norm_dict(norm_path)

    return net, params, hk_state, config, num_list, norm_dict


def load_ensemble(directory):
    """Load an ensemble of models."""
    net_list = []
    params_list = []
    hk_state_list = []
    config_list = []
    num_list = []
    norm_dict_list = []
    with os.scandir(directory) as dirs:
        for entry in dirs:
            if entry.is_dir():
                workdir = entry.path
                if os.path.exists(workdir+'/checkpoints/best_state.pkl'):
                    net, params, hk_state, config, num_list, norm_dict = load_model(
                        workdir, is_training=False)
                    net_list.append(net)
                    params_list.append(params)
                    hk_state_list.append(hk_state)
                    config_list.append(config)
                    num_list.append(num_list)
                    norm_dict_list.append(norm_dict)
    return (
        net_list,
        params_list,
        hk_state_list,
        config_list[0],
        num_list[0],
        norm_dict[0]
    )


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
