"""This module defines function used for making inferences with an existing
model."""
import os
import pickle

import jax
import numpy as np
from absl import logging

from jraph_MPEU.input_pipeline import DataReader, load_data
from jraph_MPEU.utils import get_valid_mask
from jraph_MPEU.models import load_model


def get_predictions(dataset, net, params):
    """Get predictions for a single dataset split.

    Args:
        dataset: list of jraph.GraphsTuple
        net: the model object with an apply function that applies the GNN model
            on a batch of graphs.
        params: haiku parameters used by the net.apply function

    Returns:
        1-D numpy array of predictions from the dataset
    """
    reader = DataReader(
        data=dataset, batch_size=32, repeat=False)
    @jax.jit
    def predict_batch(graphs):
        mask = get_valid_mask(graphs)
        pred_graphs = net.apply(params, graphs)
        predictions = pred_graphs.globals
        return predictions, mask
    preds = np.array([])
    for graph in reader:
        preds_batch, mask = predict_batch(graph)
        # get only the valid, unmasked predictions
        preds_valid = preds_batch[mask]
        preds = np.concatenate([preds, preds_valid], axis=0)

    return preds


def load_inference_file(workdir, redo=False):
    """Return the inferences of the model and data defined in workdir.

    This function finds inferences that have already been saved in the working
    directory. If a file with inferences has been found, they are loaded and
    returned in a dictionary with splits as keys.
    If there is no file with inferences in workdir or 'redo' is true, the model
    is loaded and inferences are calculated.
    """
    inference_dict = {}
    path = workdir + '/inferences.pkl'
    if not os.path.exists(path) or redo:
        # compute the inferences
        logging.info('Loading model.')
        net, params = load_model(workdir)
        logging.info('Loading datasets.')
        dataset, _, mean, std = load_data(workdir)
        splits = dataset.keys()
        print(splits)

        for split in splits:
            data_list = dataset[split].data
            logging.info(f'Predicting {split} data.')
            preds = get_predictions(data_list, net, params)
            targets = [graph.globals[0] for graph in data_list]
            # scale the predictions and targets using the std
            preds = preds*float(std) + mean
            targets = np.array(targets)*float(std) + mean

            inference_dict[split] = {}
            inference_dict[split]['preds'] = preds
            inference_dict[split]['targets'] = targets

        with open(path, 'wb') as inference_file:
            pickle.dump(inference_dict, inference_file)
    else:
        # load inferences from dict
        logging.info('Loading existing inference.')
        with open(path, 'rb') as inference_file:
            inference_dict = pickle.load(inference_file)
    return inference_dict
