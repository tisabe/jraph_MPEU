"""This module defines function used for making inferences with an existing
model."""
import os
import pickle

import jax
import numpy as np
from absl import logging
import ase.db
import pandas

from jraph_MPEU.input_pipeline import (
    DataReader, load_data, load_split_dict, ase_row_to_jraph,
    atoms_to_nodes_list
)
from jraph_MPEU.utils import (
    get_valid_mask, load_config, add_labels_to_graphs, normalize_targets,
    scale_targets
)
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
        dataset, mean, std = load_data(workdir)
        splits = dataset.keys()
        print(splits)

        for split in splits:
            data_list = dataset[split]
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


def get_results_df(workdir):
    """Return a pandas dataframe with predictions and their database entries.

    This function loads the config, and splits. Then connects to the
    database specified in config.data_file, gets the predictions of entries in
    the database. Finally the function puts together the key_val_pairs
    including id from the database, the predictions and splits of each database
    row into a pandas.DataFrame."""

    config = load_config(workdir)
    split_dict = load_split_dict(workdir)
    label_str = config.label_str

    graphs = []
    labels = []
    ase_db = ase.db.connect(config.data_file)
    inference_df = pandas.DataFrame({})
    for i, (id_single, split) in enumerate(split_dict.items()):
        if i%10000 == 0:
            logging.info(f'Rows read: {i}')
        row = ase_db.get(id_single)
        graph = ase_row_to_jraph(row)
        n_edge = int(graph.n_edge)
        if config.num_edges_max is not None:
            if n_edge > config.num_edges_max:  # do not include graphs with too many edges
                continue
        graphs.append(graph)
        label = row.key_value_pairs[label_str]
        labels.append(label)
        row_dict = row.key_value_pairs  # initialze row dict with key_val_pairs
        row_dict['id'] = row.id
        row_dict['n_edge'] = n_edge
        row_dict['split'] = split  # convert from one-based id
        #row_dict['symbols']
        inference_df = inference_df.append(row_dict, ignore_index=True)
    # Normalize graphs and targets
    # Convert the atomic numbers in nodes to classes and set number of classes.
    graphs, _ = atoms_to_nodes_list(graphs)
    _, mean, std = normalize_targets(
        graphs, labels, config)
    graphs = add_labels_to_graphs(graphs, labels)

    net, params = load_model(workdir)
    logging.info('Predicting on dataset.')
    preds = get_predictions(graphs, net, params)
    # scale the predictions using the std
    preds = scale_targets(graphs, preds, mean, std, config.aggregation_readout_type)

    # add row with predictions to dataframe
    inference_df['prediction'] = preds

    return inference_df
