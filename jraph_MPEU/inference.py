"""This module defines function used for making inferences with an existing
model."""
import os
import json
import glob

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
import ase.db
import pandas

from jraph_MPEU.input_pipeline import (
    DataReader, load_split_dict, ase_row_to_jraph,
    atoms_to_nodes_list
)
from jraph_MPEU.utils import (
    get_valid_mask, load_config, get_normalization_dict, scale_targets
)
from jraph_MPEU.models.loading import load_model


def get_predictions(dataset, net, params, hk_state, label_type,
    mc_dropout=False, batch_size=32):
    """Get predictions for a single dataset split.

    Args:
        dataset: list of jraph.GraphsTuple
        net: the model object with an apply function that applies the GNN model
            on a batch of graphs.
        params: haiku parameters used by the net.apply function
        hk_state: haiku state for batch norm in apply function
        label_type: if the predictions are scalars for regression (value:
            'scalar') or logits for classification (value: 'class')
        mc_dropout: whether to use monte-carlo dropout to estimate the
            uncertainty of the model. If true, preds gets and extra dimension
            for the extra sampling of each graph

    Returns:
        1-D numpy array of predictions from the dataset
    """
    n_samples = 10 if mc_dropout else 1
    # TODO: test this, especially that it does not modify dataset in place
    key = jax.random.PRNGKey(42)

    @jax.jit
    def predict_batch(graphs, rng, hk_state):
        pred_graphs, _ = net.apply(params, hk_state, rng, graphs)
        predictions = pred_graphs.globals
        return predictions
    preds = []
    for _ in range(n_samples):
        reader = DataReader(
            data=dataset, batch_size=batch_size, repeat=False)
        preds_sample = np.array([])
        for graph in reader:
            key, subkey = jax.random.split(key)

            mask = get_valid_mask(graph)
            preds_batch = predict_batch(graph, subkey, hk_state)
            if label_type == 'class':
                # turn logits into probabilities by applying softmax function
                #print('Converted array: ', jnp.array(preds_batch))
                preds_batch = jax.nn.softmax(jnp.array(preds_batch), axis=1)
                # get only one probability p, the other one is just 1-p
                # this corresponds to the predicted probability of being in the
                # zeroth class
                preds_batch = preds_batch[:, 0]
                preds_batch = jnp.expand_dims(preds_batch, 1)

            # get only the valid, unmasked predictions
            preds_valid = preds_batch[mask]
            preds_sample = np.concatenate([preds_sample, preds_valid], axis=0)
        preds.append(preds_sample)
    return preds


def get_results_df(workdir, limit=None, mc_dropout=False):
    """Return a pandas dataframe with predictions and their database entries.

    This function loads the config, and splits. Then connects to the
    database specified in config.data_file, gets the predictions of entries in
    the database. Finally the function puts together the key_val_pairs
    including id from the database, the predictions and splits of each database
    row into a pandas.DataFrame."""

    config = load_config(workdir)
    split_dict = load_split_dict(workdir)
    label_str = config.label_str
    net, params, hk_state = load_model(workdir, is_training=mc_dropout)

    graphs = []
    labels = []
    ase_db = ase.db.connect(config.data_file)
    inference_df = pandas.DataFrame({})
    for i, (id_single, split) in enumerate(split_dict.items()):
        if i%10000 == 0:
            logging.info(f'Rows read: {i}')
        if limit is not None:
            if i >= limit:
                # limit the number of read graphs, for faster loading
                break
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
        row_dict['asedb_id'] = row.id
        row_dict['n_edge'] = n_edge
        row_dict['split'] = split  # convert from one-based id
        row_dict['numbers'] = row.numbers  # get atomic numbers, when loading
        # the csv from file, this has to be converted from string to list
        row_dict['formula'] = row.formula
        inference_df = pandas.concat(
            [inference_df, pandas.DataFrame([row_dict])], ignore_index=True)
    # Normalize graphs and targets
    # Convert the atomic numbers in nodes to classes and set number of classes.
    num_path = os.path.join(workdir, 'atomic_num_list.json')
    with open(num_path, 'r', encoding="utf-8") as num_file:
        num_list = json.load(num_file)

    # convert to dict for atoms_to_nodes function
    graphs_dict = dict(enumerate(graphs))
    labels_dict = dict(enumerate(labels))
    graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)
    graphs_dict = {key: graph._replace(globals=[label])
        for (key, graph), label in zip(graphs_dict.values(), labels_dict.values())}
    pooling = config.aggregation_readout_type  # abbreviation
    norm_dict = get_normalization_dict(
        graphs_dict.values(), pooling)
    graphs = list(graphs_dict.values())
    #labels = list(graphs_dict.values())

    logging.info('Predicting on dataset.')
    preds = get_predictions(
        graphs, net, params, hk_state, config.label_type, mc_dropout,
        config.batch_size)
    if config.label_type == 'scalar':
        # scale the predictions using the std and mean
        logging.debug(f'using {pooling} pooling function')
        #preds = np.array(preds)*std + mean
        preds = scale_targets(graphs, preds, norm_dict)

    if mc_dropout:
        preds = np.transpose(np.array(preds))
        print(np.shape(preds))
        print(preds)
        preds_mean = np.mean(preds, axis=1)
        preds_std = np.std(preds, axis=1)
        inference_df['prediction_mean'] = preds_mean
        inference_df['prediction_std'] = preds_std
    else:
        preds = preds[0]
        # add row with predictions to dataframe
        inference_df['prediction'] = preds

    return inference_df

def get_results_kfold(workdir_super, mc_dropout=False):
    """Generate dataframe with results from k models and data split into k
    folds.
    """
    directories = glob.glob(workdir_super+'/id*')
    # make dataframe to append to
    inference_df = pandas.DataFrame({})
    base_ids = None
    base_config = None
    for i, workdir in enumerate(directories):
        split_dict = load_split_dict(workdir)
        ids = list(split_dict.keys())
        if i == 0:
            base_ids = list(split_dict.keys())
            base_config = load_config(workdir)
        else:
            # make sure that the folds have the same underlying data
            assert set(ids) == set(base_ids)

    ase_db = ase.db.connect(base_config.data_file)
    graphs = []
    labels = []
    for i, (id_single, split) in enumerate(split_dict.items()):
        if i%10000 == 0:
            logging.info(f'Rows read: {i}')
        row = ase_db.get(id_single)
        graph = ase_row_to_jraph(row)
        n_edge = int(graph.n_edge)
        graphs.append(graph)
        label = row.key_value_pairs[base_config.label_str]
        labels.append(label)
        row_dict = row.key_value_pairs  # initialze row dict with key_val_pairs
        row_dict['asedb_id'] = row.id
        row_dict['n_edge'] = n_edge
        row_dict['split'] = split  # convert from one-based id
        row_dict['numbers'] = row.numbers  # get atomic numbers, when loading
        # the csv from file, this has to be converted from string to list
        row_dict['formula'] = row.formula
        inference_df = inference_df.append(row_dict, ignore_index=True)
    # Normalize graphs and targets
    # Convert the atomic numbers in nodes to classes and set number of classes.
    num_path = os.path.join(workdir, 'atomic_num_list.json')
    with open(num_path, 'r', encoding="utf-8") as num_file:
        num_list = json.load(num_file)

    # convert to dict for atoms_to_nodes function
    graphs_dict = dict(enumerate(graphs))
    labels_dict = dict(enumerate(labels))
    graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)
    pooling = base_config.aggregation_readout_type  # abbreviation
    graphs = list(graphs_dict.values())


    net, params, hk_state = load_model(workdir, is_training=mc_dropout)
    # TODO: maybe finish this. Other possibility: after end of training,
    # generate results dataframe and combine afterwards
