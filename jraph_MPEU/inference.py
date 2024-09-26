"""This module defines function used for making inferences with an existing
model."""
import os
import json
import glob
from collections import defaultdict

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from absl import logging
import ase.db
import pandas
from tqdm import tqdm

from jraph_MPEU.input_pipeline import (
    DataReader, load_split_dict, ase_row_to_jraph,
    atoms_to_nodes_list
)
from jraph_MPEU.utils import (
    get_valid_mask, load_config, get_normalization_dict, scale_targets,
    load_norm_dict
)
from jraph_MPEU.models.loading import load_model


def get_predictions_graph(dataset, net, params, hk_state,
    mc_dropout=False, batch_size=32):
    """Get whole graph predictions for a single dataset of graphs.
    
    Returns a nested list with length of len(dataset),
    each item is a list of graphs with length 1 if mc_dopout=False,
    length 10 otherwise.
    """
    n_samples = 10 if mc_dropout else 1
    key = jax.random.PRNGKey(42)

    graphs_all = []
    @jax.jit
    def predict_batch(graphs, rng, hk_state):
        graphs_pred, _ = net.apply(params, hk_state, rng, graphs)
        return graphs_pred
    if len(dataset) > 1:
        reader = DataReader(
            data=dataset, batch_size=batch_size, repeat=False)
    else:
        reader = iter(dataset)
    for graph_batch in reader:
        key, subkey = jax.random.split(key)
        graphs_sample = []
        for _ in range(n_samples):
            subkey, pred_key = jax.random.split(subkey)
            graphs_pred = predict_batch(graph_batch, pred_key, hk_state)
            if len(dataset) > 1:
                graphs_pred = jraph.unpad_with_graphs(graphs_pred)
                graphs = jraph.unbatch(graphs_pred)
            else:
                graphs = graphs_pred
            graphs_sample += graphs
        graphs_all.append(graphs_sample)
    return graphs_all


def get_predictions(dataset, net, params, hk_state, mc_dropout=False,
    label_type='scalar', batch_size=32):
    """Get predictions for a single dataset split.

    Args:
        dataset: list of jraph.GraphsTuple
        net: the model object with an apply function that applies the GNN model
            on a batch of graphs.
        params: haiku parameters used by the net.apply function
        hk_state: haiku state for batch norm in apply function
        mc_dropout: whether to use monte-carlo dropout to estimate the
            uncertainty of the model. If true, preds gets and extra dimension
            for the extra sampling of each graph
        label_type: string that describes the type of label

    Returns:
        numpy array with shape (N, m, d), where N is the length of the dataset,
        m is number of mc_dropout samples, d is feature length of prediction
    """
    n_samples = 10 if mc_dropout else 1
    key = jax.random.PRNGKey(42)

    @jax.jit
    def predict_batch(graphs, rng, hk_state):
        pred_graphs, _ = net.apply(params, hk_state, rng, graphs)
        predictions = pred_graphs.globals
        return predictions
    predictions = []
    reader = DataReader(
        data=dataset, batch_size=batch_size, repeat=False)
    # for the following comments: N=number of datapoints, n=batch_size,
    # n_valid=number of valid datapoints in batch, d=feature size,
    # m=number of mc_dropout samples, 1 without mc_dropout

    # new version
    for graph_batch in reader:
        key, subkey = jax.random.split(key)
        # mask is used to index the batch, so squeeze the array
        mask = np.squeeze(get_valid_mask(graph_batch))  # shape (n,)
        valid_indices = np.nonzero(mask)  # shape (n_valid,)
        preds_batch = []
        for _ in range(n_samples):
            subkey, pred_key = jax.random.split(subkey)
            preds_sample = predict_batch(graph_batch, pred_key, hk_state)
            if isinstance(preds_sample, dict):
                preds_sample = np.concatenate(
                    [preds_sample['mu'], preds_sample['sigma']], axis=1
                )
            # preds_sample shape: (n, 2) if uncertainty quantification or
            # classification, (n, f) otherwise (f=1, for now)
            # add additional axis for mc_dropout, shape (n, 1, d)
            preds_sample = np.expand_dims(preds_sample, 1)
            preds_batch.append(preds_sample)
        preds_batch = np.concatenate(preds_batch, axis=1)  # shape (n, m, d)
        preds_batch = preds_batch[valid_indices]  # shape (n_valid, m, d)
        predictions.append(preds_batch)

    predictions = np.concatenate(predictions, 0)  # shape (N, m, d)

    if label_type=='class':
        # compute softmax over the last dimension, and then get probability
        # of the 0th class
        predictions_exp = np.exp(predictions)
        predictions = predictions_exp/np.sum(
            predictions_exp, axis=2, keepdims=True)  # shape (N, m, 2)
        predictions = np.expand_dims(predictions[:, :, 0], 2)  # shape (N, m, 1)

    return predictions


def get_predictions_ensemble(dataset, models, label_type, batch_size=32):
    """Get predictions on dataset from an ensemble of models.
    Args:
        dataset: list of jraph.GraphsTuple
        models: list of tuples, each with (net, params, hk_state) as returned
            by models.loading.load_ensemble
        label_type: string that describes the type of label

    Returns:
        numpy array with shape (N, m, d), where N is the length of the dataset,
        m is number of ensemble models, d is feature length of prediction
    """
    predictions_ensemble = []
    for (net, params, hk_state) in tqdm(models):
        predictions_single = get_predictions(
            dataset, net, params, hk_state, False, label_type, batch_size)
        predictions_ensemble.append(predictions_single)
    predictions_ensemble = np.concatenate(predictions_ensemble, axis=1)
    return predictions_ensemble


def get_predictions_graph_ensemble(dataset, models, batch_size=32):
    predictions_ensemble = []
    for (net, params, hk_state) in tqdm(models):
        predictions_single = get_predictions_graph(
            dataset, net, params, hk_state, False, batch_size)
        predictions_single = [pred[0] for pred in predictions_single]
        predictions_ensemble.append(predictions_single)
    return predictions_ensemble


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

    # we need to order the graphs and labels as dicts to be able to refer
    # to their splits later
    graphs_dict = {} # key: asedb_id, value corresponding graph
    labels_dict = {} # key: asedb_id, value corresponding label
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
        n_edge = int(graph.n_edge[0])
        if config.num_edges_max is not None:
            if n_edge > config.num_edges_max:  # do not include graphs with too many edges
                continue
        graphs_dict[id_single] = graph
        label = row.key_value_pairs[label_str]
        labels_dict[id_single] = label
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
    #num_list = list(range(100))  # TODO: this is only a hack to make inference
    # across databases easier, this should be reverted in the future
    # aflow_x_mp
    graphs_dict = atoms_to_nodes_list(graphs_dict, num_list)

    # also save the graphs in lists corresponding to split
    graphs_split = defaultdict(list)
    for (key_label, label), (key_graph, graph) in zip(
            labels_dict.items(), graphs_dict.items()):
        assert key_label == key_graph  # make very sure that keys match
        graph = graph._replace(globals=np.array([label]))
        graphs_dict[key_graph] = graph
        split = split_dict[key_graph]
        graphs_split[split].append(graph)

    # get and apply normalization to graph targets
    norm_path = os.path.join(workdir, 'normalization.json')
    norm_dict = {}
    if not os.path.exists(norm_path):
        match config.label_type:
            case 'scalar':
                norm_dict = get_normalization_dict(
                    graphs_split['train'], config.aggregation_readout_type)
            case 'scalar_non_negative':
                norm_dict = get_normalization_dict(
                    graphs_split['train'], 'scalar_non_negative')
            case 'class'|'class_binary'|'class_multi':
                norm_dict = {}
    else:
        norm_dict = load_norm_dict(norm_path)
    graphs = list(graphs_dict.values())

    logging.info('Predicting on dataset.')
    preds = get_predictions(
        graphs, net, params, hk_state, mc_dropout, config.label_type,
        config.batch_size)  # shape (N, m, d)
    match (config.label_type, config.model_str):
        case ['scalar'|'scalar_non_negative', 'MPEU_uq']:
            # scale the predicted mean (preds[:, :, 0])
            # uncertainty (preds[:, :, 1]) independently, since the
            # uncertainties should not have an offset
            preds[:, :, 0] = scale_targets(
                graphs, preds[:, :, 0], norm_dict)
            std = norm_dict['std']
            std = np.where(std==0, 1, std)
            preds[:, :, 1] *= std
        case ['scalar'|'scalar_non_negative', _]:
            # scale the predictions using norm_dict
            for i in range(preds.shape[1]):  # account for mc_dropout sampling
                preds[:, i, :] = scale_targets(graphs, preds[:, i, :], norm_dict)
        case _:
            pass

    # NOTE: for now, only works with scalar predictions, and uncertainties
    match (mc_dropout, config.model_str):
        case [False, 'MPEU_uq']:
            inference_df['prediction'] = preds[:, 0, 0]
            inference_df['prediction_uq'] = preds[:, 0, 1]
        case [True, 'MPEU_uq']:
            inference_df['prediction'] = np.mean(preds[:, :, 0], axis=1)
            inference_df['prediction_uq'] = np.mean(preds[:, :, 1], axis=1)
            inference_df['prediction_std'] = np.std(preds[:, :, 0], axis=1)
        case [False, _]:
            inference_df['prediction'] = preds[:, 0, :]
        case [True, _]:
            inference_df['prediction'] = np.mean(preds[:, :, 0], axis=1)
            inference_df['prediction_std'] = np.std(preds[:, :, 0], axis=1)

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
