"""This module defines function used for making inferences with an existing
model."""
import os
import json
import glob
from collections import defaultdict
from typing import Optional, Sequence

import jax
import jraph
import haiku as hk
import numpy as np
from absl import logging
import ase.db
import pandas as pd

from jraph_MPEU.input_pipeline import (
    DataReader, load_split_dict, ase_row_to_jraph,
    atoms_to_nodes_list
)
from jraph_MPEU.utils import (
    get_valid_mask, load_config, get_normalization_dict, scale_targets
)
from jraph_MPEU.models.loading import load_model, load_ensemble


def get_predictions_graph(
    dataset: Sequence[jraph.GraphsTuple],
    net: hk.Transformed,
    params: hk.Params,
    hk_state: hk.State,
    mc_dropout: Optional[bool] = False,
    batch_size: Optional[int] = 32,
) -> Sequence[jraph.GraphsTuple]:
    """Get whole graph predictions for a single dataset of graphs.

    Args:
        dataset: list of jraph.GraphsTuple
        net: the model object with an apply function that applies the GNN model
            on a batch of graphs.
        params: haiku parameters used by the net.apply function
        hk_state: haiku state for batch norm in apply function
        mc_dropout: whether to use monte-carlo dropout to estimate the
            uncertainty of the model. If true, preds gets and extra dimension
            for the extra sampling of each graph
        batch_size: size of batches used in the data_reader
    
    Returns:
        a nested list with length of len(dataset), each item is a list of
        graphs with length 1 if mc_dopout=False, length 10 otherwise.
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


def get_predictions(
    dataset: Sequence[jraph.GraphsTuple],
    net: hk.Transformed,
    params: hk.Params,
    hk_state: hk.State,
    mc_dropout: Optional[bool] = False,
    batch_size: Optional[int] = 32,
    label_type: Optional[str] = 'scalar',
) -> np.ndarray:
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
        batch_size: size of batches used in the data_reader
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


def get_predictions_ensemble(
    dataset: Sequence[jraph.GraphsTuple],
    net_list: Sequence[hk.Transformed],
    params_list: Sequence[hk.Params],
    hk_state_list: Sequence[hk.State],
    batch_size: Optional[int] = 32,
    label_type: Optional[str] = 'scalar',
) -> np.ndarray:
    """Get predictions on dataset from an ensemble of models.
    Args:
        dataset: list of jraph.GraphsTuple
        net_list: list of model objects (see get_predictions)
        params_list: list of haiku parameter trees (see get_predictions)
        hk_state_list: list of haiku states (see get_predictions)
        label_type: string that describes the type of label

    Returns:
        numpy array with shape (N, m, d), where N is the length of the dataset,
        m is number of ensemble models, d is feature length of prediction
    """
    predictions_ensemble = []
    for i, (net, params, hk_state) in enumerate(zip(net_list, params_list, hk_state_list)):
        logging.info(f"Inference using model {i}/{len(net_list)}")
        predictions_single = get_predictions(
            dataset, net, params, hk_state, False, batch_size, label_type)
        predictions_ensemble.append(predictions_single)
    predictions_ensemble = np.concatenate(predictions_ensemble, axis=1)
    return predictions_ensemble


def get_predictions_graph_ensemble(
    dataset: Sequence[jraph.GraphsTuple],
    net_list: Sequence[hk.Transformed],
    params_list: Sequence[hk.Params],
    hk_state_list: Sequence[hk.State],
    batch_size: Optional[int] = 32,
) -> Sequence[jraph.GraphsTuple]:
    """Get predictions on dataset from an ensemble of models.
    Args:
        dataset: list of jraph.GraphsTuple
        net_list: list of model objects (see get_predictions)
        params_list: list of haiku parameter trees (see get_predictions)
        hk_state_list: list of haiku states (see get_predictions)

    Returns:
        nested list of jraph.GraphsTuple with shape (len(dataset),len(net_list),)
    """
    predictions_ensemble = []
    for i, (net, params, hk_state) in enumerate(zip(net_list, params_list, hk_state_list)):
        logging.info(f"Inference using model {i}/{len(net_list)}")
        predictions_single = get_predictions_graph(
            dataset, net, params, hk_state, False, batch_size)
        predictions_single = [pred[0] for pred in predictions_single]
        predictions_ensemble.append(predictions_single)
    return predictions_ensemble


def _chunks(rows, n):
    """Yield successive n-sized chunks from rows, and start, stop indices."""
    for i in range(0, len(rows), n):
        yield rows[i:i+n], i, i+n


def get_results_df(
    workdir: str,
    results_path: str,
    limit: Optional[int] = None,
    mc_dropout: Optional[bool] = False,
    ensemble: Optional[bool] = False,
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return a pandas dataframe with predictions and their database entries.
    Args:
        workdir: directory to load model, dataset from
        results_path: file path where results csv is saved
        limit: maximum number of datapoints to load and predict on, None means
            no limit on number
        mc_dropout: whether to use monte-carlo dropout to get prediction
            standard deviations
        ensemble: whether to load an ensemble of models. If True, workdir has
            to point to multiple valid working directories, each with models
            saved
        data_path: path to database to predict on, if None, path in
            config.data_file will be used

    This function loads the config, and splits. Then connects to the
    database specified in config.data_file, gets the predictions of entries in
    the database. Finally the function puts together the key_val_pairs
    including id from the database, the predictions and splits of each database
    row into a pandas.DataFrame."""

    assert not (mc_dropout and ensemble),"mc_dropout and ensemble not supported at the same time."

    if ensemble:
        net, params, hk_state, config, num_list, norm_dict = load_ensemble(workdir)
    else:
        net, params, hk_state, config, num_list, norm_dict = load_model(
            workdir, is_training=mc_dropout)
    label_str = config.label_str

    # we need to order the graphs and labels as dicts to be able to refer
    # to their splits later
    graphs_dict = {} # key: asedb_id, value corresponding graph
    labels_dict = {} # key: asedb_id, value corresponding label
    split_loaded = False
    if data_path is None:
        data_path = config.data_file
        assert os.path.exists(data_path),f"ASE_db not found at {data_path}"
        ase_db = ase.db.connect(data_path)
        try:
            split_dict = load_split_dict(workdir)
            split_loaded = True
        except FileNotFoundError:
            logging.info("No split file found, assuming everything is test data.")
            split_dict = {i+1: 'test' for i in range(ase_db.count())}
            split_loaded = False
    else:
        assert os.path.exists(data_path),f"ASE_db not found at {data_path}"
        ase_db = ase.db.connect(data_path)
        logging.info("No split file found, assuming everything is test data.")
        split_dict = {i+1: 'test' for i in range(ase_db.count())}
        split_loaded = False

    if split_loaded:
        iterator = split_dict.items()
    else:
        iterator = ase_db.select(limit=limit)

    rows = []
    for i, iter_return in enumerate(iterator):
        if split_loaded and (config.data_file == data_path):
            id_single, split = iter_return
            row = ase_db.get(id_single)
        else:
            row = iter_return
            id_single = row.id
            split = 'test'

        if i%10000 == 0:
            logging.info(f'Rows read: {i}')
        if limit is not None:
            if i >= limit:
                # limit the number of read graphs, for faster loading
                break
        graph = ase_row_to_jraph(row)
        # test if nodes are a subset of num_list, otherwise ignore this row
        if not set(graph.nodes) <= set(num_list):
            continue # TODO: test this
        n_edge = int(graph.n_edge[0])
        if config.num_edges_max is not None:
            if n_edge > config.num_edges_max:  # do not include graphs with too many edges
                continue
        graphs_dict[id_single] = graph
        if label_str in row.key_value_pairs:
            label = row.key_value_pairs[label_str]
        else:
            label = 0 # TODO: test this
        labels_dict[id_single] = label
        row_dict = row.key_value_pairs  # initialze row dict with key_val_pairs
        row_dict['asedb_id'] = row.id
        row_dict['n_edge'] = n_edge
        row_dict['split'] = split  # convert from one-based id
        row_dict['formula'] = row.formula
        if 'auid' in row_dict:
            row_dict.pop('aurl', None) # redundant, if auid is present
        rows.append(pd.DataFrame([row_dict]))

    
    # Normalize graphs and targets
    #num_list = list(range(100))  # TODO: this is only a hack to make inference
    # across databases easier, this should be reverted in the future
    # aflow_x_mp
    logging.info("Converting atomic numbers...")
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
    if norm_dict is None:
        logging.info("Getting normalization...")
        match config.label_type:
            case 'scalar':
                norm_dict = get_normalization_dict(
                    graphs_split['train'], config.aggregation_readout_type)
            case 'scalar_non_negative':
                norm_dict = get_normalization_dict(
                    graphs_split['train'], 'scalar_non_negative')
            case 'class'|'class_binary'|'class_multi':
                norm_dict = {}

    logging.info('Predicting on dataset...')
    graphs = list(graphs_dict.values())

    if ensemble:
        preds = get_predictions_ensemble(
            graphs, net, params, hk_state, config.batch_size, config.label_type,
            )  # shape (N, m, d)
    else:
        preds = get_predictions(
            graphs, net, params, hk_state, mc_dropout, config.batch_size,
            config.label_type)  # shape (N, m, d)

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

    logging.info('Saving model predictions...')
    # NOTE: for now, only works with scalar predictions, and uncertainties
    inference_df = pd.DataFrame({})
    match (mc_dropout or ensemble, config.model_str):
        case [False, 'MPEU_uq']:
            inference_df['prediction'] = preds[:, 0, 0]
            inference_df['prediction_uq'] = preds[:, 0, 1]
        case [True, 'MPEU_uq']:
            inference_df['prediction'] = np.mean(preds[:, :, 0], axis=1)
            inference_df['prediction_uq'] = np.mean(preds[:, :, 1], axis=1)
            inference_df['prediction_std'] = np.std(preds[:, :, 0], axis=1)
        case [False, _]:
            inference_df['prediction'] = preds[:, 0, 0]
        case [True, _]:
            inference_df['prediction'] = np.mean(preds[:, :, 0], axis=1)
            inference_df['prediction_std'] = np.std(preds[:, :, 0], axis=1)

    df_path = os.path.join(workdir, results_path)
    max_rows = 1_000_000
    if len(rows) > max_rows:
        header = True
        # split the data into chunks to avoid out of memory (due to pd.concat)
        logging.info("Concatenating and writing in chunks...")
        for rows_chunk, i_start, i_stop in _chunks(rows, max_rows):
            rows_df = pd.concat(rows_chunk, axis=0, ignore_index=True)
            pred_df = inference_df[i_start:i_stop]
            pred_df.index = rows_df.index
            rows_df = pd.concat([rows_df, pred_df], axis=1)
            rows_df.to_csv(df_path, header=header, index=False, mode='a')
            header = False
    else:
        logging.info("Concatenating rows, writing directly...")
        rows_df = pd.concat(rows, axis=0, ignore_index=True)
        rows_df = pd.concat([rows_df, inference_df], axis=1)
        rows_df.to_csv(df_path, index=False, mode='w')
