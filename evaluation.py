"""Evaluate the model using checkpoints and best state from workdir.

The model with the best validation loss is saved during training and loaded
here. The model weights are saved in the pickle file, after they are loaded,
the model can be built using the config.json.
"""

import pickle
import argparse

from absl import logging
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import jax

from models import GNN
from utils import (
    load_config,
    get_valid_mask
)
from input_pipeline import (
    get_datasets,
    DataReader
)


def load_data(workdir):
    """Load evaluation splits."""
    config = load_config(workdir)
    dataset, dataset_raw, mean, std = get_datasets(config)  # might refactor
    return dataset, dataset_raw, mean, std


def load_model(workdir):
    """Load model to evaluate on."""
    state_dir = workdir+'/checkpoints/best_state.pkl'
    with open(state_dir, 'rb') as state_file:
        best_state = pickle.load(state_file)
    config = load_config(workdir)
    # load the model params
    params = best_state['state']['params']
    net_fn = GNN(config)
    net = hk.without_apply_rng(hk.transform(net_fn))
    return net, params


def get_predictions(dataset, net, params):
    """Get predictions for a single dataset split."""
    reader = DataReader(
        data=dataset, batch_size=32, repeat=False)
    @jax.jit
    def predict_batch(graphs):
        labels = graphs.globals
        #graphs = replace_globals(graphs)

        mask = get_valid_mask(labels, graphs)
        pred_graphs = net.apply(params, graphs)
        predictions = pred_graphs.globals
        return predictions, mask
    preds = np.array([])
    for graph in reader:
        preds_batch, mask = predict_batch(graph)
        # get only the valid, unmasked predictions
        preds_valid = preds_batch[mask]
        print(preds_valid)
        preds = np.concatenate([preds, preds_valid], axis=0)

    return preds


def main(args):
    workdir = args.folder

    print('Loading datasets.')
    dataset, dataset_raw, mean, std = load_data(workdir)
    data_list = dataset['validation'].data
    print('Loading model.')
    net, params = load_model(workdir)
    print('Predicting.')
    preds = get_predictions(data_list, net, params)
    targets = [graph.globals[0] for graph in data_list]
    print(np.shape(preds))
    print(np.shape(targets))
    plt.scatter(targets, preds)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show regression plot and loss curve.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/default',
        help='input directory name')
    args = parser.parse_args()
    main(args)
