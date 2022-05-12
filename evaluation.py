"""Evaluate the model using checkpoints and best state from workdir.

The model with the best validation loss is saved during training and loaded
here. The model weights are saved in the pickle file, after they are loaded,
the model can be built using the config.json.
"""

import pickle
import argparse
import os

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


def main(args):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder

    inference_dict = load_inference_file(workdir, redo=args.redo)

    fig, ax = plt.subplots(2)
    marker_size = 0.3

    splits = inference_dict.keys()
    units = input("Type units of prediction and target: ")
    for split in splits:
        preds = inference_dict[split]['preds']
        targets = inference_dict[split]['targets']
        error = np.abs(preds - targets)
        mae = np.mean(error)
        mse = np.mean(np.square(error))
        print(f'Number of graphs: {len(preds)}')
        print(f'MSE: {mse} {units}')
        print(f'MAE: {mae} {units}')
        label_string = f'{split} \nMAE: {mae:9.3f} {units}'
        ax[0].scatter(targets, preds, s=marker_size, label=label_string)
        ax[1].scatter(targets, error, s=marker_size, label=split)

    # plot x = y regression lines
    x_ref = np.linspace(*ax[0].get_xlim())
    ax[0].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    ax[0].set_title('Model regression performance')
    ax[0].set_ylabel(f'prediction ({units})', fontsize=12)
    ax[1].set_xlabel(f'target ({units})', fontsize=12)
    ax[1].set_ylabel(f'abs. error ({units})', fontsize=12)
    ax[0].set_aspect('equal')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale('log')
    plt.tight_layout()

    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show regression plot and loss curve.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/test_eval',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
