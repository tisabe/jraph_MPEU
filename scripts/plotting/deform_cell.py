"""This script plots the energy or other properties depending on different
deformations of the unit cell of a specific material."""

import os
import argparse
from typing import Sequence, Iterable
import json

from absl import logging
from ase import Atoms
import ase.db
import numpy as np
import jax
import jraph
import matplotlib.pyplot as plt

from jraph_MPEU.models import load_model
from jraph_MPEU.input_pipeline import (
    get_graph_knearest,
    get_graph_cutoff,
    get_graph_fc,
    DataReader
)
from jraph_MPEU.utils import (
    get_valid_mask,
    load_config
)


def get_predictions(dataset, net, params):
    """Get predictions for a single dataset split."""
    # TODO: put this into package model
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


def get_graph_type(atoms, cutoff_type, cutoff_val):
    if cutoff_type == 'const':
        nodes, _, edges, senders, receivers = get_graph_cutoff(atoms, cutoff_val)
    elif cutoff_type == 'knearest':
        cutoff_val = int(cutoff_val)
        nodes, _, edges, senders, receivers = get_graph_knearest(atoms, cutoff_val)
    elif cutoff_type == 'fc':
        nodes, _, edges, senders, receivers = get_graph_fc(atoms)
    else:
        raise ValueError(f'Cutoff type {cutoff_type} not recognised.')
    graph = jraph.GraphsTuple(
        n_node=np.asarray([len(nodes)]),
        n_edge=np.asarray([len(senders)]),
        nodes=nodes, edges=edges,
        globals=None,
        senders=np.asarray(senders), receivers=np.asarray(receivers))
    return graph


def atoms_to_jraph(
        atoms: Atoms, senders, receivers, edges
) -> jraph.GraphsTuple:
    nodes = atoms.get_atomic_numbers()

    graph = jraph.GraphsTuple(
        n_node=np.asarray([len(nodes)]),
        n_edge=np.asarray([len(senders)]),
        nodes=nodes, edges=edges,
        globals=None,
        senders=np.asarray(senders), receivers=np.asarray(receivers))

    return graph

def get_deformations(
        atoms: Atoms, multipliers: Iterable
) -> Sequence[Atoms]:
    """Return list with different deformations of Atoms.

    Args:
        atoms: ase.Atoms object to be deformed
        start: smallest multiplier of unit cell length
        stop: largest multiplier of unit cell length
        num: number of different deformations to calculate
    """
    original_cell = atoms.get_cell()
    if sum(sum(original_cell[:])) == 0: # TODO: get better criterion
        # if there is no unit cell defined, define a cubic one with unit length
        atoms.set_cell(np.identity(3))
        original_cell = atoms.get_cell()
    atoms_list = []
    for mult in multipliers:
        atoms.set_cell(original_cell*mult, scale_atoms=True)
        atoms_list.append(atoms[:])
    return atoms_list


def main(args):
    """Main function to get atoms and predict with model."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder

    config = load_config(workdir)
    logging.info('Loading model.')
    net, params = load_model(workdir)
    logging.info('Loading datasets.')
    #dataset, _, mean, std = load_data(workdir)
    with open(os.path.join(workdir, 'atomic_num_list.json'), 'r') as list_file:
        num_list = json.load(list_file)

    file = config.data_file
    ase_db = ase.db.connect(file)
    ids = ['mp-149', 'mp-165', 'mp-2352', 'mp-1061395', 'mp-1224973']
    ids = [
        'aflow:09e8e3c8f41716e4',
        'aflow:20a474bdfc8057ac',
        #'aflow:6454e4f00b068452',
        #'aflow:2bb55fb3d0b119fb'
        ]
    for id_single in ids:
        #row = ase_db.get(selection=f'mp_id={id_single}')
        row = ase_db.get(selection=f'auid={id_single}')
        cutoff_type = row['cutoff_type']
        cutoff_val = row['cutoff_val']
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()

        logging.info('Calculating deformations.')
        multipliers = np.linspace(0.7, 1.3, 100)
        deforms = get_deformations(atoms, multipliers)
        graphs = []
        # get graphs from deformations
        for atom in deforms:
            graph = get_graph_type(atom, cutoff_type, cutoff_val)
            graphs.append(graph)

        # transform node values from atomic numbers to classes
        for graph in graphs:
            nodes = graph.nodes
            for i, num in enumerate(nodes):
                nodes[i] = num_list.index(num)
            graph._replace(nodes=nodes)

        preds = get_predictions(graphs, net, params)
        plt.plot(multipliers, preds, label=f'{formula}, {id_single}')
    plt.xlabel('unit cell multiplier')
    plt.ylabel(f'{config.label_str} prediction, standardized')
    plt.title('Stability of prediction for volume expansion')
    plt.legend()
    plt.show()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction for deformations of the unit cell.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/fc',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
