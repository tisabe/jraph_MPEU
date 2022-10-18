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
from jraph_MPEU.inference import get_predictions
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
    net, params, hk_state = load_model(workdir, is_training=False)
    logging.info('Loading datasets.')
    #dataset, _, mean, std = load_data(workdir)
    with open(os.path.join(workdir, 'atomic_num_list.json'), 'r') as list_file:
        num_list = json.load(list_file)

    file = config.data_file
    ase_db = ase.db.connect(file)
    ids = [
        'aflow:09e8e3c8f41716e4',
        'aflow:20a474bdfc8057ac',
        'aflow:6454e4f00b068452',
        'aflow:13398b3b86de2c68'
        ]
    """ids = [
        'mp-149', 'mp-165', 'mp-1',
        'mp-1105', 'mp-1265', 'mp-69',
        'mp-10779', 'mp-126', 'mp-1840']"""
    for id_single in ids:
        #row = ase_db.get(selection=f'mp_id={id_single}')
        row = ase_db.get(selection=f'auid={id_single}')
        cutoff_type = row['cutoff_type']
        cutoff_val = row['cutoff_val']
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()

        logging.info('Calculating deformations.')
        multipliers = np.linspace(0.9, 1.1, 101)
        deforms = get_deformations(atoms, multipliers)
        volumes = [deform.get_volume()/len(deforms[0]) for deform in deforms]
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

        preds = get_predictions(graphs, net, params, hk_state)
        preds = np.array(preds)*1.20607 - 1.405046

        # get prediction and target for undeformed graph
        middle_index = int(len(multipliers)/2)
        vol = volumes[middle_index]
        target = row[config.label_str]
        plt.plot(volumes, preds[0], label=f'{formula}, {id_single}')
        plt.plot([vol], [target], marker='o', markersize=3, color='red')
    plt.xlabel('Volume/atom')
    plt.ylabel(f'{config.label_str} prediction, standardized')
    plt.title('Stability of prediction for volume expansion')
    plt.legend()
    plt.show()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction for deformations of the unit cell.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/mp/knn',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
