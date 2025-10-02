"""This script plots the energy or other properties depending on different
deformations of the unit cell of a specific material."""

import os
import argparse
from typing import Sequence, Iterable
import json
import pickle

from ase import Atoms
import ase.db
import numpy as np
import jraph
import matplotlib.pyplot as plt

from jraph_MPEU.models.loading import load_ensemble
from jraph_MPEU.inference import (
    get_predictions_ensemble,
    get_predictions_graph_ensemble
)
from jraph_MPEU.input_pipeline import (
    get_graph_knearest,
    get_graph_cutoff,
    get_graph_fc
)
from jraph_MPEU.utils import (
    load_config,
    load_norm_dict,
    scale_targets
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
        multipliers: iterable with different multipliers that 
            unit cell and positions are scaled by
    """
    cell = atoms.get_cell()
    if not any(atoms.get_pbc()):
        # if there is no unit cell defined, define a cubic one with unit length
        atoms.set_cell(np.identity(3))
        cell = atoms.get_cell()
    atoms_list = []
    for mult in multipliers:
        atoms.set_cell(cell*mult, scale_atoms=True)
        atoms_list.append(atoms[:])
    return atoms_list


def plot_length_scaling(workdir):
    """Plot model uncertainty for varying length scale."""
    config = load_config(workdir)
    models = load_ensemble(workdir)
    with open(os.path.join(workdir, 'atomic_num_list.json'), 'r', encoding='utf-8') as list_file:
        num_list = json.load(list_file)

    norm_path = os.path.join(workdir, 'normalization.json')
    norm_dict = load_norm_dict(norm_path)

    file = config.data_file
    ase_db = ase.db.connect(file)
    ids = [
        'aflow:09e8e3c8f41716e4',
        'aflow:20a474bdfc8057ac',
        'aflow:6454e4f00b068452',
        'aflow:13398b3b86de2c68']
    """ids = [
        'mp-149', 'mp-165', 'mp-1',
        'mp-1105', 'mp-1265', 'mp-69',
        'mp-10779', 'mp-126', 'mp-1840']"""
    ids = [10001, 20001, 30001]
    fig, ax = plt.subplots()
    for id_single in ids:
        #row = ase_db.get(selection=f'mp_id={id_single}')
        #row = ase_db.get(selection=f'auid={id_single}')
        row = ase_db.get(id_single)
        cutoff_type = row['cutoff_type']
        cutoff_val = row['cutoff_val']
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()
        print("Formula: ", formula)

        print('Calculating deformations.')
        multipliers = np.linspace(0.9, 1.1, 101)
        deforms = get_deformations(atoms, multipliers)
        #volumes = [deform.get_volume()/len(deforms[0]) for deform in deforms]
        volumes = multipliers
        graphs = []
        # get graphs from deformations
        for atom in deforms:
            graph = get_graph_type(atom, cutoff_type, cutoff_val)
            nodes = graph.nodes
            for i, num in enumerate(nodes):
                nodes[i] = num_list.index(num)
            graph._replace(nodes=nodes)
            graphs.append(graph)

        preds = get_predictions_ensemble(graphs, models, config.label_type)
        # rescale the predictions
        preds[:, :, 0] = scale_targets(
            graphs, preds[:, :, 0], norm_dict)
        std = norm_dict['std']
        std = np.where(std==0, 1, std)
        preds[:, :, 1] *= std

        # get prediction and target for undeformed graph
        middle_index = int(len(multipliers)/2)
        vol = volumes[middle_index]
        target = row[config.label_str]
        mean_preds = np.mean(preds[:, :, 0], axis=1)
        sigma_al = np.mean(preds[:, :, 1], axis=1)
        sigma_ep = np.mean(preds[:, :, 0]**2 - np.expand_dims(mean_preds, 1)**2, axis=1)
        sigma = sigma_al + sigma_ep

        # get recalibration model and recalibrate uncertainties
        model_path = os.path.join(workdir, 'recal_model.pkl')
        with open(model_path, 'rb') as model_file:
            recal_model = pickle.load(model_file)
        sigma = recal_model.transform(sigma)

        #ax.plot(volumes, mean_preds, label=f'{formula}, {id_single}')
        #ax.fill_between(
        #    volumes, mean_preds-sigma, mean_preds+sigma, color='b', alpha=.1)
        #ax.plot([vol], [target], marker='o', markersize=3, color='red')
        ax.plot(volumes, sigma, label=f'{formula}, {id_single}')
    #plt.yscale('log')
    plt.xlabel('Length multiplier')
    plt.ylabel(f'{config.label_str} total uncertainty, standardized')
    plt.title('Uncertainty quantification for volume expansion')
    plt.legend()
    plt.show()
    fig.savefig(workdir + '/deform_cell.png', bbox_inches='tight', dpi=600)


def displace_atom(atoms, vector, index):
    """Displace one atom at index in atoms by vector."""
    pos = atoms.get_positions()
    pos[index] += vector
    atoms.set_positions(pos)
    return atoms


def get_displacements(workdir):
    config = load_config(workdir)
    models = load_ensemble(workdir)
    file = config.data_file
    ase_db = ase.db.connect(file)
    db_id = 1
    row = ase_db.get(db_id)
    cutoff_type = row['cutoff_type']
    cutoff_val = row['cutoff_val']
    atoms = row.toatoms()
    formula = atoms.get_chemical_formula()
    print("Formula: ", formula)
    graph = get_graph_type(atoms, cutoff_type, cutoff_val)
    preds = get_predictions_graph_ensemble([graph, graph], models)
    print(preds)


def main(args):
    """Main function to get atoms and predict with model."""
    workdir = args.folder

    #plot_length_scaling(workdir)
    get_displacements(workdir)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction for deformations of the unit cell.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/mp/knn',
        help='input directory name')
    args_main = parser.parse_args()
    main(args_main)
