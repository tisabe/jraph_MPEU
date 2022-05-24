"""This script plots the energy or other properties depending on different
deformations of the unit cell of a specific material."""

import argparse
from typing import Sequence

from absl import logging
from ase import Atoms
import numpy as np

from jraph_MPEU.models import load_model
from jraph_MPEU.input_pipeline import (
    load_data,
    get_graph_knearest
)


def get_deformations(
        atoms: Atoms, start: float, stop: float, num: int
) -> Sequence[Atoms]:
    """Return list with different deformations of Atoms.

    Args:
        atoms: ase.Atoms object to be deformed
        start: smallest multiplier of unit cell length
        stop: largest multiplier of unit cell length
        num: number of different deformations to calculate
    """
    original_cell = atoms.get_cell()
    print(original_cell[:])
    multipliers = np.linspace(start, stop, num)
    print(multipliers)
    atoms_list = []
    for mult in multipliers:
        atoms.set_cell(original_cell*mult, scale_atoms=True)
        atoms_list.append(atoms[:])
    print(atoms_list)
    return atoms_list


def main(args):
    """Main function to get atoms and predict with model."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder

    logging.info('Loading model.')
    net, params = load_model(workdir)
    logging.info('Loading datasets.')
    dataset, _, mean, std = load_data(workdir)

    atoms = Atoms('N3', [(0, 0, 0), (1, 0, 0), (0, 0, 1)])
    atoms.set_cell(1 * np.identity(3))
    deforms = get_deformations(atoms, 0.5, 1.5, 3)
    for atom in deforms:
        print(atom.get_positions())
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction for deformations of the unit cell.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/test',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
