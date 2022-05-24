"""This script plots the energy or other properties depending on different
deformations of the unit cell of a specific material."""

import argparse

from jraph_MPEU.input_pipeline import get_datasets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show cell deformation energies.')
    parser.add_argument('-f', '-F', type=str, dest='file', default=None,
                        help='data directory name')
    parser.add_argument('-limit', type=int, dest='limit', default=None,
                        help='limit number of database entries to be selected')
    parser.add_argument('-key', type=str, dest='key', default=None,
                        help='key name to plot')
    args_main = parser.parse_args()
    main(args_main)
