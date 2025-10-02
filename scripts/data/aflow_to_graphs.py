"""This script converts a csv file that has aflow data (previously generated
with get_aflow_csv.py) and turns it into a ase.db with graph features."""

from functools import partial
from ast import literal_eval

import numpy as np
import pandas
import ase
from ase import Atoms

from absl import app
from absl import flags
from absl import logging

from jraph_MPEU.input_pipeline import (
    get_graph_cutoff,
    get_graph_knearest
)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'file_in', 'databases/aflow/default.csv', 'input csv filename')
flags.DEFINE_string(
    'file_out', 'databases/aflow/graphs_12knn_vec.db', 'output ase.db filename')
flags.DEFINE_string('cutoff_type', 'knearest', 'choose the cutoff type, \
    knearest or const')
flags.DEFINE_float('cutoff', 12.0, 'cutoff value for knearest or const cutoff')


def str_to_array(str_array):
    """Return a numpy array converted from a single string, representing an array."""
    return np.array(literal_eval(str_array))

def dict_to_ase(aflow_dict):
    """Return ASE atoms object from a pandas dict, produced by AFLOW json response."""
    cell = str_to_array(aflow_dict['geometry'])
    positions = str_to_array(aflow_dict['positions_fractional'])
    symbols = aflow_dict['compound']
    structure = Atoms(
        symbols=symbols,
        scaled_positions=positions,
        cell=cell,
        pbc=(1, 1, 1))
    structure.wrap()
    return structure


def convert_row(row_df, cutoff, cutoff_type):
    _, row = row_df
    atoms = dict_to_ase(row)  # get the atoms object from each row
    row = row.to_dict()
    row.pop('geometry', None)
    row.pop('positions_fractional', None)
    row.pop('compound', None)
    row['cutoff_type'] = cutoff_type
    row['cutoff_val'] = cutoff

    if cutoff_type == 'const':
        _, _, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
    elif cutoff_type == 'knearest':
        cutoff = int(cutoff)
        _, _, edges, senders, receivers = get_graph_knearest(atoms, cutoff)
    elif cutoff_type == 'fc':
        raise ValueError(f'Cutoff type {cutoff_type} only available for \
            non-periodic systems.')
    else:
        raise ValueError(f'Cutoff type {cutoff_type} not recognised.')

    data = {}
    data['senders'] = senders
    data['receivers'] = receivers
    data['edges'] = edges

    return atoms, row, data


def main(args):
    """Load aflow data from csv file into ase database with graph features.

    We add edges and adjacency list. Only update the database with missing
    entries by comparing list of identifiers (AUIDs).
    """
    logging.set_verbosity(logging.INFO)
    if len(args) > 1:
        raise app.UsageError('Too many command-line arguments.')
    convert_row_ = partial(
        convert_row, cutoff=FLAGS.cutoff, cutoff_type=FLAGS.cutoff_type)
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    df = pandas.read_csv(FLAGS.file_in, index_col=0)
    auids_csv = df['auid']
    n_rows_df = len(auids_csv)
    logging.info(f"Length of dataframe: {n_rows_df}")

    n_rows_db: int
    with ase.db.connect(FLAGS.file_out, append=True) as db:
        n_rows_db = db.count()
        logging.info(f"Length of ASE-DB: {n_rows_db}")
    n_batch = 1000 # number of rows that are converted and written in a single
    # transaction
    i_start = n_rows_db # start where the db index ends
    while i_start < n_rows_df:
        logging.info(f"Indices to write: {i_start}:{i_start+n_batch}")
        df_to_write = df[i_start:i_start+n_batch]

        # convert df_rows to db_rows
        rows_converted = []
        for row in df_to_write.iterrows():
            rows_converted.append(convert_row_(row))
        logging.info("Writing rows...")
        # write db_rows to db
        with ase.db.connect(FLAGS.file_out, append=True) as db:
            for atoms, row, data in rows_converted:
                db.write(atoms, key_value_pairs=row, data=data)
        i_start += n_batch

    return 0


if __name__ == "__main__":
    app.run(main)
