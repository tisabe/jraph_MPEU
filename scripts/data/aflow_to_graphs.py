import numpy as np
import pandas
import ase
from ase import Atoms
from ase.visualize import view

from absl import app
from absl import flags
from absl import logging

from ast import literal_eval

from asedb_to_graphs import (
    get_graph_fc,
    get_graph_cutoff,
    get_graph_knearest
)


FLAGS = flags.FLAGS
flags.DEFINE_string('file_in', None, 'input csv filename')
flags.DEFINE_string('file_out', None, 'output ase.db filename')
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


def main(args):
    """Load aflow data from csv file into ase database with graph features.

    We add edges and adjacency list. Only update the database with missing
    entries by comparing list of identifiers (AUIDs).
    """
    logging.set_verbosity(logging.INFO)
    if len(args) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # needs parameters: cutoff type, cutoff dist, discard unconnected graphs
    aflow_df = pandas.read_csv(FLAGS.file_in, index_col=0)
    auids_csv = aflow_df['auid']
    logging.info(f"Length of dataframe: {len(auids_csv)}")

    # get list of AUIDs from ASE-DB
    auids_db = []
    with ase.db.connect(FLAGS.file_out, append=True) as db_out:
        logging.info(f"Length of ASE-DB: {db_out.count()}")
        for row in db_out.select():
            auids_db.append(row.key_value_pairs['auid'])
    # calculate the difference between the sets of auids
    auids_diff = set(auids_csv).difference(set(auids_db))
    logging.info(f"Difference between auid sets: {len(auids_diff)}")

    # filter df by auids that are not in the db yet
    aflow_df = aflow_df[aflow_df['auid'].isin(auids_diff)]
    logging.info(f"Structures to write: {len(aflow_df.index)}")

    #with ase.db.connect(FLAGS.file_out, append=True) as db_out:
    db_out = ase.db.connect(FLAGS.file_out, append=True)
    for count, (i, row) in enumerate(aflow_df.iterrows()):
        if count % 10000 == 0:
            logging.info(f'Step {count}')
        atoms = dict_to_ase(row)  # get the atoms object from each row
        row = row.to_dict()
        row.pop('geometry', None)
        row.pop('positions_fractional', None)
        row.pop('compound', None)

        # calculate adjacency of graph as senders and receivers
        cutoff = FLAGS.cutoff
        if FLAGS.cutoff_type == 'const':
            nodes, atom_positions, edges, senders, receivers = get_graph_cutoff(atoms, cutoff)
        elif FLAGS.cutoff_type == 'knearest':
            cutoff = int(cutoff)
            nodes, atom_positions, edges, senders, receivers = get_graph_knearest(atoms, cutoff)
        elif FLAGS.cutoff_type == 'fc':
            nodes, atom_positions, edges, senders, receivers = get_graph_fc(atoms)
        else:
            raise ValueError(f'Cutoff type {args.cutoff_type} not recognised.')

        # get property dict from all keys in the row
        prop_dict = {}
        data = {}
        for key in row.keys():
            val = row[key]
            prop_dict[key] = val
        data['senders'] = senders
        data['receivers'] = receivers
        data['edges'] = edges
        # add information about cutoff
        prop_dict['cutoff_type'] = FLAGS.cutoff_type
        prop_dict['cutoff_val'] = cutoff

        # save in new database
        db_out.write(atoms, key_value_pairs=prop_dict, data=data)
        if count < 3:
            logging.info(prop_dict)
            #logging.info(atoms)
            #view(atoms)
    return 0


if __name__ == "__main__":
    app.run(main)
