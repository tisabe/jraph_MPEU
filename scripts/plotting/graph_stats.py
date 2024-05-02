import os
import json

from absl import app
from absl import flags
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ase.db
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

FLAGS = flags.FLAGS
flags.DEFINE_list('files', ['databases/aflow/graphs_12knn_vec.db'],
    'database file names')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_string('out_path', 'databases/data_summary.csv', 'Where csv is saved')


def main(_):
    symbols_intersect = set()
    for filename in FLAGS.files:
        if not os.path.exists(filename):
            raise ValueError(f"Database {filename} was not found!")
        db = ase.db.connect(filename)
        print(f"Number of entries in {filename}: {db.count()}")
        symbols_db = set()
        n_atoms_db = []
        for row in tqdm(db.select()):
            symbols = set(row.symbols)
            symbols_db |= symbols  # update symbols for this database
            n_atoms_db.append(row.natoms)

        print(sorted(symbols_db), len(symbols_db))
        print("Max num of atoms: ", max(n_atoms_db))
        if symbols_intersect:
            symbols_intersect &= symbols_db
        else:
            symbols_intersect = symbols_db
    blacklist = {
        'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', # noble gases
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr' # Actinoids
    }
    symbols_intersect -= blacklist
    print(sorted(symbols_intersect), len(symbols_intersect))

    df = pd.DataFrame({})
    for filename in FLAGS.files:
        if not os.path.exists(filename):
            raise ValueError(f"Database {filename} was not found!")
        db = ase.db.connect(filename)
        valid_rows = []
        for row in tqdm(db.select()):
            # only if all species in the atoms object are part of the
            # species intersection, add row to valid rows
            symbols = set(row.symbols)
            if symbols <= symbols_intersect:
                valid_rows.append(row)
                struct_pmg = AseAtomsAdaptor.get_structure(row.toatoms())
                analyzer = SpacegroupAnalyzer(struct_pmg, symprec=0.1)
                spg = analyzer.get_space_group_number()

                row_dict = {
                    'database': filename,
                    'natoms': row.natoms,
                    'formula': row.formula,
                    'asedb_id': row.id,
                    'spacegroup_pmg': spg,
                    **row.key_value_pairs,
                }
                df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
                # add calculated spacegroup to asedb
                db.update(row.id, spacegroup_pmg=spg)
        print(len(valid_rows))
    print(df.describe())
    df.to_csv(FLAGS.out_path, index=False)


if __name__ == "__main__":
    app.run(main)
