"""Script that turns a file with list of pymatgen structures into an ase
database."""

import json

from absl import app
from absl import flags

import ase.db
import pymatgen
import pymatgen.io.ase

FLAGS = flags.FLAGS
flags.DEFINE_string('file_in', None, 'input file name')
flags.DEFINE_string('file_out', None, 'output file name')

def main(argv):
    "Convert pymatgen structures to ase atoms and put into ase.db"
    with open(FLAGS.file_in, 'r') as file_in:
        pmg_list = json.load(file_in)
    print(f'Length of pymatgen list: {len(pmg_list)}')
    adaptor = pymatgen.io.ase.AseAtomsAdaptor
    with ase.db.connect(FLAGS.file_out, append=False) as db_out:

        for i, row in enumerate(pmg_list):
            if i%10000 == 0:
                print(f'Step {i}/{len(pmg_list)}')
                pmg_struct = pymatgen.core.IStructure.from_str(
                    row['structure'], fmt='cif')
                atoms = adaptor.get_atoms(pmg_struct)
                assert atoms is not None
                key_value_pairs = {}
                key_value_pairs['delta_e'] = row['formation_energy_per_atom']
                key_value_pairs['band_gap'] = row['band_gap']
                key_value_pairs['material_id'] = row['material_id']
                db_out.write(atoms, key_value_pairs=key_value_pairs)
        print(f'Count of rows in ase.db: {len(db_out)}')
    return 0

if __name__ == "__main__":
    app.run(main)
