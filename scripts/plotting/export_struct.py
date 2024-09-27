"""Script to export a structure from a database in xsf format for plotting."""

from absl import app
from absl import flags
import ase.db
from ase import Atoms

FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'databases/aflow/graphs_12knn_vec.db', 'input database name')
flags.DEFINE_string('id', 'aflow:c9ba9ee6a035b51a', 'identifier for database')


def main(argv):
    db = ase.db.connect(FLAGS.db)
    row = db.get(selection=f"auid={FLAGS.id}")
    atoms = row.toatoms()
    ase.io.write(
        filename=FLAGS.db.split('/')[0]+'/'+FLAGS.id+'.xsf',
        images=atoms, format='xsf'
    )
    return 0

if __name__ == "__main__":
    app.run(main)
