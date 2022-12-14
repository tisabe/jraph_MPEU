"""
Test a specified database by visualizing unit cell and printing atoms object.
"""
from absl import flags
from absl.testing import absltest

import ase.db
from ase.visualize import view
from ase.spacegroup import get_spacegroup
from ase.neighborlist import NeighborList

from jraph_MPEU.input_pipeline import get_graph_cutoff


FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'QM9/qm9_graphs_fc.db', 'database filename')
flags.DEFINE_integer('num', 1, 'Number of structures to print.')
flags.DEFINE_integer(
    'verb', 1,
    'Verbosity of output. <2 only spacegroup ' +
    'is printed, <3 objects are printed also, =3 atoms are visualized'
)


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""
    def test_not_empty(self):
        """Test that the database is not empty."""
        database = ase.db.connect(FLAGS.file)
        self.assertTrue(len(database) > 0)
    def test_vis_and_print(self):
        """Test db by visualizing and printing database entries."""
        database = ase.db.connect(FLAGS.file)
        num = FLAGS.num
        rows = database.select(limit=num)
        for row in rows:
            if FLAGS.verb > 1:
                print(row)
            #for key, value in row.key_value_pairs.items():
            #    print(key, value)
            atoms = row.toatoms()
            if FLAGS.verb > 1:
                print(atoms)
            if FLAGS.verb > 2:
                view(atoms)

            if atoms.pbc.all():
                # get spacegroup and compare
                sg_num = get_spacegroup(atoms).no
                if FLAGS.verb > 0:
                    print("Actual spacegroup: ", sg_num)

                if "spacegroup_relax" in row.key_value_pairs.keys():
                    if FLAGS.verb > 0:
                        print(
                            "Expected spacegroup: ",
                            row.key_value_pairs["spacegroup_relax"])

    def test_neighbors(self):
        """Check if neighbors in graphs are calculated correctly."""

        database = ase.db.connect(FLAGS.file)
        num = FLAGS.num
        rows = database.select(limit=num)
        for row in rows:
            atoms = row.toatoms()
            key_value_pair = row.key_value_pairs
            cutoff = key_value_pair['cutoff_val']
            cutoff_type = key_value_pair['cutoff_type']

            edges = row.data['edges']
            senders = row.data['senders']
            receivers = row.data['receivers']
            len_edges = len(edges)
            self.assertTrue(len_edges > 0)
            self.assertTrue(len(senders) == len_edges)
            self.assertTrue(len(receivers) == len_edges)
            # TODO: more tests on specific connectivity


if __name__ == "__main__":
    absltest.main()
