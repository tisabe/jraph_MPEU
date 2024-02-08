"""Add the fold keyword to a ase database that indicates whether row contains noble gas element."""

import ase.db
from absl import flags
from absl import app


FLAGS = flags.FLAGS
flags.DEFINE_string('db_name', None, 'database file name')


def main(argv):
    noble_symbols = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']

    with ase.db.connect(FLAGS.db_name) as db:
        for i, row in enumerate(db.select()):
            if i%10000 == 0:
                print(f'Step {i}')
            # set default value
            fold = 1
            for sym in noble_symbols:
                if sym in row.symbols:
                    # change the value if there is a noble element
                    fold = 0
            db.update(row.id, fold=fold)

        count = db.count(selection='fold=0')
        print(f'Number of rows with fold 0: {count}')

        count = db.count(selection='fold=1')
        print(f'Number of rows with fold 1: {count}')

    return 0

if __name__ == "__main__":
    app.run(main)
