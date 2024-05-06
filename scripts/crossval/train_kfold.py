import os

from typing import (
    Sequence
)

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import tensorflow as tf
from sklearn.model_selection import KFold
import ase.db

from jraph_MPEU.train import train_and_evaluate
from jraph_MPEU.input_pipeline import save_split_dict

flags.DEFINE_integer('n_splits', 10, help='number of cross validation splits')
flags.DEFINE_integer('i_fold', 0, help='cross validation fold index')
FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def get_db_ids(db_name, selection, limit):
    """Get the database entry ids using db_name for the database filename,
    and using subset specified in selection. limit is the maximum number of
    entries selected."""
    db = ase.db.connect(db_name)
    ids = []
    for row in db.select(selection=selection, limit=limit):
        ids.append(row.id)
    return ids


def get_data_indices_ith_fold(data: list, n_splits: int, i_fold: int, seed: int):
    """Return the data split in a three-way-split (training, validation, test).
    Gives the i-th fold of n_split folds."""
    assert i_fold < n_splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_split_ids = []
    for i, (_, test_index) in enumerate(kf.split(data)):
        if i == i_fold:
            # the i-th split is the test split
            test_split_ids = test_index.tolist()
        elif i == (i_fold+1)%n_splits:
            # the (i+1)th split is the validation split
            val_split_ids = test_index.tolist()
        else:
            # all other splits get added to the train split
            train_split_ids.extend(test_index.tolist())
    # NOTE: kf.split gives indices of the specific split, not the data we put
    # into it back, but split up, i.e. [10, 20, 30, 40, 50] gets split into
    # [0, 1, 2], [3] and [4].
    train_split = [data[i] for i in train_split_ids]
    val_split = [data[i] for i in val_split_ids]
    test_split = [data[i] for i in test_split_ids]
    return train_split, val_split, test_split

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    if not os.path.exists(f'./{FLAGS.workdir}'):
        os.makedirs(f'./{FLAGS.workdir}')
    if FLAGS.config.log_to_file:
        logging.get_absl_handler().use_absl_log_file(
            'absl_logging', f'./{FLAGS.workdir}')
        flags.FLAGS.mark_as_parsed()
        logging.set_verbosity(logging.INFO)

    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    indices = get_db_ids(
        FLAGS.config.data_file, FLAGS.config.selection,
        FLAGS.config.limit_data)

    train_ids, val_ids, test_ids = get_data_indices_ith_fold(
        indices, FLAGS.n_splits, FLAGS.i_fold, FLAGS.config.seed)

    split_path = os.path.join(FLAGS.workdir, 'splits.json')
    if not os.path.exists(split_path):
        save_split_dict(
            {'train': train_ids, 'validation': val_ids, 'test': test_ids},
            FLAGS.workdir)

    train_and_evaluate(FLAGS.config, FLAGS.workdir)

if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
