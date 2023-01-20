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


def get_data_indices_ith_fold(ids: list, n_splits: int, i_fold: int, seed: int):
    """Save split data for the specified fold of n splits."""
    assert i_fold < n_splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_split = []
    val_split = []
    test_split = []
    for i, (_, test_index) in enumerate(kf.split(ids)):
        if i == i_fold:
            # the i-th split is the test split
            test_split = test_index.tolist()
        elif i == (i_fold+1)%n_splits:
            # the (i+1)th split is the validation split
            val_split = test_index.tolist()
        else:
            # all other splits get added to the train split
            train_split.extend(test_index.tolist())
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
        logging.get_absl_handler().use_absl_log_file('absl_logging', f'./{FLAGS.workdir}') 
        flags.FLAGS.mark_as_parsed()
        logging.set_verbosity(logging.INFO)

    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    indices = get_db_ids(
        FLAGS.config.data_file, FLAGS.config.selection,
        FLAGS.config.limit_data)
    print(indices)

    train_ids, val_ids, test_ids = get_data_indices_ith_fold(
        indices, FLAGS.n_splits, FLAGS.i_fold, FLAGS.config.seed)
    print(train_ids)
    print(val_ids)
    print(test_ids)
    save_split_dict(
        {'train': train_ids, 'validation': val_ids, 'test': test_ids},
        FLAGS.workdir)

    train_and_evaluate(FLAGS.config, FLAGS.workdir)

if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
