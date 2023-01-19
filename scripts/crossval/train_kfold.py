import os

from typing import (
    Sequence
)

from absl import app
from absl import flags
from absl import logging
import jax
import jraph
from ml_collections import config_flags
import ml_collections
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import ase.db

from jraph_MPEU.train import train_and_evaluate

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

    for i, (train_index, test_index) in enumerate(kf.split(ids)):
        if i == i_fold:
            return train_index.tolist(), test_index.tolist()


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

    train_and_evaluate(FLAGS.config, FLAGS.workdir)

if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
