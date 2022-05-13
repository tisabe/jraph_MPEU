"""Perform cross validation with montecarlo sampling parameters.

The paramters are saved as lists in a ml_collections config dict.
By calling this script with different seeds, a random parameter combination
is taken from the possible grid of parameters, with replacement, so duplicate
configs are possible. Each run is saved in a working directory identified by
its seed.
"""
import os
import random

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf
import jax

from utils import Config_iterator
from train import train_and_evaluate

flags.DEFINE_integer('seed', 0, help='cross validation seed index')
FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    """Start training with a randomly sampled config."""
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

    # set the random seed, so in each run we get a different choice of config
    random.seed(FLAGS.seed)

    iterator = Config_iterator(FLAGS.config)
    # get all possible configs
    configs = [config for config in iterator]
    # get a random config
    config = random.choice(configs)
    train_and_evaluate(config, FLAGS.workdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
