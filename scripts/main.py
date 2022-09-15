"""Run training and evaluation on a model with parameters defined in config.

Results and checkpoints are saved in workdir."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import tensorflow as tf

from jraph_MPEU import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    """Check GPU, flags and perform model training."""
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if os.path.exists(FLAGS.workdir + '/STOPPED_EARLY'):
        logging.warning('Started training on model that stopped early.')
        return
    if os.path.exists(FLAGS.workdir + '/REACHED_MAX_STEPS'):
        logging.warning('Started training on model that \
            reached maximum number of steps.')
        return
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    if not os.path.exists(f'./{FLAGS.workdir}'):
        os.makedirs(f'./{FLAGS.workdir}')
    if FLAGS.config.log_to_file:
        logging.get_absl_handler().use_absl_log_file(
            'absl_logging',
            f'./{FLAGS.workdir}')
        flags.FLAGS.mark_as_parsed()
        logging.set_verbosity(logging.INFO)

    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
