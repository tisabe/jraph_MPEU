from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import tensorflow as tf

import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)