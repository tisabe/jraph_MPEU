"""Evaluate model(s) in workdir on dataset specified in arguments.

Creates a results dataframe that is saved in workdir."""

import os
import json

from absl import app
from absl import flags
from absl import logging
import jax

from jraph_MPEU.inference import get_results_df


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory where model data is saved.', required=True)
flags.DEFINE_string('data_path', None, 'Path to database to evaluate model on.', required=True)
flags.DEFINE_string('results_path', None, 'Path to results file.', required=True)
flags.DEFINE_string('split_file', None, 'Directory to source split file.')
flags.DEFINE_integer('limit', None, 'Limit for number of data to predict.')
flags.DEFINE_bool('mc_dropout', False, 'If monte-carlo dropout is used for evaluation.')
flags.DEFINE_bool('ensemble', False, 'If an ensemble is loaded and used for evaluation.')


def main(argv):
    """Check GPU, flags and perform model training."""
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # import split file from different workdir if the argument is given
    if FLAGS.split_file is not None:
        # load the splits dict from different workdir
        with open(FLAGS.split_file, 'r', encoding="utf-8") as splits_file:
            splits_dict = json.load(splits_file, parse_int=True)
        # save the splits dict in this workdir
        with open(os.path.join(FLAGS.workdir, 'splits.json'), 'w', encoding="utf-8") as splits_file:
            json.dump(splits_dict, splits_file, indent=4, separators=(',', ': '))

    get_results_df(
        workdir=FLAGS.workdir,
        results_path=FLAGS.results_path,
        limit=FLAGS.limit,
        mc_dropout=FLAGS.mc_dropout,
        ensemble=FLAGS.ensemble,
        data_path=FLAGS.data_path)


if __name__ == '__main__':
    app.run(main)
