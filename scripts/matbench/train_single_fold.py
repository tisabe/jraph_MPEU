"""Prepare data and models for training on the matbench benchmark tasks.

This preparation step ensures that models can be trained in parallel in a
slurm array job."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
from matbench.bench import MatbenchBenchmark

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_list('tasks', ['matbench_mp_gap'], 'Tasknames for matbench. Usage: \
    string separated names, e.g. matbench_mp_gap,matbench_perovskites')
flags.DEFINE_integer(
    'index', 0, help='Array index for combined matbench task and fold. A value \
    of 0 corresponds to doing all tasks and folds in sequence. Index must be \
    between 1 and 5*len(tasks), or 0.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    """Check GPU, flags and perform model training."""
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    mb = MatbenchBenchmark(
        autoload=False, subset=FLAGS.tasks)

    id_counter = 1 # initialize counter to make indexable directories
    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            if FLAGS.index==0 or FLAGS.index==id_counter:
                train_inputs, train_outputs = task.get_train_and_val_data(fold)
                if not os.path.exists(f'./{FLAGS.workdir}/matbench_id{id_counter}'):
                    os.makedirs(f'./{FLAGS.workdir}/matbench_id{id_counter}')
            id_counter += 1


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
