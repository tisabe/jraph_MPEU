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

import train
import input_pipeline

flags.DEFINE_integer('n_splits', 10, help='number of cross validation splits')
flags.DEFINE_integer('i_fold', 0, help='cross validation fold index')
FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def get_data_indices_ith_fold(N: int, n_splits: int, i_fold: int, seed: int):
    '''Return train and validation indices for the i-th of n_splits folds
    in a dataset of N samples.'''
    assert i_fold<n_splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dataset = [0]*N
    
    for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        if i==i_fold:
            return train_index.tolist(), val_index.tolist()

def cross_validation_single_fold(
    config: ml_collections.ConfigDict, 
    workdir: str, 
    dataset: Sequence[jraph.GraphsTuple],
    std: float,
    n_splits: int, 
    i_fold: int
    ):
    '''Perform cross validation on one fold of a KFold CV split.
    i_fold determines the index of which fold to evaluate in this function call.'''
    
    train_index, val_index = get_data_indices_ith_fold(len(dataset), 
        n_splits, i_fold, config.seed)

    data_dict = {}
    data_dict['train'] = [dataset[i] for i in train_index]
    data_dict['validation'] = [dataset[i] for i in val_index]
    state, min_loss = train.train(config, data_dict, std, workdir)




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

    dataset, mean, std = input_pipeline.get_dataset_single(FLAGS.config)
    cross_validation_single_fold(FLAGS.config, 
        FLAGS.workdir, 
        dataset, std, FLAGS.n_splits, FLAGS.i_fold)

if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)