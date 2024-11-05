"""
Test a specified database by visualizing unit cell and printing atoms object.
"""
import os
import pickle
import re

from absl import flags
from absl.testing import absltest
import jax
import haiku as hk
import numpy as np

from jraph_MPEU.utils import load_config
from jraph_MPEU.models.loading import create_model
from jraph_MPEU.input_pipeline import get_datasets, DataReader


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'workdir', None, 'Working directory to print config and params from.',
    required=True)


def string_correlation(str_a, str_b):
    """Find correlation of strings by shifting them 'past each other',
    and calculating how many letters match at every shift."""
    count = 0
    for shift in range(1, len(str_a)+len(str_b)):
        for letter_a, letter_b in zip(str_a[:shift], str_b[-shift:]):
            count += (letter_a == letter_b)
    return count/min(len(str_a), len(str_b))


def save_state_safely(state: dict, workdir: str):
    """This saves the state dict in checkpoints/best_state.pkl under wordir.
    Also creates a backup as checkpoints/best_state_copy.pkl."""
    state_path = os.path.join(workdir, 'checkpoints/best_state.pkl')
    backup_path = os.path.join(workdir, 'checkpoints/best_state_copy.pkl')
    # just to be save, write backup first, then write new state
    with open(state_path, 'rb') as file:
        state_backup = pickle.load(file)
    with open(backup_path, 'wb') as file:
        pickle.dump(state_backup, file)
    with open(state_path, 'wb') as file:
        pickle.dump(state, file)


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""
    def test_print(self):
        config = load_config(FLAGS.workdir)
        state_path = os.path.join(FLAGS.workdir, 'checkpoints/best_state.pkl')
        with open(state_path, 'rb') as state_file:
            state = pickle.load(state_file)
        keys_old = list(state['state']['params'].keys())
        keys_old.sort()

        model = create_model(config, is_training=False)
        model_fn = hk.transform_with_state(model)

        config.limit_data = config.batch_size
        datasets, _ = get_datasets(config, FLAGS.workdir)
        train_reader = DataReader(
            data=datasets['train'],
            batch_size=config.batch_size,
            repeat=True,
            seed=config.seed_datareader)

        init_graphs = next(train_reader)
        rng = jax.random.PRNGKey(config.seed_weights)
        params, _ = model_fn.init(rng, init_graphs)
        keys_new = list(params.keys())
        keys_new.sort()

        print("Key proposal:")
        keys_matching = {}
        for key_new in keys_new:
            matches = [string_correlation(key_new, key_old) for key_old in keys_old]
            best_match_i = np.argmax(matches)
            key_old = keys_old[best_match_i]
            keys_matching[key_new] = key_old
            print(key_new, ': ', key_old, ': ', np.ceil(matches[best_match_i]))
            shape_old = state['state']['params'][key_old]['w'].shape
            shape_new = params[key_new]['w'].shape
            assert shape_old == shape_new, f'{key_new}: {key_old}'
        # matches might be not correct, better solve the "assignment problem"
        # this could be done with the "Hungarian algorithm"
        for key_new, key_old in keys_matching.items():
            state['state']['params'][key_new] = state['state']['params'].pop(key_old)

        if input('Convert state with new keys? [y/n]') == 'y':
            save_state_safely(state, FLAGS.workdir)


if __name__ == "__main__":
    absltest.main()
