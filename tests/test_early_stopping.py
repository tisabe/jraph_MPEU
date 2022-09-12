"""
Test the early stopping using the evaluater in train.
"""
import tempfile

from absl import flags
from absl.testing import absltest

from jraph_MPEU_configs import test_early_stopping as cfg

from jraph_MPEU.input_pipeline import get_datasets, DataReader
from jraph_MPEU.utils import replace_globals
from jraph_MPEU.train import init_state


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""
    def test_manual_queue(self):
        config = cfg.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            datasets, _, std = get_datasets(config, test_dir)
            train_reader = DataReader(
                data=datasets['train'],
                batch_size=config.batch_size,
                repeat=True,
                seed=config.seed)

            init_graphs = next(train_reader)
            init_graphs = replace_globals(init_graphs)
            _, _, evaluater = init_state(config, init_graphs, test_dir)
            eval_splits = ['train', 'validation', 'test']
            evaluater.init_loss_lists(eval_splits)
            evaluater.set_loss_scalar(std)

            # set up decreasing loss curve (no early stopping)
            # the middle entries are irrelevant, only first and last element
            # are checked
            evaluater.early_stopping_queue = [2, 10, 100, 1]
            self.assertFalse(evaluater.check_early_stopping())
            # check that the 0th entry is dropped
            self.assertEqual(evaluater.early_stopping_queue[0], 10)

            # set up increasing loss curve (early stopping)
            evaluater.early_stopping_queue = [1, 10, 100, 2]
            self.assertTrue(evaluater.check_early_stopping())
            # check that the 0th entry is not dropped
            self.assertEqual(evaluater.early_stopping_queue[0], 1)

    def test_update(self):
        config = cfg.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            datasets, _, std = get_datasets(config, test_dir)
            train_reader = DataReader(
                data=datasets['train'],
                batch_size=config.batch_size,
                repeat=True,
                seed=config.seed)
            print(train_reader.budget)

            init_graphs = next(train_reader)
            init_graphs = replace_globals(init_graphs)
            updater, state, evaluater = init_state(config, init_graphs, test_dir)
            eval_splits = ['train', 'validation', 'test']
            evaluater.init_loss_lists(eval_splits)
            evaluater.set_loss_scalar(std)

            print(state['step'])
            evaluater.update(state, datasets, eval_splits)




if __name__ == "__main__":
    absltest.main()
