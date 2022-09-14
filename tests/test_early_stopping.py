"""
Test the early stopping using the evaluater in train.
"""
import tempfile
import os
import pickle

from absl.testing import absltest
from absl import logging
import matplotlib.pyplot as plt

from jraph_MPEU_configs import test_early_stopping as cfg

from jraph_MPEU.input_pipeline import get_datasets, DataReader
from jraph_MPEU.utils import replace_globals
from jraph_MPEU.train import init_state, train_and_evaluate


def plot_curves(folder):
    """Plot evaluation curves to check early stopping."""
    metrics_path = folder+'/checkpoints/metrics.pkl'
    splits = ['train', 'validation', 'test']
    with open(metrics_path, 'rb') as metrics_file:
        metrics_dict = pickle.load(metrics_file)

    _, ax = plt.subplots(2, sharex=True)

    for split in splits:
        metrics = metrics_dict[split]
        loss_mse = [row[1][0] for row in metrics]
        loss_mae = [row[1][1] for row in metrics]
        step = [int(row[0]) for row in metrics]
        ax[0].plot(step, loss_mae, label=split)
        ax[1].plot(step, loss_mse, label=split)

    ax[0].legend()
    ax[1].set_xlabel('Gradient step', fontsize=12)
    #ax[0].set_ylabel(r'MSE $(eV^2)$', fontsize=12)
    ax[0].set_ylabel(r'MSE (eV$^2$/atom$^2$)', fontsize=12)
    #ax[1].set_ylabel('MAE (eV)', fontsize=12)
    ax[1].set_ylabel('MAE (eV/atom)', fontsize=12)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.tight_layout()

    plt.show()


class UnitTests(absltest.TestCase):
    """Unit test class. This string only exists to make my linter happy."""
    def test_manual_queue(self):
        """Test the early stopping by setting the queue manually and checking
        the stopping logic."""
        logging.set_verbosity(logging.WARNING)
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
            self.assertFalse(evaluater.check_early_stopping(10_000))
            # check that the 0th entry is dropped
            self.assertEqual(evaluater.early_stopping_queue[0], 10)

            # set up increasing loss curve (early stopping)
            evaluater.early_stopping_queue = [1, 10, 100, 2]
            self.assertTrue(evaluater.check_early_stopping(10_000))
            # check that the 0th entry is not dropped
            self.assertEqual(evaluater.early_stopping_queue[0], 1)

            # check that early stopping is not true when step number is too low
            evaluater.early_stopping_queue = [1, 10, 100, 2]
            self.assertFalse(evaluater.check_early_stopping(10))

    def test_intraining(self):
        """Test early stopping on very small dataset, by overfitting on
        training split."""
        logging.set_verbosity(logging.WARNING)
        config = cfg.get_config()
        with tempfile.TemporaryDirectory() as test_dir:
            self.assertFalse(os.path.exists(test_dir + 'STOPPED_EARLY'))

            train_and_evaluate(config, test_dir)
            #plot_curves(test_dir)
            # check that STOPPED_EARLY file has been created
            self.assertTrue(os.path.exists(test_dir + 'STOPPED_EARLY'))


if __name__ == "__main__":
    absltest.main()
