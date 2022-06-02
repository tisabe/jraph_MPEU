"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""

import tempfile
import unittest

from jraph_MPEU_configs import default_test as cfg

import jraph_MPEU.train as train
from jraph_MPEU.models import load_model
from jraph_MPEU.inference import (
    get_predictions,
    load_inference_file
)
from jraph_MPEU.input_pipeline import get_datasets


class TestInference(unittest.TestCase):
    """Test train.py methods and classes."""
    def setUp(self):
        self.config = cfg.get_config()
        self.config.limit_data = 100
        self.assertEqual(self.config.batch_size, 32)
        # get testing datasets
        datasets, _, _, _, _ = get_datasets(self.config)
        self.datasets = datasets
        self.data_val_list = self.datasets['validation'].data[:]
        self.test_dir = tempfile.TemporaryDirectory()
        self.workdir = self.test_dir.name
        budget = datasets['train'].budget
        print(budget.n_node)
        print(budget.n_edge)
        print(budget.n_graph)
        budget = datasets['validation'].budget
        print(budget.n_node)
        print(budget.n_edge)
        print(budget.n_graph)
        budget = datasets['test'].budget
        print(budget.n_node)
        print(budget.n_edge)
        print(budget.n_graph)
        # run training to produce model in temp directory
        _, _ = train.train_and_evaluate(
            self.config, self.workdir)

    def test_get_predictions_qm9(self):
        net, params = load_model(self.workdir)
        preds = get_predictions(self.data_val_list, net, params)

if __name__ == '__main__':
    unittest.main()
