"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""

import tempfile
import unittest

import numpy as np

from jraph_MPEU_configs import default_test as cfg

import jraph_MPEU.train as train
from jraph_MPEU.models import load_model
from jraph_MPEU.inference import (
    get_predictions,
    load_inference_file
)
from jraph_MPEU.input_pipeline import get_datasets, load_data


class TestInference(unittest.TestCase):
    """Test inference.py methods and classes."""
    def test_get_predictions_qm9(self):
        workdir = 'tests/qm9_test_run'
        net, params = load_model(workdir)
        dataset, _, _, _ = load_data(workdir)
        dataset = {key: reader.data for key, reader in dataset.items()}
        #preds = get_predictions(dataset, net, params)
        preds = {
            key: get_predictions(data_split, net, params)
            for key, data_split in dataset.items()
        }
        for split, pred in preds.items():
            self.assertIsInstance(pred, type(np.array([])))


if __name__ == '__main__':
    unittest.main()
