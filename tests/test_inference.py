"""Test the functions in the train module.

Most tests consist of running model training for some steps and then looking at
the resulting metrics and checkpoints.
"""

import tempfile
import unittest

import numpy as np
import jraph
import haiku as hk

from jraph_MPEU.inference import get_predictions


class TestInference(unittest.TestCase):
    """Test inference.py methods and classes."""
    def test_get_predictions(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
