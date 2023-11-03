"""Test the functions of the input_pipeline_test.py module."""

import tempfile
import unittest

import numpy as np
import jraph

from jraph_MPEU.normalization import GraphPreprocessor, ElementEncoder


class Unittest(unittest.TestCase):
    """Testing class."""
    def setUp(self):
        """Set up default test case."""
        graph0 = jraph.GraphsTuple(
            n_node=np.array([4]), nodes={'feature': np.array([0,0,2,2])},
            n_edge=np.array([4]), edges={'feature': np.array([3,3,4,4])},
            senders=np.array([0,1,2,3]), receivers=np.array([0,1,2,3]),
            globals={'categorical_feature': np.array([0]),
            'scalar_feature': np.array([0])})
        graph1 = jraph.GraphsTuple(
            n_node=np.array([6]), nodes={'feature': np.array([0,0,2,2,4,4])},
            n_edge=np.array([6]), edges={'feature': np.array([3,3,4,4,5,5])},
            senders=np.array([0,1,2,3,4,5]), receivers=np.array([0,1,2,3,4,5]),
            globals={'categorical_feature': np.array([1]),
            'scalar_feature': np.array([1])})
        self.graphs_default = [graph0, graph1]
        self.property_dict_globals = {
            'globals': {
                'categorical_feature': 'one_hot',
                'scalar_feature': 'scalar_intrinsic'
            }
        }
        self.property_dict_default = {
            'globals': {
                'feature': 'one_hot'
            },
            'nodes': {
                'feature': 'scalar_intrinsic'
            },
            'edges': {
                'feature': 'scalar_intrinsic'
            }
        }

    def test_ElementEncoder(self):
        element_list = [
            np.array([8,1,1]), # H2O
            np.array([6,1,1,1,1])] # Methane
        encoder = ElementEncoder()
        encoder.fit(element_list)
        transformed_list = encoder.transform(element_list)
        print(transformed_list)

    def test_GraphPreprocessor(self):
        config = {}
        processor = GraphPreprocessor(self.property_dict_globals, config)
        transform_dict = processor.fit(self.graphs_default)
        print(transform_dict)
        for prop_name, transform in transform_dict['globals'].items():
            print(transform.get_params())
        print(transform_dict['globals']['scalar_feature'].mean_)
        print(transform_dict['globals']['categorical_feature'].categories_)

    def test_batched_graphs(self):
        """Test that an error is raised, if there are batched graphs in list."""
        graphs_batched = jraph.batch(self.graphs_default)
        list_batched = [graphs_batched, graphs_batched]
        config = {}
        processor = GraphPreprocessor(self.property_dict_globals, config)
        self.assertRaises(AssertionError, processor.fit(list_batched))


if __name__ == '__main__':
    unittest.main()