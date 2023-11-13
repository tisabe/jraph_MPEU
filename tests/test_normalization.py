"""Test the functions of the input_pipeline_test.py module."""

import tempfile
import unittest

import numpy as np
import jraph
import sklearn.preprocessing as skp

from jraph_MPEU.normalization import (
    GraphPreprocessor, ElementEncoder, ExtrinsicScalarEncoder)


class Unittest(unittest.TestCase):
    """Testing class."""
    def setUp(self):
        """Set up default test case."""
        graph0 = jraph.GraphsTuple(
            n_node=np.array([4]), nodes={'feature': np.array([[0],[0],[2],[2]])},
            n_edge=np.array([4]), edges={'feature': np.array([[3],[3],[4],[4]])},
            senders=np.array([0,1,2,3]), receivers=np.array([0,1,2,3]),
            globals={'categorical_feature': np.array([0]),
            'scalar_feature': np.array([0])})
        graph1 = jraph.GraphsTuple(
            n_node=np.array([6]), nodes={'feature': np.array([[0],[0],[2],[2],[4],[4]])},
            n_edge=np.array([6]), edges={'feature': np.array([[3],[3],[4],[4],[5],[5]])},
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
                'categorical_feature': 'one_hot'
            },
            'nodes': {
                'feature': 'element_list'
            },
            'edges': {
                'feature': 'scalar_intrinsic'
            }
        }

    def test_ExtrinsicScalarEncoder(self):
        features = [
            np.random.randint(0, 10, 4),
            np.random.randint(0, 10, 4)]
        n_nodes = np.random.randint(1, 10, len(features))

        features_per_atom = [
            feature/n_node for feature, n_node in zip(features, n_nodes)]
        mean_expected = sum(features_per_atom)/len(features_per_atom)

        enc = ExtrinsicScalarEncoder()
        enc.fit(features, n_nodes)
        np.testing.assert_array_equal(mean_expected, enc.enc.mean_)
        diff_from_mean = [
            feature_per_atom - mean_expected for feature_per_atom in features_per_atom]
        sq_diffs = [np.square(diff) for diff in diff_from_mean]
        std_expected = np.sqrt(sum(sq_diffs)/len(sq_diffs))
        # for zero variance, sklearn puts scale to 1, so this is how we test it
        std_expected[std_expected == 0] = 1.
        np.testing.assert_array_equal(std_expected, enc.enc.scale_)

        features_tr = enc.transform(features, n_nodes)
        for feature_tr, feature, n_node in zip(features_tr, features, n_nodes):
            np.testing.assert_array_equal(
                feature_tr, (feature - n_node*mean_expected)/std_expected)

        features_fit_tr = enc.fit_transform(features, n_nodes)
        for feature_tr, feature, n_node in zip(features_fit_tr, features, n_nodes):
            np.testing.assert_array_equal(
                feature_tr, (feature - n_node*mean_expected)/std_expected)

        features_inverse = enc.inverse_transform(features_tr, n_nodes)
        for feature_inverse, feature in zip(features_inverse, features):
            np.testing.assert_array_equal(feature_inverse, feature)


    def test_ElementEncoder(self):
        element_list = [
            np.array([[8],[1],[1]]), # H2O
            np.array([[6],[1],[1],[1],[1]])] # Methane
        encoder = ElementEncoder()
        encoder.fit(element_list)
        transformed_list = encoder.transform(element_list)
        np.testing.assert_array_equal(
            transformed_list[0], np.array([[0,0,1],[1,0,0],[1,0,0]]))
        np.testing.assert_array_equal(
            transformed_list[1],
            np.array([[0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]))

        element_list = [
            np.array([['H'],['H'],['O']]), # H2O
            np.array([['C'],['H'],['H'],['H'],['H']])] # Methane
        encoder = ElementEncoder()
        encoder.fit(element_list)
        transformed_list = encoder.transform(element_list)
        np.testing.assert_array_equal(
            transformed_list[0], np.array([[0,1,0],[0,1,0],[0,0,1]]))
        np.testing.assert_array_equal(
            transformed_list[1],
            np.array([[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]))

        inv_list = encoder.inverse_transform(transformed_list)
        for elements, inv_elements in zip(element_list, inv_list):
            np.testing.assert_array_equal(elements, inv_elements)

    def test_ElementEncoder_rand(self):
        n_samples = 50
        element_list = [
            np.random.randint(1, 10, size=(np.random.randint(1, 20),1)) for _ in range(n_samples)]
        encoder = ElementEncoder()
        transformed_list = encoder.fit_transform(element_list)

        inv_list = encoder.inverse_transform(transformed_list)
        for elements, inv_elements in zip(element_list, inv_list):
            np.testing.assert_array_equal(elements, inv_elements)

    def test_GraphPreprocessor_globals(self):
        config = {}
        processor = GraphPreprocessor(self.property_dict_globals, config)
        transform_dict = processor.fit(self.graphs_default)
        #print(transform_dict)
        #for prop_name, transform in transform_dict['globals'].items():
        #    print(transform.get_params())
        self.assertIsInstance(
            transform_dict['globals']['scalar_feature'], skp.StandardScaler)
        self.assertEqual(
            transform_dict['globals']['scalar_feature'].mean_, 0.5)
        self.assertIsInstance(
            transform_dict['globals']['categorical_feature'],
            skp.OneHotEncoder)
        self.assertEqual(
            len(transform_dict['globals']['categorical_feature'].categories_[0]),2)

    def test_batched_graphs(self):
        """Test that an error is raised, if there are batched graphs in list."""
        graphs_batched = jraph.batch(self.graphs_default)
        list_batched = [graphs_batched, graphs_batched]
        config = {}
        processor = GraphPreprocessor(self.property_dict_globals, config)
        with self.assertRaises(AssertionError):
            processor.fit(list_batched)

    def test_graphs_transform(self):
        config = {}
        processor = GraphPreprocessor(self.property_dict_default, config)
        transform_dict = processor.fit(self.graphs_default)

        for graph in self.graphs_default:
            print(graph)
        graphs_tr = processor.transform(self.graphs_default)
        print()
        for graph in graphs_tr:
            print(graph)


if __name__ == '__main__':
    unittest.main()