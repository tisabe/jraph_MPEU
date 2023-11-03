"""This module defines the classes and functions for transformation of input
data. The API is meant to resemble functions in sklearn.preprocessing."""

from typing import List
import pickle
from collections import defaultdict

import jraph
import sklearn.preprocessing as skp


TRANSFORMS = {
    'scalar_intrinsic': skp.StandardScaler,
    'one_hot': skp.OneHotEncoder,
}


def init_new_transformer(input: List, transform_name: str):
    transformer = TRANSFORMS[transform_name]()
    return transformer


class GraphPreprocessor:
    """Class to calculate and apply transformation to graph structured data."""
    def __init__(self, property_dict, config, workdir=None):
        """property_dict contains information about the graph data to be
        transformed. The keys at level 0 are 'globals', 'nodes', 'edges'.
        At level 1, keys are the names of the properties (one or multiple).

        Here is an example for a property_dict:
        property_dict = {
            'globals': {
                'Egap': 'scalar_intrinsic',
                'spacegroup': 'one_hot'
            },
            'nodes': {
                'atomic_number': 'element_list'
            },
            'edges': {
                'distance': 'rbf',
                'bond': 'one_hot'
            }
        }
        
        If workdir argument is passed, automatically saves parameters."""
        if workdir is not None:
            self._autosave = True
        else:
            self._autosave = False
        self._workdir = workdir
        self.property_dict = dict(property_dict) # create a copy

    def save_parameters(self):
        raise NotImplementedError

    def fit(self, graphs: List[jraph.GraphsTuple]):
        graph_values_dict = {
            'globals': {},
            'nodes': {},
            'edges': {}
        }
        for graph in graphs:
            # check that the graphs are not batched
            #assert(type(graph.n_node[0])==int)
            for prop_name, value in graph.globals.items():
                if not prop_name in graph_values_dict['globals']:
                    graph_values_dict['globals'][prop_name] = []
                graph_values_dict['globals'][prop_name].append(value)

            for prop_name, value in graph.nodes.items():
                if not prop_name in graph_values_dict['nodes']:
                    graph_values_dict['nodes'][prop_name] = []
                graph_values_dict['nodes'][prop_name].append(value)
            
            for prop_name, value in graph.edges.items():
                if not prop_name in graph_values_dict['edges']:
                    graph_values_dict['edges'][prop_name] = []
                graph_values_dict['edges'][prop_name].append(value)

        print(graph_values_dict)

        for feature, feature_dict in self.property_dict.items():
            # feature is e.g. 'globals'
            for prop_name, prop_transform in feature_dict.items():
                # prop_name is e.g. 'Egap', prop_transform is e.g. 'scalar'
                transform = TRANSFORMS[prop_transform]()
                self.property_dict[feature][prop_name] = transform
                transform.fit(X=graph_values_dict[feature][prop_name])

        return self.property_dict

    def transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError

    def fit_transform(self, graphs: List[jraph.GraphsTuple]):
        self.fit(graph)
        return self.transform(graphs)

    def inverse_transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError
