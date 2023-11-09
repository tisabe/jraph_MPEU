"""This module defines the classes and functions for transformation of input
data. The API is meant to resemble functions in sklearn.preprocessing."""

from typing import List
import pickle
from collections import defaultdict

import jraph
import sklearn.preprocessing as skp
import numpy as np


class ElementEncoder:
    """Class to encode lists of elements. Specifically for node vectors
    representing elements or atomic numbers."""
    def __init__(self):
        self.enc = skp.OneHotEncoder(sparse=False)

    def fit(self, element_list: List[np.array]):
        flattened_list = np.hstack(element_list) # stacks and flattens arrays
        self.enc.fit(np.expand_dims(flattened_list, 1))

    def transform(self, element_list: List[np.array])->List[np.array]:
        transformed_list = []
        for elements in element_list:
            elements = np.expand_dims(elements, 1)
            transformed = self.enc.transform(elements)
            transformed_list.append(transformed)
        return transformed_list

    def fit_transform(self, element_list: List[np.array])->List[np.array]:
        self.fit(element_list)
        return self.transform(element_list)

    def inverse_transform(self, one_hot_list: List[np.array])->List[np.array]:
        raise NotImplementedError


class ExtrinsicScalarEncoder:
    """Class to encode lists of extrinsic scalars.
    
    Extrinsic scalars, as opposed to intrinsic scalars, scale proportionally
    with the number of nodes in a graph. I.e. total energy might be an example
    of an extrinsic scalar of an atomic graph."""
    def __init__(self):
        self.enc = skp.StandardScaler()

    def fit(self, X: List[np.array], n_nodes: List[int]):
        """Fit encoder using features X and n_node in n_nodes."""
        scaled_X = []
        for feature, n_node in zip(X, n_nodes):
            scaled_X.append(feature / n_node)
        self.enc.fit(scaled_X)

    def transform(self, X: List[np.array], n_nodes: List[int]):
        """Return transformed X using n_node in n_nodes."""
        std = self.enc.scale_
        mean = self.enc.mean_
        return [(feature - n_node*mean)/std for feature, n_node in zip(X, n_nodes)]

    def fit_transform(self, X: List[np.array], n_nodes: List[int]):
        self.fit(X, n_nodes)
        return self.transform(X, n_nodes)

    def inverse_transform(self, X_tr: List[np.array], n_nodes: List[int]):
        """Scale back the data to the original representation, using n_node in n_nodes."""
        std = self.enc.scale_
        mean = self.enc.mean_
        return [feature_tr*std + n_node*mean for feature_tr, n_node in zip(X_tr, n_nodes)]


TRANSFORMS = {
    'scalar_intrinsic': skp.StandardScaler,
    'one_hot': skp.OneHotEncoder,
    'element_list': ElementEncoder,
    'scalar_extrinsic': ExtrinsicScalarEncoder
}
TRANSFORMS_THAT_TAKE_N_NODE = (
    'scalar_extrinsic'
)


def collect_graph_values_dict(graphs: List[jraph.GraphsTuple]):
    """Returns a dict with graph properties in lists."""
    graph_values_dict = {
        'globals': {},
        'nodes': {},
        'edges': {},
        'n_nodes': [],
        'n_edges': []
    }
    
    for graph in graphs:
        # check that the graphs are not batched
        assert(len(graph.n_node)==1)
        graph_values_dict['n_nodes'].append(graph.n_node[0])
        graph_values_dict['n_edges'].append(graph.n_edge[0])
        # TODO: the next block repeats three times, could be shorter
        # TODO: maybe change to looping over property_dict, could be faster
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
    return graph_values_dict
        

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
        self.property_dict = dict(property_dict) # copy to loop over
        self.transforms_dict = dict(property_dict) # copy to re-populate values

    def save_parameters(self):
        raise NotImplementedError

    def fit(self, graphs: List[jraph.GraphsTuple]):
        graph_values_dict = collect_graph_values_dict(graphs)

        for feature, feature_dict in self.property_dict.items():
            # feature is e.g. 'globals'
            for prop_name, prop_transform in feature_dict.items():
                # prop_name is e.g. 'Egap', prop_transform is e.g. 'scalar'
                transform = TRANSFORMS[prop_transform]()
                self.transforms_dict[feature][prop_name] = transform
                if prop_transform in TRANSFORMS_THAT_TAKE_N_NODE:
                    transform.fit(
                        X=graph_values_dict[feature][prop_name],
                        n_nodes=graph_values_dict['n_nodes'])
                else:
                    transform.fit(X=graph_values_dict[feature][prop_name])

        return self.transforms_dict

    def transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError

    def fit_transform(self, graphs: List[jraph.GraphsTuple]):
        self.fit(graph)
        return self.transform(graphs)

    def inverse_transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError
