"""This module defines the classes and functions for normalizing and scaling of
input data. The API is meant to resemble functions in sklearn.preprocessing."""

from typing import List
import jraph


class GraphNormalizer:
    """Class to calculate and apply normalization to graph structured data."""
    def __init__(self, workdir=None):
        pass

    def fit(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError

    def fit_transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError

    def transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError

    def inverse_transform(self, graphs: List[jraph.GraphsTuple]):
        raise NotImplementedError