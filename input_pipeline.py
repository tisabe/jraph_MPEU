from xmlrpc.client import Boolean
import ml_collections
from absl import logging

from typing import Dict, Iterable, Sequence, Tuple
import jraph
import sklearn.model_selection
import random
import numpy as np

from utils import *

class DataReader:
    def __init__(self, data: Sequence[jraph.GraphsTuple], 
    batch_size: int, repeat: Boolean, key: jax.random.PRNGKey):
        self.data = data
        self.batch_size = batch_size
        self.repeat = repeat
        self.total_num_graphs = len(data)
        self.rng = key
        self._generator = self._make_generator()
        
        self.budget = estimate_padding_budget_for_batch_size(data, batch_size,
            num_estimation_graphs=100)

        # this makes this thing complicated. From outside of DataReader
        # we interface with this batch generator, but this batch_generator
        # needs an iterator itself which is also defined in this class
        self.batch_generator = jraph.dynamically_batch(
            self._generator, 
            self.budget.n_node,
            self.budget.n_edge,
            self.budget.n_graph)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.batch_generator)

    def _reset(self):
        # reset the generator object
        # does not work 
        self._generator = self._make_generator()
        self.batch_generator = jraph.dynamically_batch(
            self._generator, 
            self.budget.n_node,
            self.budget.n_edge,
            self.budget.n_graph)

    def _make_generator(self):
        random.seed(a=0)
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if not self.repeat:
                if idx == self.total_num_graphs:
                    idx = 0
                    '''here's the problem. At the end of iterating through this dataset,
                    this return is reached and when trying to iterate with this generator
                    again, only one None is returned'''
                    return
            else:
                if idx == self.total_num_graphs:
                    self.rng, data_rng = jax.random.split(self.rng)
                    random.shuffle(self.data)
                # This will reset the index to 0 if we are at the end of the dataset.
                idx = idx % self.total_num_graphs
            graph = self.data[idx]
            idx += 1
            yield graph


def get_labels_atomization(graphs, labels, label_str):
    '''Wrapper function for get_atomization_energies_QM9,
    to make it compatible with non-QM9 datasets.'''
    return get_atomization_energies_QM9(graphs, labels, label_str)

def get_datasets(config: ml_collections.ConfigDict, key
) -> Tuple[Dict[str, Iterable[jraph.GraphsTuple]], Dict[str, Sequence[jraph.GraphsTuple]], float, float]:
    '''Return a dict with a dataset for each split (training, validation, testing),
    in normalized and in raw form/labels. Also return the mean and standard deviation.

    Each dataset is an iterator that yields batches of graphs.
    The training dataset will reshuffle each time the end of the list has been 
    reached, while the validation and test sets are only iterated over once.

    The raw dataset is just a dict with lists of graphs.

    The graphs have their regression label as a global feature attached.
    '''
    # data will be split into normaized data for regression and raw data for analyzing later
    graphs_list, labels_list, _ = get_data_df_csv(config.data_file, label_str=config.label_str)
    labels_raw = labels_list
    labels_list = get_labels_atomization(graphs_list, labels_list, config.label_str)
    labels_list, mean, std = normalize_targets_config(graphs_list, labels_list, config)
    logging.info(f'Mean: {mean}, Std: {std}')
    graphs_list = add_labels_to_graphs(graphs_list, labels_list)
    graphs_raw = add_labels_to_graphs(graphs_list, labels_raw)
    

    # split the graphs into three splits using the fractions defined in config
    train_set, val_and_test_set, train_raw, val_and_test_raw = sklearn.model_selection.train_test_split(
        graphs_list, graphs_raw,
        test_size=config.test_frac+config.val_frac,
        random_state=0
    )
    val_set, test_set, val_raw, test_raw = sklearn.model_selection.train_test_split(
        val_and_test_set, val_and_test_raw,
        test_size=config.test_frac/(config.test_frac+config.val_frac),
        random_state=1
    )
    
    # define iterators and generators
    reader_train = DataReader(data=train_set, 
        batch_size=config.batch_size, repeat=True, key=key)
    reader_val = DataReader(data=val_set, 
        batch_size=config.batch_size, repeat=False, key=key)
    reader_test = DataReader(data=test_set, 
        batch_size=config.batch_size, repeat=False, key=key)

    dataset = {'train': reader_train, 
        'validation': reader_val, 
        'test': reader_test}

    dataset_raw = {'train': train_raw, 
        'validation': val_raw, 
        'test': test_raw}

    return dataset, dataset_raw, mean, std