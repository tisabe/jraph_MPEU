from xmlrpc.client import Boolean
import ml_collections
from typing import Dict, Iterable, Sequence
import jraph
import sklearn.model_selection
from random import shuffle

from utils import *

class DataReader:
    def __init__(self, data: Sequence[jraph.GraphsTuple], 
    batch_size: int, repeat: Boolean):
        self.data = data
        self.batch_size = batch_size
        self.repeat = repeat
        self.total_num_graphs = len(data)
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
                    shuffle(self.data)
                # This will reset the index to 0 if we are at the end of the dataset.
                idx = idx % self.total_num_graphs
            graph = self.data[idx]
            idx += 1
            yield graph



def get_datasets(config: ml_collections.ConfigDict
) -> Dict[str, Iterable[jraph.GraphsTuple]]:
    '''Return a dict with a dataset for each split (training, validation, testing).

    Each dataset is an iterator that yields batches of graphs.
    The training dataset will reshuffle each time the end of the list has been 
    reached, while the validation and test sets are only iterated over once.

    The graphs have their regression label as a global feature attached.
    '''
    graphs_list, labels_list, _ = get_data_df_csv(config.data_file)
    graphs_list = add_labels_to_graphs(graphs_list, labels_list)

    # split the graphs into three splits using the fractions defined in config
    train_set, val_and_test_set = sklearn.model_selection.train_test_split(
        graphs_list,
        test_size=config.test_frac+config.val_frac,
        random_state=0
    )
    val_set, test_set = sklearn.model_selection.train_test_split(
        val_and_test_set,
        test_size=config.test_frac/(config.test_frac+config.val_frac)
    )

    # define iterators and generators
    reader_train = DataReader(data=train_set, 
        batch_size=config.batch_size, repeat=True)
    reader_val = DataReader(data=val_set, 
        batch_size=config.batch_size, repeat=False)
    reader_test = DataReader(data=test_set, 
        batch_size=config.batch_size, repeat=False)

    dataset = {'train': reader_train, 
        'validation': reader_val, 
        'test': reader_test}

    return dataset