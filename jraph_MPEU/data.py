"""Module for different kinds of dataset classes. This should gradually replace
functionality from input_pipeline, mainly 'get_datasets'.

This is inspired mainly by API in pytorch-geometric.data"""

import os
from typing import (
    Optional,
    Union,
    List
)
from multiprocessing import Pool

import ase.db
import numpy as np
from sklearn.model_selection import GroupKFold

from jraph_MPEU.input_pipeline import(
    ase_row_to_jraph
)


class AsedbDataset():
    """Class for creating a dataset from an existing ASE database.

    Args:
        db_dir (str): directory of source ASE database.
        workdir (str, optional): directory where information about dataset,
            like split ids, normalization, etc., will be saved.
    """
    def __init__(
        self,
        db_dir: str,
        target: str,
        selection: str = None,
        limit: int = None,
        workdir: Optional[str] = None,
        globals_strs: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self._db_dir = db_dir
        self._target = target
        self._selection = selection
        self._limit = limit
        self._workdir = workdir
        self._globals_strs = globals_strs

        self._load()
        self._process()
        self._split()
        self._normalize()

    def _load(self):
        if not os.path.exists(self._db_dir):
            raise RuntimeError(f"Could not find ase.db at {self._db_dir}")
        with ase.db.connect(self._db_dir) as db:
            rows = db.select(selection=self._selection, limit=self._limit)
        self._data_raw = list(rows)

    def _get_row_tuple(self, row):
        """Turn an ase.db.row.AtomsRow into a tuple consisting of input, output
        and meta data."""
        graph = ase_row_to_jraph(row, self._globals_strs)
        target = row[self._target]
        graph_out = graph._replace(globals=np.asarray(target.reshape((1, -1))))
        meta_data = row.key_value_pairs
        meta_data['formula'] = row['formula']
        meta_data['asedb_id'] = row.id
        return graph, graph_out, meta_data

    def _process(self, n_workers=None):
        """Apply self._get_row_tuple to all rows in self._data_raw. Store
        result in self._data."""
        with Pool(processes=n_workers) as pool:
            self._data = pool.map(self._get_row_tuple, self._data_raw)

    def _split(self, n_folds=10, i_fold=0):
        self._kfold_test = GroupKFold(n_splits=n_folds)
        groups = None
        for i, (train_val_index, test_index) in enumerate(
                self._kfold_test.split(X=self._data, y=None, groups=groups)):
            if i==i_fold:
                self._train_val_index = train_val_index
                self._test_index = test_index

    def _normalize(self):
        raise NotImplementedError

    def _download(self):
        """Download the dataset to disk, of corresponding file was not found
        in self.root."""
        raise NotImplementedError
