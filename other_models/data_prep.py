"""Data preparation for ElemNet.

Here we fed an ASE atomic objects and we need to return atomic information
about the elements in the ASE atoms object."""
import ase
import collections
import numpy as np
import pandas as pd
import logging
import multiprocessing
from functools import partial

class DataPrep():
    """Class for data preparation to grab elemental data."""
    def __init__(
            self, ase_db_path: str, functional: str = 'pbe',
            elemental_features_csv=None):
        """Constructor for DataPrep class to initialize csv paths.
        
        Args:
            ase_db_path: path to ASE database.
            functional: What functional was used in the simulation for the
                materials in the ase db path.
        """
        self.elemental_features_csv = elemental_features_csv
        #
        #         # 'other_models/really_tight_full_cut20_revpbe.csv')
        #         'really_tight_full_cut20_revpbe.csv')
        # else:
        #     raise ValueError('functional can only be PBE currently.')

        self.ase_db_path = ase_db_path
        self.pbe_features_df = pd.read_csv(self.elemental_features_csv)

    def get_csv_val(self, atomic_number, feature):
        """Get elemental data from elemental data CSV for a given element."""
        csv_val = float(self.pbe_features_df[
            self.pbe_features_df['Atomic number'] == atomic_number][feature])
        return csv_val


    def get_feature_bar_val(
            self, atomic_number_list, fraction_list, feature):
        """"Get the feature average result for a given compound."""
        feature_val_list = np.zeros(len(atomic_number_list))
        # Loop over every atomic number in our list and get the relevant
        # value from the CSV for a given feature. We specify in our CSV
        # the atomic number and the feature and this gives us a spefic cell
        # value of our CSV. The atomic numbers are the rows and the features
        # are the columns.
        for i, atomic_number in enumerate(atomic_number_list):
            feature_val_list[i] = self.get_csv_val(
                atomic_number, feature)
        # The dot product gives us our summation equation from the paper.
        # We are getting the sum of f_i*x_i over i where f_i is the feature val
        # and x_i is the fraction value.
        return np.dot(feature_val_list, fraction_list), feature_val_list

    def get_feature_hat_val(
            self, fraction_list, feature_val_list, f_bar):
        """"Get the feature average deviation result for a given compound.
        
        This function should be called after get_feature_bar_val has been called
        since we want to feed in the feature average value (f_bar) and the
        feature_val_list.
        """
        feature_hat = 0
        for i in range(len(fraction_list)):
            feature_hat += fraction_list[i]*np.abs(feature_val_list[i] - f_bar)
        return feature_hat

    def get_atomic_number_and_fraction_list(self, ase_atoms_obj):
        """Atoms object input."""
        atomic_numbers_list = list(ase_atoms_obj.numbers)
        counter = collections.Counter(atomic_numbers_list)
        atomic_number_list = list(counter.keys())
        fraction_list = [i/len(atomic_numbers_list) for i in counter.values()]
        return atomic_number_list, fraction_list

    # TODO: test this function, do this first.
    def get_features_row(self, compound_name, features_list):
        """Get the row of CSV data for a features list for a given compound."""
        # We need to get f_hat and f_bar for each feature in our feature list.
        # This is why we initialize the row to be twice as big as the number
        # of features.
        row_dict = {}
        ase_atoms_obj = ase.Atoms(compound_name)
        atomic_number_list, fraction_list = self.get_atomic_number_and_fraction_list(
            ase_atoms_obj)
        # We are given a specific compound, this narrows us down to a row in
        # our elemental data CSV. Now we want to iterate over the features
        # we are interested in. For a given compound, feature (row, column)
        # we get a specific value from our CSV list. We need to transform
        # these values into f_bar and f_hat values (average, and average
        # deviation) before saving them. We return in this function a row
        # of data which is f_bar and f_hat values for a given compound.
        for feature in features_list:
            row_dict[feature + '_bar'], feature_val_list = self.get_feature_bar_val(
                atomic_number_list, fraction_list, feature)
            row_dict[feature + '_hat'] = self.get_feature_hat_val(
                fraction_list=fraction_list,
                feature_val_list=feature_val_list,
                f_bar=row_dict[feature + '_bar'])
        row_dict['compound_name'] = compound_name
        return row_dict

    def get_features_df(self, compound_name_list, features_list):
        # Initialize an empty dataframe with the desired feature columns as columns
        features_df = pd.DataFrame({})
        features_list = features_list
        i = 0
        for compound_name in compound_name_list:
            row_dict = self.get_features_row(compound_name, features_list)
            # Append the row of new elemental averaged data to the dataframe.
            features_df = pd.concat([features_df, pd.DataFrame([row_dict])], ignore_index=True)
            i += 1
            if i % 1000 == 0:
                logging.warn(f'Computed {i} materials.')
        return features_df

    def get_features_df_parallelize(self, compound_name_list, features_list):
        #TODO(Salman): Write this method, parallilize the for loop above with the multiprocessing library
        # Write a test for the method if possible
        num_processes = multiprocessing.cpu_count()  # No. of available CPU cores
        with multiprocessing.Pool(processes=num_processes) as pool:

        # Create a partial function to pass additional arguments to get_features_row
         get_features_row_partial = partial(self.get_features_row, features_list=features_list)

        # Map the function to the list of compound names using multiple processes
         result_list = pool.map(get_features_row_partial, compound_name_list)
        # Concatenate the results into a single DataFrame
        features_df = pd.DataFrame(result_list)
        return features_df



    #     self.pbe_features_df = pd.read_csv(self.pbe_features_csv)
    #     self.lda_features_df = pd.read_csv(self.lda_features_csv)

    # def get_data_from_csv(self, atom_num, functional, column):
    #     """Get data from csv file for functional, atom num, column.

    #     atom_num: atom number by # of protons.
    #     functional (string): lda or pbe?
    #     column (string): what column of data are we looking at from the
    #         the fhi-aims monomers database are we looking at. ex. 'EA_half'.
    #         Check the csv to find the names of the columns available.
    #     """
    #     if functional == 'pbe':
    #         csv_val = float(self.pbe_features_df[
    #             self.pbe_features_df[
    #                 'Atomic number'] == atom_num][column])
    #     elif functional == 'pw-lda':
    #         csv_val = float(self.lda_features_df[
    #             self.lda_features_df[
    #                 'Atomic number'] == atom_num][column])
    #     else:
    #         raise BadFunctional(functional)

    #     return csv_val
