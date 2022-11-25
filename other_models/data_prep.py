"""Data preparation for ElemNet.

Here we fed an ASE atomic objects and we need to return atomic information
about the elements in the ASE atoms object."""

import pandas as pd


class DataPrep():
    """Class for data preparation to grab elemental data."""
    def __init__(self, ase_db_path: str, functional: str = 'pbe'):
        """Constructor for DataPrep class to initialize csv paths.
        
        Args:
            ase_db_path: path to ASE database.
            functional: What functional was used in the simulation for the
                materials in the ase db path.
        """
        if functional == 'pbe':
            self.elemental_features_csv = (
                'other_models/elemnet/really_tight_full_cut20_revpbe.csv')
        else:
            raise ValueError('functional can only be PBE currently.')

        self.ase_db_path = ase_db_path
        self.pbe_features_df = pd.read_csv(self.elemental_features_csv)

    def get_elemental_data_of_atom(self, atomic_number):
        ea_half = float(self.pbe_features_df[
            self.pbe_features_df['Atomic number'] == 1]['EA_half'])
        return ea_half




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