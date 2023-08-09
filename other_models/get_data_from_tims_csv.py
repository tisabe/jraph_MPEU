import pandas as pd
import time
import data_prep as dp


home_dir = '/home/bepdansp/jraph_MPEU/other_models'
enlarged_atomic_features_data_csv_path = (
    home_dir + '/elem_rf/enlarged_elemental_data.csv')
# Get the AFLOW dataset and read the CSV into a pandas dataframe.
# csv_data_source = '/home/bepdansp/elemnet/aflow_bandgap_greater_than_zero_data_july_27_2023.csv'
csv_data_source = home_dir + '/band_gap_data/result_best_egap_model_tim.csv'
feature_engineered_csv = home_dir + '/band_gap_data/result_best_egap_model_tim_plus_features.csv'

# Read in the csv data source
# tim_df = pd.read_csv(csv_data_source, index_col=0)
# input_list = tim_df

# columns = [
#     'spacegroup_relax', 'energy_cutoff', 'density',
#     'cutoff_type', 'cutoff_val', 'n_edge']
    

    #split,numbers,formula,prediction]



# atomic_data_df = pd.read_csv(enlarged_atomic_features_data_csv_path, index_col=None)
# aflow_csv_data_source = '/band_gap_data/classification_data.csv'
# Read the data into a dataframe:
# aflow_df = pd.read_csv(home_dir+aflow_csv_data_source, index_col=0)
aflow_df = pd.read_csv(csv_data_source, index_col=0)

# # Print out some data about the CSV.
# aflow_df.describe()
print('AFLOW columns:')
print(aflow_df.columns)

# Ok now we want to create a new dataframe with the atomic features added in.
# Import our data prep module.

# start = time.time()
# print("hello")
# Read the data into a dataframe.

data_prep_obj = dp.DataPrep(
    ase_db_path='/None',
    functional='pbe',
    elemental_features_csv=enlarged_atomic_features_data_csv_path)

features_list = [
       'EA_half', 'IP_half', 'EA_delta', 'IP_delta', 'HOMO', 'LUMO',
       'rs', 's index', 'rp', 'p index', 'rd', 'd index', 'rf', 'f index',
       'atomic_number', 'atomic_weight', 'mendeleev_number', 'melting_point',
       'covalent_radius_cordero', 'en_allen', 'en_ghosh',
       'en_pauling', 'period', 'atomic_volume', 'n_valence', 'HOMO_LUMO_diff']

compound_name_list = aflow_df['formula']

feature_engineered_data_df_with_compound = data_prep_obj.get_features_df(
    compound_name_list=compound_name_list, features_list=features_list)

feature_engineered_data_df_with_compound.to_csv(feature_engineered_csv, index=False)