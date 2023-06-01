import pandas as pd
import timeit
home_dir = '/home/dts/Documents/hu/jraph_MPEU/other_models'
enlarged_atomic_features_data_csv_path = (
    home_dir + '/elem_rf/enlarged_elemental_data.csv')
atomic_data_df = pd.read_csv(enlarged_atomic_features_data_csv_path, index_col=None)
aflow_csv_data_source = '/band_gap_data/classification_data.csv'
# Read the data into a dataframe:
aflow_df = pd.read_csv(home_dir+aflow_csv_data_source, index_col=0)

# Ok now we want to create a new dataframe with the atomic features added in.
# Import our data prep module.
import data_prep as dp
import time

start = time.time()
print("hello")

data_prep_obj = dp.DataPrep(
    ase_db_path='/None',
    functional='pbe',
    elemental_features_csv=enlarged_atomic_features_data_csv_path)

features_list = [
       'EA_half']

# 'IP_half', 'EA_delta', 'IP_delta', 'HOMO', 'LUMO',
#        'rs', 's index', 'rp', 'p index', 'rd', 'd index', 'rf', 'f index',
#        'atomic_number', 'atomic_weight', 'mendeleev_number', 'melting_point',
#        'covalent_radius_cordero', 'en_allen', 'en_ghosh',
#        'en_pauling', 'atomic_volume', 'n_valence', 'HOMO_LUMO_diff']

compound_name_list = aflow_df['compound']

feature_engineered_data_df_with_compound = data_prep_obj.get_features_df(
    compound_name_list=compound_name_list, features_list=features_list)
end = time.time()
print(end - start)