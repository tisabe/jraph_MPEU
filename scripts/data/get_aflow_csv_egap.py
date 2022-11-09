"""Pulling Data from AFLOW. Taking iterative slices out of the database."""

import json
from urllib.request import urlopen
import urllib
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'dft_type(*),ldau_type(*),energy_cutoff(*),'
    'energy_atom(*),density(*),'#volume_cell(*),'
    'geometry,positions_fractional,compound' # geometry parameters needed for unit cell
    )
DIRECTIVES = '$paging(0)'
summons = MATCHBOOK+","+DIRECTIVES

df_all = pandas.DataFrame({})

ranges = ['Egap(*),Egap_type(metal),', 'Egap(*),Egap_type(!metal),']

for val_range in ranges:
    URL = SERVER+API+val_range+summons
    response = urlopen(URL)
    print("Response code:" + str(response.code))
    data = response.read()
    data = json.loads(data)

    df = pandas.DataFrame(data)
    n_rows = df.shape[0]
    print('Num rows: ', n_rows)
    df_all = df_all.append(df, ignore_index=True)

duplicates = pandas.concat(g for _, g in df_all.groupby('auid') if len(g) > 1)
print(duplicates[['compound', 'auid']])
# remove duplicates by auid
df_all = df_all.drop_duplicates(subset='auid')
print(df_all.keys())
print(df_all.head())
print(df_all.describe())
if input("Save the dataset? [y/n]") == "y":
    df_all.to_csv(
        (input('Type directory and filename as "dir/filename.csv": ')))
