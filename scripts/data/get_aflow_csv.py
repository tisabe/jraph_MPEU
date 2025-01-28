"""Pulling Data from AFLOW."""

import json
from pathlib import Path
from urllib.request import urlopen
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'enthalpy_formation_atom(*),'
    'Egap(*),Egap_type(*),'
    'dft_type(*),ldau_type(*),energy_cutoff(*),'
    'energy_atom(*),density(*),'#volume_cell(*),'
    'geometry,positions_fractional,compound' # geometry parameters needed for unit cell
    )
DIRECTIVES = '$paging(0)'
summons = MATCHBOOK+","+DIRECTIVES
print(summons)
URL = SERVER+API+summons
print("URL:", URL)
response = json.loads(urlopen(URL).read())

df = pandas.DataFrame(response)
print("Before removing aurl duplicates")
print(df.describe())

# remove duplicates by auid
#df = df.drop_duplicates(subset='auid')
df = df.drop_duplicates(subset='aurl')
#print(df.head())
print("After removing aurl duplicates")
print(df.describe())

if input("Save the dataset? [y/n]") == "y":
    if input("Use default path 'databases/aflow/default.csv'? [y/n]") == "y":
        output_file = 'default.csv'
        output_dir = Path('databases/aflow')
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / output_file)
    else:
        df.to_csv(input('Type existing directory and new filename as "dir/filename.csv": '))
