"""Pulling Data from AFLOW."""

import json, sys, os
from urllib.request import urlopen
import pandas

# Save location:
csv_filename = 'other_models/band_gap_data/classification_data.csv'

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'Egap(0.001*),'
    'dft_type(*),ldau_type(*),energy_cutoff(*),'
    'energy_atom(*),density(*),volume_cell(*),'
    'catalog(ICSD),nspecies(1*),compound' # geometry parameters needed for unit cell
    )
DIRECTIVES = '$paging(0)'
summons = MATCHBOOK+","+DIRECTIVES
print(summons)
URL = SERVER+API+summons
print("URL:", URL)
response = json.loads(urlopen(URL).read().decode('utf-8'))
print(type(response))

df = pandas.DataFrame(response)
print(df.head())
print(df.describe())
if input("Save the dataset? [y/n]") == "y":
    df.to_csv(csv_filename)
        # (input('Type directory and filename as "dir/filename.csv": ')))
