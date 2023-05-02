"""Pulling Data from AFLOW."""

import json, sys, os
from urllib.request import urlopen
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'Egap(*),Egap_type(metal),'
    #'Egap(*),Egap_type(!metal),'
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
print(type(response))

df = pandas.DataFrame(response)
#print(df.head())
print(df.describe())

# remove duplicates by auid
#df = df.drop_duplicates(subset='auid')
df = df.drop_duplicates(subset='aurl')
#print(df.head())
print(df.describe())

if input("Save the dataset? [y/n]") == "y":
    df.to_csv(
        (input('Type directory and filename as "dir/filename.csv": ')))
