"""Pulling Data from AFLOW."""

import json, sys, os
from urllib.request import urlopen
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'enthalpy_atom(*),enthalpy_formation_atom(*),'
    'Egap_type(!metal),Egap(*),agl_heat_capacity_Cp_300K,'
    'geometry_orig,positions_cartesian,compound' # geometry parameters needed for unit cell
    )
DIRECTIVES = '$paging(0)'
summons = MATCHBOOK+","+DIRECTIVES
print(summons)
print("URL:", SERVER+API+summons)
response = json.loads(urlopen(SERVER+API+summons).read())
print(type(response))

df = pandas.DataFrame(response)
print(df.head())
print(df.describe())
if input("Save the dataset? [y/n]") == "y":
    df.to_csv(
        ('aflow/aflow_binary_enthalpy_atom.csv'))