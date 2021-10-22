"""Pulling Data from AFLOW."""

import json, sys, os
from urllib.request import urlopen
import pandas

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    #'nspecies(2),Egap(0*,*1000000),'
    'nspecies(2),enthalpy_atom,enthalpy_formation_atom,'
    'geometry_orig,positions_cartesian,compound,' # geometry parameters needed for unit cell
    'dft_type,kpoints,lattice_system_orig,natoms,'
    'prototype,species,volume_atom')
    # ,geometry_orig,icsd_number,kpoints,lattice_system_orig,natoms,
    # 'positions_cartesian,prototype,spacegroup_orig,species,volume_atom')
DIRECTIVES = '$paging(1,25000)'
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