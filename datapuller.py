"""Pulling Data from AFLOW."""

import json, sys, os
from urllib.request import urlopen
import pandas

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'nspecies(2),Egap(0*,*1000000),'
    'energy_cell,Bravais_lattice_orig,crystal_class_orig,crystal_family_orig,'
    'dft_type,geometry_orig,kpoints,lattice_system_orig,natoms,positions_cartesian,'
    'prototype,spacegroup_orig,species,volume_atom')
    # ,geometry_orig,icsd_number,kpoints,lattice_system_orig,natoms,
    # 'positions_cartesian,prototype,spacegroup_orig,species,volume_atom')
DIRECTIVES = '$paging(0)'
summons = MATCHBOOK+","+DIRECTIVES
print(summons)

response = json.loads(urlopen(SERVER+API+summons).read())
print(type(response))

df = pandas.DataFrame(response)
print(df.head())
print(df.describe())
df.to_csv(
    ('aflow/aflow_binary_egap_above_zero_below_ten_mill.csv'))