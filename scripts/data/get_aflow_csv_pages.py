"""Pulling Data from AFLOW."""

import json
from urllib.request import urlopen
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    'enthalpy_formation_atom(*),'#Egap(*),Egap_type(*),'
    #'dft_type(*),ldau_type(*),species_pp_ZVAL(*),energy_cutoff(*),'
    #'energy_atom(*),density(*),'#volume_cell(*),'
    'geometry_orig,positions_cartesian,compound' # geometry parameters needed for unit cell
    )
print("URL:", SERVER+API+MATCHBOOK)

i = 1
while i < 50:
    print("Page: " + str(i))
    DIRECTIVES = f'$paging({int(i)},100000)'
    summons = MATCHBOOK+","+DIRECTIVES

    response = urlopen(SERVER+API+summons)
    print("Response code:" + str(response.code))
    data = response.read()
    data = json.loads(data)
    print(type(data))

    df = pandas.DataFrame(data)
    if df.empty:
        break
    else:
        i += 1

"""if input("Save the dataset? [y/n]") == "y":
    df.to_csv(
        (input('Type directory and filename as "dir/filename.csv": ')))"""
