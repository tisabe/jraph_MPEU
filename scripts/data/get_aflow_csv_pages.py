"""Pulling Data from AFLOW."""

import json
from urllib.request import urlopen
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/?"
MATCHBOOK = (
    'Egap(*),catalog(ICSD),'#'enthalpy_formation_atom(*),'#Egap(*),Egap_type(*),'
    #'dft_type(*),ldau_type(*),species_pp_ZVAL(*),energy_cutoff(*),'
    #'energy_atom(*),density(*),'#volume_cell(*),'
    # 'geometry,positions_fractional,compound' # geometry parameters needed for unit cell
    )
print("URL:", SERVER+API+MATCHBOOK)

df_all = pandas.DataFrame({})

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
        df_all = df_all.append(df, ignore_index=True)
        i += 1

print(df_all.head())
print(df_all.describe())
if input("Save the dataset? [y/n]") == "y":
    df.to_csv(
        (input('Type directory and filename as "dir/filename.csv": ')))
