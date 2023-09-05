"""Pulling Data from AFLOW. Taking iterative slices out of the database."""

import json
from urllib.request import urlopen
import urllib
import pandas

# Find all properties in http://aflow.org/API/aflux/?schema

SERVER = "http://aflow.org"
API = "/API/aflux/v1.0/?"
MATCHBOOK = (
    #'enthalpy_formation_atom(*),'
    #'Egap(*),Egap_type(*),'
    'dft_type(*),ldau_type(*),energy_cutoff(*),'
    'energy_atom(*),density(*),PV_atom(*),'
    'geometry,geometry_orig,positions_fractional,compound' # geometry parameters needed for unit cell
    )
print("URL:", SERVER+API+MATCHBOOK)
DIRECTIVES = '$paging(0)'

df_all = pandas.DataFrame({})

i = 1
max_iter = 100  # maximum number of iterations
last_n_rows = 0  # store the last number of rows in pulled dataframe
lower = -20.0  # lower bound for search
delta = 16  # search window
while i < max_iter:
    if delta < 1e-3:
        print(f'Halving failed, range too small at delta={delta}')
        break
    elif delta > 1e5:
        print(f'Range exceeded maximum at delta={delta}')
    summons = MATCHBOOK+","+DIRECTIVES
    # transform upper and lower bounds into correct format
    lower_str = '{:10.4f}'.format(lower)
    upper_str = '{:10.4f}'.format(lower + delta)
    val_range = f'enthalpy_formation_atom({lower_str}*,*{upper_str})'
    val_range = val_range.replace(' ', '')  # delete spaces
    print('trying '+val_range)
    try:
        URL = SERVER+API+val_range+','+summons
        #print('URL: ', URL)
        response = urlopen(URL)
        print("Response code:" + str(response.code))
        data = response.read()
        data = json.loads(data)

        df = pandas.DataFrame(data)
        n_rows = df.shape[0]
        print('Num rows: ', n_rows)
        if n_rows == 0:
            # increase lower bound by delta
            lower = lower + delta
            # increase window size
            delta = delta * 2
        elif last_n_rows > n_rows:
            df_all = pandas.concat([df_all, df], ignore_index=True)
            # increase lower bound by delta
            lower = lower + delta
            # increase window size
            delta = delta * 2
        else:
            df_all = pandas.concat([df_all, df], ignore_index=True)
            # increase lower bound by delta
            lower = lower + delta
        last_n_rows = n_rows

    except urllib.error.HTTPError:
        print(f'Pull failed at i={i}, http error')
        # pull failed, decrease value range
        delta = delta/2
    except json.decoder.JSONDecodeError:
        print(f'Pull failed at i={i}, json error')
        # pull failed, decrease value range
        delta = delta/2
    i += 1

# remove duplicates by auid
df_all = df_all.drop_duplicates(subset='auid')
print(df_all.keys())
print(df_all.head())
print(df_all.describe())
if input("Save the dataset? [y/n]") == "y":
    df_all.to_csv(
        (input('Type directory and filename as "dir/filename.csv": ')))
