import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from datahandler import dict_to_ase
from ase.data import atomic_numbers
from ast import literal_eval
from utils import *

#folder_res = 'results_test/res_500ep_avg_5lr_binaries_enthalpy/'
folder_res = 'results_test/'
#folder_res = 'results_norm/res_500ep_256c_binaries_enthalpy'
folder_data = 'aflow/'
file_data = 'aflow_binary_enthalpy_atom.csv'
file_graphs = 'graphs_enthalpy_cutoff4A.csv'

df_data = pandas.read_csv(folder_data+file_data)
df_graphs = pandas.read_csv(folder_data+file_graphs)

df_test_pre = pandas.read_csv(folder_res+'test_pre.csv')
df_train_pre = pandas.read_csv(folder_res+'train_pre.csv')

df_test_post = pandas.read_csv(folder_res+'test_post.csv')
df_train_post = pandas.read_csv(folder_res+'train_post.csv')

#print(df_data.head())
#print(df_graphs.head())

labels = df_test_post['x']
preds = df_test_post['y']
auids = df_test_post['auid'].to_numpy()

errors = np.abs(labels - preds).to_numpy()
print(errors)

k = 10
idx = np.argpartition(errors, -k)[-k:]
errors_topk = errors[idx]
auids_topk = auids[idx]
print(idx)
print(errors_topk)
print(auids_topk)

for i in range(k):
    auid = auids_topk[i]
    row = df_data[df_data['auid']==auid] # get the row from original data with auid
    print(row[['compound', 'enthalpy_atom', 'kpoints']])
    print(idx[i], df_test_post['x'][idx[i]], df_test_post['y'][idx[i]])
'''
fig, ax = plt.subplots()
marker_size = 2
ax.scatter(labels, errors, s=marker_size)
ax.set_yscale('log')
ax.set_xlabel('label')
ax.set_ylabel('error')
plt.show()
'''
df = df_test_post
# look at errors depending on atomic numbers
counts = np.zeros(100)
errors_summed = np.zeros(100)
for index, row in df.iterrows():
    label = row['x']
    pred = row['y']
    auid = row['auid']
    row_raw_data = df_data[df_data['auid']==auid]
    symbols = row_raw_data['species']
    symbols = str(symbols.iloc[0])
    symbols = literal_eval(symbols)

    for sym in symbols:
        num = atomic_numbers[sym]
        #print(num)
        counts[num] += 1
        errors_summed[num] += np.abs(label - pred)
print(counts)
print(errors_summed)
errors_mean = errors_summed / counts
print(errors_mean)

fig, (ax1, ax2) = plt.subplots(2, 1)
#ax1.scatter(counts, errors_mean)
ax1.scatter(np.arange(0,100), errors_mean)
ax1.set_xlabel('atomic number')
ax1.set_ylabel('MAE')
ax2.bar(np.arange(0,100),counts)
ax2.set_yscale('log')
ax2.set_xlabel('atomic number')
ax2.set_ylabel('count')
plt.show()











