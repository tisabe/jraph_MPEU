import argparse

import numpy as np
import ase.db
import pandas as pd

import matplotlib.pyplot as plt



def main(args):
    file = args.file
    limit = args.limit
    key = args.key

    num_senders = []
    num_receivers = []
    num_edges = []
    edges_all = np.array([]) # collect all edge distances for histogram
    key_val_list = [] # list of key-value-pairs

    with ase.db.connect(file) as asedb:
        for i, row in enumerate(asedb.select(limit=limit)):
            if i%10000 == 0:
                print(f'Reading step {i}')
            key_value_pairs = row.key_value_pairs
            key_val_list.append(key_value_pairs)
            #data = row.data
            #senders = data['senders']
            #receivers = data['receivers']
            #edges = data['edges']
            #edges_all = np.concatenate((edges_all, np.array(edges)))
    
    
    key_df = pd.DataFrame(key_val_list)
    print(key_df.head())
    print(key_df.describe())
    key_y = key_df.get(key).to_numpy()
    plt.hist(key_y, bins=1000, log=True)
    plt.show()
    plt.savefig('scripts/plots/key_hist.png')
    
    # print(edges_all.shape)
    # plt.hist(edges_all, bins=1000, log=True)
    # plt.show()
    # plt.savefig('scripts/plots/edges_hist.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show data analysis plots.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='QM9/qm9_graphs.db',
                        help='data directory name')
    parser.add_argument('-limit', type=int, dest='limit', default=None,
                        help='limit number of database entries to be selected')
    parser.add_argument('-key', type=str, dest='key', default=None,
                        help='key name to plot')
    args = parser.parse_args()
    main(args)