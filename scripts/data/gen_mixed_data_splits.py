import json
from collections import defaultdict

from absl import app
from absl import flags
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


FLAGS = flags.FLAGS
flags.DEFINE_string('out_path', 'databases/data_summary.csv', 'Where csv is saved')


def main(_):
    df = pd.read_csv(FLAGS.out_path)
    df['struct_hash'] = df['formula'].str.cat(df['spacegroup_pmg'].astype(str), sep='_')
    df['id+database'] = df['database'].str.cat(df['asedb_id'].astype(str), sep=',')
    df = df.dropna(subset=['struct_hash', 'asedb_id'])
    df['ef'] = df[['delta_e', 'enthalpy_formation_atom']].sum(axis=1)
    df['gap'] = df[['band_gap', 'Egap']].sum(axis=1)

    # filter data, limit range
    df = df[(df['dft_type']=="['PAW_PBE']") | (df['dft_type'].isna())]
    df = df[(df['ef']>-10) & (df['ef']<70)]

    # separate into aflow and mp data
    df_aflow = df[df['auid'].notna()]
    df_mp = df[df['auid'].isna()]

    n_total = df['struct_hash'].nunique()
    n_aflow = df_aflow['struct_hash'].nunique()
    n_mp = df_mp['struct_hash'].nunique()
    print("# unique structures in total: ", n_total)
    print("# unique structures in AFLOW: ", n_aflow)
    print("# unique structures in MP: ", n_mp)
    print("# structures overlapping: ", n_aflow+n_mp-n_total)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x='ef', ax=ax, hue='database')
    fig.savefig('databases/hist.png', bbox_inches='tight', dpi=600)
    plt.show()

    structs_all = set(df['struct_hash'].to_list())
    structs_aflow = set(df_aflow['struct_hash'].to_list())
    structs_mp = set(df_mp['struct_hash'].to_list())
    structs_shared = structs_aflow & structs_mp
    df_shared = df[df['struct_hash'].isin(structs_shared)]

    fig, ax = plt.subplots()
    sns.histplot(data=df_shared, x='ef', ax=ax, hue='database')
    fig.savefig('databases/hist_shared.png', bbox_inches='tight', dpi=600)
    plt.show()

    kfold = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    groups = df['struct_hash']
    for fold, (train_index, test_index) in enumerate(
            kfold.split(X=df['id+database'], groups=groups)):
        print(f"Fold {fold}:")
        print(f"#train: {len(train_index)}, #test: {len(test_index)}")
        if fold != 0: # only use the zeroth fold
            continue
        train_index, val_index = train_test_split(
            train_index,
            test_size=0.1,
            random_state=1)
        print(type(train_index), type(train_index[0]))
        train_set = df['id+database'].iloc[train_index]
        val_set = df['id+database'].iloc[val_index]
        test_set = df['id+database'].iloc[test_index]
        print(f"#val: {len(val_set)}")

        aflow_split = {}
        mp_split = {}

        for id_and_database in train_set:
            database, asedb_id = id_and_database.split(',')
            asedb_id = int(asedb_id)
            match database:
                case "databases/aflow/graphs_12knn_vec.db":
                    aflow_split[asedb_id] = "train"
                case "databases/matproj/mp2018_graphs.db":
                    mp_split[asedb_id] = "train"

        for id_and_database in val_set:
            database, asedb_id = id_and_database.split(',')
            asedb_id = int(asedb_id)
            match database:
                case "databases/aflow/graphs_12knn_vec.db":
                    aflow_split[asedb_id] = "validation"
                case "databases/matproj/mp2018_graphs.db":
                    mp_split[asedb_id] = "validation"

        for id_and_database in test_set:
            database, asedb_id = id_and_database.split(',')
            asedb_id = int(asedb_id)
            match database:
                case "databases/aflow/graphs_12knn_vec.db":
                    aflow_split[asedb_id] = "test"
                case "databases/matproj/mp2018_graphs.db":
                    mp_split[asedb_id] = "test"

        with open("databases/aflow_split.json", 'w', encoding="utf-8") as splits_file:
            json.dump(aflow_split, splits_file, indent=4, separators=(',', ': '))
        with open("databases/mp_split.json", 'w', encoding="utf-8") as splits_file:
            json.dump(mp_split, splits_file, indent=4, separators=(',', ': '))
        break


if __name__ == "__main__":
    app.run(main)
