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
flags.DEFINE_string('task', 'ef', "Type of task, 'ef' for formation energies\
    'egap' for all electronic band gaps, 'egap_ins' for only non-zero bandgaps.")


def main(_):
    df = pd.read_csv(FLAGS.out_path)
    df['struct_hash'] = df['formula'].str.cat(df['spacegroup_pmg'].astype(str), sep='_')
    df['id+database'] = df['database'].str.cat(df['asedb_id'].astype(str), sep=',')
    df = df.dropna(subset=['struct_hash', 'asedb_id'])
    df['ef'] = df[['delta_e', 'enthalpy_formation_atom']].sum(axis=1)
    df['gap'] = df[['band_gap', 'Egap']].sum(axis=1)

    # filter data, limit range
    df = df[(df['dft_type']=="['PAW_PBE']") | (df['dft_type'].isna())]

    match FLAGS.task:
        case 'ef':
            label = 'ef'
            df = df[(df['ef']>-10) & (df['ef']<70)]
            x_label = r'E$_f$ (eV/atom)'
        case 'egap':
            label = 'gap'
            x_label = r'E$_g$ (eV)'
        case 'egap_ins':
            label = 'gap'
            df = df[df['gap'] > 0]
            x_label = r'E$_g$ (eV)'
        case _ :
            raise ValueError(f'Invalid task: {FLAGS.task}')


    print("Total value counts: ")
    print(df.value_counts(subset='struct_hash'))

    # separate into aflow and mp data
    df_aflow = df[df['auid'].notna()]
    df_mp = df[df['auid'].isna()]

    print("AFLOW value counts: ")
    print(df_aflow.value_counts(subset='struct_hash'))
    print("MP value counts: ")
    print(df_mp.value_counts(subset='struct_hash'))


    n_total = df['struct_hash'].nunique()
    n_aflow = df_aflow['struct_hash'].nunique()
    n_mp = df_mp['struct_hash'].nunique()
    print("# unique structures in total: ", n_total)
    print("# unique structures in AFLOW: ", n_aflow)
    print("# unique structures in MP: ", n_mp)
    print("# structures overlapping: ", n_aflow+n_mp-n_total)

    fontsize = 18
    ticksize = 16

    fig, ax = plt.subplots(2, 1, sharex=True)
    gfg = sns.histplot(data=df, x=label, ax=ax[0], hue='database', palette=['red', 'green'])
    ax[0].legend(labels=['Materials Project', 'AFLOW'])
    ax[0].set_xlabel('Number of atoms in unit cell', fontsize=fontsize)
    ax[0].set_ylabel('Count', fontsize=fontsize)
    ax[0].tick_params(which='both', labelsize=ticksize)
    plt.setp(gfg.get_legend().get_texts(), fontsize=fontsize-2)

    structs_all = set(df['struct_hash'].to_list())
    structs_aflow = set(df_aflow['struct_hash'].to_list())
    structs_mp = set(df_mp['struct_hash'].to_list())
    structs_shared = structs_aflow & structs_mp
    df_shared = df[df['struct_hash'].isin(structs_shared)]
    print("# entries in shared set from AFLOW: ", sum(df_shared['auid'].notna()))
    print("# entries in shared set from MP: ", sum(df_shared['auid'].isna()))

    sns.histplot(data=df_shared, x=label, ax=ax[1], hue='database', palette=['red', 'green'])
    ax[1].get_legend().remove()
    ax[1].set_xlabel(x_label, fontsize=fontsize)
    ax[1].set_ylabel('Count', fontsize=fontsize)
    ax[1].tick_params(which='both', labelsize=ticksize)
    plt.tight_layout()
    fig.savefig(f'databases/hist_shared_{FLAGS.task}.png', bbox_inches='tight', dpi=600)
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

        with open(f"databases/{FLAGS.task}_splits/aflow_split.json", 'w', encoding="utf-8") as splits_file:
            json.dump(aflow_split, splits_file, indent=4, separators=(',', ': '))
        with open(f"databases/{FLAGS.task}_splits/mp_split.json", 'w', encoding="utf-8") as splits_file:
            json.dump(mp_split, splits_file, indent=4, separators=(',', ': '))
        break


if __name__ == "__main__":
    app.run(main)
