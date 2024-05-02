from absl import app
from absl import flags
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold


FLAGS = flags.FLAGS
flags.DEFINE_string('out_path', 'databases/data_summary.csv', 'Where csv is saved')


def main(_):
    df = pd.read_csv(FLAGS.out_path)
    df['struct_hash'] = df['formula'].str.cat(df['spacegroup_pmg'].astype(str), sep='_')
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
    plt.semilogy()
    fig.savefig('databases/hist.png', bbox_inches='tight', dpi=600)
    plt.show()

    kfold = GroupKFold(n_splits=10)
    groups = df['struct_hash']
    for i, (train_index, test_index) in enumerate(
            kfold.split(X=df['asedb_id'], groups=groups)):
        print(f"Fold {i}:")
        print(f"#train: {len(train_index)}, #test: {len(test_index)}")


if __name__ == "__main__":
    app.run(main)
