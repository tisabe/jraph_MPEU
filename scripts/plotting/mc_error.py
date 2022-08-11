"""This script plots and generates monte-carlo dropout inferences."""
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/test_dropout', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')

def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    workdir = FLAGS.file
    df_path = workdir + '/result_mc.csv'
    config = load_config(workdir)

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir, limit=FLAGS.limit, mc_dropout=True)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)

    df['abs. error'] = abs(df['prediction_mean'] - df[config.label_str])
    bins = [0, 2, 15, 74, 142, 167, 194, 230]
    labels = [
        'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
        'Trigonal', 'Hexagonal', 'Cubic']
    df['crystal system'] = pd.cut(df['spacegroup_relax'], bins, labels=labels)
    
    # get dataframe with only split data
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    mean_abs_err_train = df_train.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on train set: {mean_abs_err_train}')

    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    mean_abs_err_val = df_val.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on validation set: {mean_abs_err_val}')

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    mean_abs_err_test = df_test.mean(0, numeric_only=True)['abs. error']
    print(f'MAE on test set: {mean_abs_err_test}')

    mean_target = df.mean(0, numeric_only=True)[config.label_str]
    std_target = df.std(0, numeric_only=True)[config.label_str]
    print(f'Target mean: {mean_target}, std: {std_target} for {config.label_str}')

    sns.set_context('paper')
    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='abs. error',
        y='prediction_std',
        data=df,
        hue='split'
    )
    ax.set_xlabel('absolute error', fontsize=12)
    ax.set_ylabel('prediction std', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_error', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='crystal system',
        y='prediction_std',
        data=df_test,
        inner='quart',
        linewidth=1,
    )
    ax.set_ylabel('prediction std', fontsize=12)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_crystal_system', bbox_inches='tight', dpi=600)

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='ldau_type',
        y='prediction_std',
        data=df_test,
        inner='quart',
        linewidth=1,
    )
    ax.set_ylabel('prediction std', fontsize=12)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_ldau_type', bbox_inches='tight', dpi=600)




if __name__ == "__main__":
    app.run(main)
