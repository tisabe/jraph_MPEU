"""This script plots and generates monte-carlo dropout inferences."""
import os
from collections import Counter

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
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV/atom', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')

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

    df_train_test = df.loc[lambda df_temp: df_temp['split'] != 'validation']
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

    # convert split labels for dataframe with all splits
    split_convert = {
        'train': 'Training', 'validation': 'Validation', 'test': 'Test'}
    df_train_test['split'] = df_train_test['split'].apply(
        lambda x: split_convert[x])

    sns.set_context('paper')
    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='abs. error',
        y='prediction_std',
        data=df_train_test,
        hue='split'
    )
    ax.set_xlabel('Absolute error', fontsize=FLAGS.font_size)
    ax.set_ylabel('Prediction STDEV', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.legend(title='', fontsize=FLAGS.font_size-3)  # disable 'split' title
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_error', bbox_inches='tight', dpi=600)
    corr = df_test['abs. error'].corr(df_test['prediction_std'])
    print(f"Correlation on test split: {corr}")

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='crystal system',
        y='prediction_std',
        data=df_test,
        linewidth=1,
        color=u'#1f77b4',
        #cut=0,  # limit the violins to data range
    )
    ax.set_ylabel('Prediction STDEV', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size, width=0)
    ax.set_xlabel('')
    ax.set_ylim(bottom=0, top=0.72)

    col = df_train['crystal system']
    counts = Counter(col)
    ax.text(0.04, 0.90, counts['Triclinic'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.17, 0.90, counts['Monoclinic'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.31, 0.90, counts['Orthorhombic'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.455, 0.90, counts['Tetragonal'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.60, 0.90, counts['Trigonal'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.74, 0.90, counts['Hexagonal'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)
    ax.text(0.88, 0.90, counts['Cubic'],
        fontsize=FLAGS.font_size-3, transform=ax.transAxes)

    plt.xticks(rotation=60)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_crystal_system', bbox_inches='tight', dpi=600)
    # print counts of different crystal systems
    col = df_train['crystal system']
    print(Counter(col))
    print(f"Minimum std: {df_test['prediction_std'].min()}")

    # convert integer dft_type to string type
    int_to_string_type = {0: 'No correction', 2: 'DFT+U correction'}
    df_test['ldau_type'] = df_test['ldau_type'].apply(
        lambda x: int_to_string_type[x])

    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x='ldau_type',
        y='prediction_std',
        data=df_test,
        linewidth=1,
        color=u'#1f77b4',
        #cut=0,  # limit the violins to data range
    )
    ax.set_ylabel('Prediction STDEV', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size, width=0)
    ax.set_xlabel('')
    ax.set_ylim(bottom=0, top=0.72)

    col = df_train['ldau_type']
    counts = Counter(col)
    ax.text(0.2, 0.90, counts[0.0], fontsize=FLAGS.font_size, transform=ax.transAxes)
    ax.text(0.72, 0.90, counts[2.0], fontsize=FLAGS.font_size, transform=ax.transAxes)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/mc_ldau_type', bbox_inches='tight', dpi=600)
    # print counts of different dft types
    col = df_train['ldau_type']
    print(Counter(col))




if __name__ == "__main__":
    app.run(main)
