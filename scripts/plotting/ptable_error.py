"""Plot the average error for each species in a periodic table structure."""

import os
from collections import defaultdict
import csv

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from ase.formula import Formula
import numpy as np
import seaborn as sns

from jraph_MPEU.utils import load_config, str_to_list
from jraph_MPEU.inference import get_results_df
from jraph_MPEU.input_pipeline import cut_egap

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_string('label', 'ef', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "ef" or "egap"')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')
flags.DEFINE_bool('element_names', False, 'Whether to print element names.')

# define the element types from the periodic table
element_types = {
    'Noble gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
    'Oxides': ['O'],
    'Reactive non metals': [
        'H', 'C', 'N', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
    'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'At'],
    'Post transition metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'],
    'Transition metals': [
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs'],
    'Lanthanoids/Actinoids': [
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
        'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
        'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
    'Alkali (earth) metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Li', 'Na',
        'K', 'Rb', 'Cs', 'Fr']
}


def get_type(element: str):
    for key, element_list in element_types.items():
        if element in element_list:
            return key
    raise KeyError('Element type not found')

def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    workdir = FLAGS.file
    df_path = workdir + '/result.csv'
    config = load_config(workdir)

    # set correct axis labels
    x_label = '# compounds in training set'
    if config.label_type == 'scalar':
        if FLAGS.label == 'egap':
            y_label = r'MAE ($E_g$/eV)'
        elif FLAGS.label == 'energy':
            y_label = r'Calculated $U_0$ (eV)'
        else:
            y_label = r'MAE ($E_{F}$ per atom/eV)'
    else:
        y_label = 'Accuracy per species'

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)
        df['numbers'] = df['numbers'].apply(str_to_list)

    if not 'prediction' in df.columns:
        df['prediction'] = df['prediction_mean']

    if config.label_type == 'scalar':
        df['abs. error'] = abs(df['prediction'] - df[config.label_str])
    else:
        df['class_true'] = df['Egap'].apply(cut_egap) #  1 is insulator, 0 is metal
        df['p_insulator'] = 1 - df['prediction']
        # calculate the class prediction by applying a threshold. Because of the
        # softmax outputs probability, the threshold is exactly 1/2
        df['class_pred'] = df['p_insulator'].apply(lambda p: (p > 0.5)*1)
        df['class_correct'] = df['class_true'] == df['class_pred']
        # make column with accuracy. TODO: explain better
        df['abs. error'] = 1 * df['class_correct']
    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    df = df.loc[lambda df_temp: df_temp['split'] == 'test']

    # dict with species as keys and list of errors
    errors_dict = defaultdict(list)

    for _, row in df.iterrows():
        symbols = row['formula']
        if not isinstance(symbols, str):
            continue

        counts = Formula(symbols).count()  # dictionary with species and number
        for symbol in counts.keys():
            errors_dict[symbol].append(row['abs. error'])

    # get species counts from train split dataframe
    count_dict = defaultdict(int)
    ldau_dict = defaultdict(int)

    for _, row in df_train.iterrows():
        symbols = row['formula']
        if not isinstance(symbols, str):
            continue

        counts = Formula(symbols).count()  # dictionary with species and number
        for symbol in counts.keys():
            count_dict[symbol] += 1
            ldau_dict[symbol] += row.ldau_type

    mae_dict = {}
    for species, error_list in errors_dict.items():
        mae_dict[species] = np.mean(error_list)

    with open(workdir + '/species_mae.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mae_dict.items():
            writer.writerow([key, value])

    # calculate the intersection of keys, for train and test set
    keys_intersect = count_dict.keys() & mae_dict.keys()

    # gather counts and mae's into lists
    species = []
    counts = []
    maes = []
    for key in keys_intersect:
        species.append(key)
        counts.append(count_dict[key])
        maes.append(mae_dict[key])
    df_plot = pd.DataFrame(
        data={'species': species, 'counts': counts, 'maes': maes}
    )
    df_plot['element class'] = df_plot['species'].apply(get_type)
    df_plot = df_plot.sort_values('element class', axis=0, ascending=False)

    # plot number of fit metric depending on number of compounds
    # split dataframe in half for better legend
    df1 = df_plot[df_plot['element class'] > 'Noble gases']
    df2 = df_plot[df_plot['element class'] <= 'Noble gases']

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_plot, x='counts', y='maes', hue='element class', ax=ax, s=80)
    if FLAGS.element_names:
        for txt, count, mae in zip(keys_intersect, counts, maes):
            ax.annotate(txt, (count, mae))
    ax.set_xlabel(x_label, fontsize=FLAGS.font_size)
    ax.set_ylabel(y_label, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(title='').set_visible(True)
    plt.rc('legend', fontsize=FLAGS.tick_size-3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/species_vs_count_notext.png', bbox_inches='tight', dpi=600)


    with open(workdir + '/species_count.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in count_dict.items():
            writer.writerow([key, value])

    # plot counts of species total and with ldau correction
    df_ldau = pd.DataFrame(columns=['Element', 'Number', 'Type'])
    for i, key in enumerate(count_dict.keys()):
        df_ldau.loc[i] = [key, count_dict[key], 'Total']
    N = len(df_ldau)
    for i, key in enumerate(count_dict.keys()):
        df_ldau.loc[i + N] = [key, ldau_dict[key], 'DFT+U']
    
    
    print(df_ldau.head())
    fig, ax = plt.subplots()
    sns.catplot(data=df_ldau, x='Element', y='Number', hue='Type', ax=ax, kind='bar')
    #ax.set_xlabel(x_label, fontsize=FLAGS.font_size)
    #ax.set_ylabel(y_label, fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    #ax.legend(title='').set_visible(True)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    #fig.savefig(workdir+'/species_vs_count_notext.png', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    app.run(main)
