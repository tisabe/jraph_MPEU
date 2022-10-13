"""Plot the average error for each species in a periodic table structure."""

import os
from collections import defaultdict
import csv

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
from ase.formula import Formula
import numpy as np
import seaborn as sns

from jraph_MPEU.utils import load_config, str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')

# define the element types from the periodic table
element_types = {
    'Noble gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
    'Reactive non metals': [
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
    'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'At'],
    'Post transition metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'],
    'Transition metals': [
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs'],
    'Lanthanoids': [
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
        'Tm', 'Yb', 'Lu'],
    'Actinoids': [
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr'],
    'Alkaline earth metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
    'Alkali metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
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

    df['abs. error'] = abs(df['prediction'] - df[config.label_str])
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

    for _, row in df_train.iterrows():
        symbols = row['formula']
        if not isinstance(symbols, str):
            continue

        counts = Formula(symbols).count()  # dictionary with species and number
        for symbol in counts.keys():
            count_dict[symbol] += 1

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

    fig, ax = plt.subplots()
    #ax.scatter(counts, maes)
    df_plot = pd.DataFrame(
        data={'species': species, 'counts': counts, 'maes': maes}
    )
    df_plot['element class'] = df_plot['species'].apply(get_type)
    df_plot = df_plot.sort_values('element class', axis=0, ascending=False)
    print(df_plot)
    sns.scatterplot(data=df_plot, x='counts', y='maes', hue='element class', ax=ax)

    for txt, x, y in zip(keys_intersect, counts, maes):
        ax.annotate(txt, (x, y))
    ax.set_xlabel(
        'Number of compounds in training split containing species', fontsize=12
    )
    #ax.set_ylabel('MAE per species (formation energy per atom / eV)', fontsize=12)
    ax.set_ylabel(r'MAE per species (E$_{BG}$ / eV)', fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/species_vs_count.png', bbox_inches='tight', dpi=600)

    # make the same plot but without text to add again manually
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_plot, x='counts', y='maes', hue='element class', ax=ax)
    ax.set_xlabel(
        'Number of compounds in training split containing species', fontsize=12
    )
    ax.set_ylabel('MAE per species (formation energy per atom / eV)', fontsize=12)
    #ax.set_ylabel(r'MAE per species (E$_{BG}$ / eV)', fontsize=12)
    ax.legend(title='').set_visible(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(workdir+'/species_vs_count_notext.png', bbox_inches='tight', dpi=600)


    with open(workdir + '/species_count.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in count_dict.items():
            writer.writerow([key, value])

if __name__ == "__main__":
    app.run(main)
