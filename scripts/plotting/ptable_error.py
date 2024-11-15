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
from jraph_MPEU.input_pipeline import cut_egap


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', 'results/qm9/test', 'input directory name')
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


def is_binary_oxide(formula: str):
    counts = Formula(formula).count()
    return (len(counts) == 2) and ('O' in counts)


def get_type(element: str):
    for key, element_list in element_types.items():
        if element in element_list:
            return key
    raise KeyError('Element type not found')


def formula_to_latex(formula: str):
    return Formula(formula).format('latex')


def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    workdir = FLAGS.workdir
    df_path = os.path.join(workdir, 'result.csv')
    config = load_config(workdir)

    # set correct axis labels
    x_label = '# compounds in training set'
    match (config.label_type, FLAGS.label):
        case ['scalar', 'egap']:
            y_label = r'MAE ($E_g$/eV)'
        case ['scalar', 'ef']:
            y_label = r'MAE ($E_f$ per atom/eV)'
        case ['class', _]:
            y_label = 'Accuracy per species'
        case _:
            raise ValueError(
                f'Unknown label combination {config.label_type, FLAGS.label}')

    if not os.path.exists(df_path):
        raise FileNotFoundError(
            "Evaluation needs to be done first, to generate result csv.")
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
        # make column with accuracy.
        df['abs. error'] = 1 * df['class_correct']

    df['formula_latex'] = df['formula'].apply(formula_to_latex)
    # filter for binary oxides:
    df = df[df['formula'].apply(is_binary_oxide)]

    df_train = df.loc[lambda df_temp: df_temp['split'] == 'train']
    df = df.loc[lambda df_temp: df_temp['split'] == 'test']
    df = df.sort_values(by=['abs. error'])
    print("Five best and worst predictions: ")
    cols = ['auid', 'formula_latex', 'spacegroup_relax', 'Egap', 'prediction']
    print(df[cols][:5])
    print(df[cols][-5:])

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
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(
        os.path.join(workdir, 'species_vs_count_notext.png'),
        bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)
