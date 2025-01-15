"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.formula import Formula


FLAGS = flags.FLAGS
flags.DEFINE_list('paths', None, 'csv file paths', required=True)
flags.DEFINE_list('labels', None, 'labels for the prediction columns', required=True)
flags.DEFINE_string('fig_path', None, 'pathe where the plotted figure is saved.', required=True)
flags.DEFINE_integer('font_size', 14, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')


def get_axis_labels(label):
    """Return a tuple of x-axis and y-axis labels, based on label string."""
    match label:
        case 'Egap':
            return r'$E_g$ (eV)', r'$\hat{E_g}$ (eV)'
        case 'U0':
            return r'$U_0$ (eV)', r'$\hat{U_0}$ (eV)'
        case 'enthalpy_formation_atom':
            return r'$E_f$ (eV/atom)', r'$\hat{E_f}$ (eV/atom)'


def plot_common_polymorph(df_list, label_list):
    """Plot parity plot for the most common polymorph in the dataframe df."""
    plt.rc('font', size=FLAGS.font_size)
    plt.rc('xtick', labelsize=FLAGS.tick_size)
    plt.rc('ytick', labelsize=FLAGS.tick_size)
    plt.rc('legend', fontsize=FLAGS.font_size)
    plt.rc('legend', title_fontsize=FLAGS.font_size)
    plt.rc('axes', labelsize=FLAGS.font_size)

    fig, ax = plt.subplots(len(df_list), 3)
    for i, (df, label) in enumerate(zip(df_list, label_list)):
        #print(df['formula'])
        #df['formula_reduce'] = df['formula'].map(
        #    lambda x: Formula(str(x)).reduce()[0].format('abc'))
        x_label, y_label = get_axis_labels(label)
        df_grouped = df.groupby(['formula', 'spacegroup_relax']).aggregate(
            {'formula': 'first',
            'spacegroup_relax': 'first',
            label: ['mean', 'std'],
            'prediction': ['mean', 'std']}
        )
        counts_formula = df_grouped['formula']['first'].value_counts()
        print(counts_formula)
        for j in range(3):
            formula = counts_formula.index[j]
            print(formula)
            df_common = df_grouped[df_grouped['formula']['first'] == formula]
            counts_sg = df_common['spacegroup_relax']['first'].value_counts()
            print(counts_sg)

            ax[i][j].errorbar(
                df_common[label]['mean'],
                df_common['prediction']['mean'],
                xerr=df_common[label]['std'],
                yerr=df_common['prediction']['std'],
                fmt='o')
            formula_display = fr"{Formula(formula).format('abc').format('latex')}"
            # custom limits for specific subplots
            match (formula, label):
                case ['O8Si4', 'enthalpy_formation_atom']:
                    lim = [-2.9, -2.3]
                    ticks = [-2.8, -2.4]
                case ['O24Si12', 'enthalpy_formation_atom']:
                    lim = [-3, -1]
                    ticks = [-2.5, -1.5]
                case ['BiFeO3', 'enthalpy_formation_atom']:
                    lim = [-0.85, -0.52]
                    ticks = [-0.8, -0.6]
                case ['O8Si4', 'Egap']:
                    lim = [3.6, 6.1]
                    ticks = [4, 6]
                case _:
                    lim = None
                    ticks = None
            # set only if there was a mathed case previously
            if lim or ticks:
                ax[i][j].set_xlim(lim)
                ax[i][j].set_xticks(ticks)
                ax[i][j].set_ylim(lim)
                ax[i][j].set_yticks(ticks)
            ax[i][j].text(0.1, 0.8, formula_display, transform=ax[i][j].transAxes)
            # plot dashed grey line at x=y
            x_ref = np.linspace(*ax[i][j].get_xlim())
            ax[i][j].plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
            if j == 1:
                ax[i][j].set_xlabel(x_label)
            if j == 0:
                ax[i][j].set_ylabel(y_label)
            ax[i][j].tick_params(which='both')
            ax[i][j].set_box_aspect(1)

    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.fig_path, bbox_inches='tight', dpi=600)


def main(argv):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    df_list = []
    for path in FLAGS.paths:
        df = pd.read_csv(path)
        if not 'prediction' in df.columns:
            df['prediction'] = df['prediction_mean']
        # get dataframe with only split data
        df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
        df_list.append(df_test)

    plot_common_polymorph(df_list, FLAGS.labels)


if __name__ == "__main__":
    app.run(main)
