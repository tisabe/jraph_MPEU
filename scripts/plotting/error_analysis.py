"""This script produces plots to evaluate the errors of the model inferences,
using different metrics such as atomic numbers, number of species etc.
"""
import argparse

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from jraph_MPEU.input_pipeline import get_datasets
from jraph_MPEU.utils import load_config
from jraph_MPEU.inference import load_inference_file


def main(args):
    """Get the model inferences and plot regression."""
    logging.set_verbosity(logging.INFO)
    workdir = args.folder

    config = load_config(workdir)
    inference_dict = load_inference_file(workdir, redo=args.redo)
    logging.info('Loading datasets.')
    dataset, _, mean, std, num_list = get_datasets(config)

    fig, ax = plt.subplots(1, 2)
    marker_size = 0.3

    splits = inference_dict.keys()
    #units = input("Type units of prediction and target: ")
    units = "test"
    df = pd.DataFrame({})
    df_atomic_numbers = pd.DataFrame({})
    num_species_list = []
    num_nodes_list = []
    split_list = []
    for split in splits:
        for i, graph in enumerate(dataset[split].data):
            row_dict = {}
            row_atomic_number = {}
            split_list.append(split)
            nodes = graph.nodes
            num_nodes_list.append(len(nodes))
            num_species_list.append(len(set(nodes)))
            atomic_numbers = [num_list[int(node)] for node in nodes]
            row_dict["split"] = split
            row_dict["num_nodes"] = len(nodes)
            row_dict["num_species"] = len(set(nodes))
            pred = inference_dict[split]['preds'][i]
            target = inference_dict[split]['targets'][i]
            error = np.abs(pred - target)
            row_dict["mae"] = error
            df = df.append(row_dict, ignore_index=True)
            for num in set(atomic_numbers):
                df_atomic_numbers = df_atomic_numbers.append(
                    {"atomic_number": num, "mae": error, "split": split},
                    ignore_index=True
                )

    sns.violinplot(ax=ax[0], x="num_species", y="mae", hue="split", data=df)
    sns.scatterplot(ax=ax[1], x="num_nodes", y="mae", data=df, hue="split")
    #ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.tight_layout()

    plt.show()
    fig.savefig(workdir+'/fit.png', bbox_inches='tight', dpi=600)

    sns.scatterplot(x="atomic_number", y="mae", hue="split", data=df_atomic_numbers)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot model prediction errors for different distributions.')
    parser.add_argument(
        '-f', '-F', type=str, dest='folder', default='results/qm9/test',
        help='input directory name')
    parser.add_argument(
        '--redo', dest='redo', action='store_true'
    )
    args_main = parser.parse_args()
    main(args_main)
