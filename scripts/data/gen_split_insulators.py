"""Generate a splits.json file that contains only predicted insulators.
Uses the model trained and saved in workdir to make predictions of metal or
non-metal (or loads results.csv if available). Using these predictions together
with the corresponding id in the ase database, a split.json file is generated,
which can be used to train band gap models on predicted insulators."""

import os
import json

from absl import app
from absl import flags
from absl import logging

import pandas as pd

from jraph_MPEU.utils import str_to_list
from jraph_MPEU.inference import get_results_df

FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'results/qm9/test', 'input directory name')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('font_size', 12, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 12, 'font size to use in labels')


def main(argv):
    """Get the model inferences and plot classification probabilities and
    scores like accuracy and ROC-AUC on the test set."""
    logging.set_verbosity(logging.INFO)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    workdir = FLAGS.file
    df_path = workdir + '/result.csv'

    if not os.path.exists(df_path) or FLAGS.redo:
        logging.info('Did not find csv path, generating DataFrame.')
        df = get_results_df(workdir, FLAGS.limit)
        df.head()
        print(df)
        df.to_csv(df_path, index=False)
    else:
        logging.info('Found csv path. Reading DataFrame.')
        df = pd.read_csv(df_path)
        df['numbers'] = df['numbers'].apply(str_to_list)

    # saved in column prediction is probability of being a metal, the
    # complement of this is the probability of being an insulator
    df['p_insulator'] = 1 - df['prediction']
    # calculate the class prediction by applying a threshold. Because of the
    # softmax outputs probability, the threshold is exactly 1/2
    df['class_pred'] = df['p_insulator'].apply(lambda p: (p > 0.5)*1)
    # convert asedb_id column to integer type
    df['asedb_id'] = df['asedb_id'].astype('int')

    df_ins = df[df['class_pred'] == 1]
    print(df_ins['asedb_id'], type(df_ins['asedb_id']))
    split_dict = {}
    for asedb_id, split in zip(list(df_ins['asedb_id']), list(df_ins['split'])):
        split_dict[str(asedb_id)] = split
    #print(split_dict)

    with open(os.path.join(workdir, 'splits_test.json'), 'w') as splits_file:
        json.dump(split_dict, splits_file, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    app.run(main)
