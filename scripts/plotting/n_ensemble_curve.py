"""Plot loss metrics for different ensemble sizes of GNN models."""

import os
import json

from absl import app
from absl import flags
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_list('dirs', ['results/aflow/egap_rand_search'],
    'input directory names')
flags.DEFINE_bool('redo', False, 'Whether to redo inference.')
flags.DEFINE_bool('plot', False, 'Whether to plot parity plots.')
flags.DEFINE_integer('limit', None, 'If not None, a limit to the amount of data \
    read from the database.')
flags.DEFINE_integer('n_best', 10, 'number of best models to pick for ensemble')
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')
flags.DEFINE_string('unit', 'eV', 'kind of label that is trained on. Used to \
    define the plot label. e.g. "eV/atom" or "eV"')


def _rmse(x):
    return np.sqrt(np.mean(np.square(x)))


def get_predictions_df(directory):
    df = pd.DataFrame({})
    label_str = None
    workdir_paths = []
    workdirs_with_result = []

    # collect all workdirs
    with os.scandir(directory) as dirs:
        for entry in dirs:
            if entry.name.startswith('id') and entry.is_dir():
                workdir_paths.append(entry.path)
    #print(workdir_paths)

    for workdir in workdir_paths:
        with os.scandir(workdir) as workdir_items:
            for entry in workdir_items:
                if entry.name.endswith('result.csv') and entry.is_file():
                    workdirs_with_result.append(workdir)
                    continue

    for workdir in tqdm(workdirs_with_result[:]):
        model_dir = os.path.basename(os.path.normpath(workdir))
        csv_path = os.path.join(workdir, 'result.csv')
        df_model = pd.read_csv(csv_path)

        if 'prediction' in df_model.keys().to_list():
            predictions = df_model['prediction'].rename(model_dir)
        elif 'prediction_mean' in df_model.keys().to_list():
            # old result.csv files have different column name
            predictions = df_model['prediction_mean'].rename(model_dir)

        if label_str is None:
            # get the label string from config
            config_path = workdir + '/config.json'
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config_dict = json.load(config_file)
                label_str = config_dict['label_str']
                targets = df_model[label_str]
                targets.index = df_model['auid']
                df['target'] = targets
                splits = df_model['split']
                splits.index = df_model['auid']
                df['split'] = splits


        predictions.index = df_model['auid']
        df = pd.concat([df, predictions], axis=1)

    return df


def calc_split_metrics(df_in, splits):
    """For each split in splits, calculate error metrics of predictions."""
    df = df_in.copy()
    metrics_dict = {'MAE': np.mean, 'RMSE': _rmse, 'MdAE': np.median}
    id_cols = df.keys().to_list()
    id_cols = [i for i in id_cols if 'id' in i]
    df_metrics = pd.DataFrame({})

    for col in id_cols:
        df[col] = abs(df[col] - df['target'])
    for split in splits:
        df_split = df[df['split']==split][id_cols]
        for metric, fun in metrics_dict.items():
            result = df_split.apply(fun)
            result.name = metric + '_' + split
            df_metrics = pd.concat([df_metrics, result], axis=1)

    return df_metrics


def get_ensemble_pred(df_in, ids_best, weights=None):
    """Ensemble the predictions of models referenced by id in ids."""
    preds_ensemble = df_in[ids_best].mean(axis=1)
    return preds_ensemble


def get_ensemble_curve_df(directory, n_max):
    df = get_predictions_df(directory)
    splits = ['train', 'validation', 'test']
    df_metrics = calc_split_metrics(df, splits)
    #print(df_metrics['RMSE_validation'])
    ids_best = df_metrics.sort_values(
        by='RMSE_validation')

    metrics_best = []
    for n_best in range(1, n_max+1):
        ids_best_n = ids_best.iloc[:n_best].index.to_list()
        #df_best = df[['target', 'split']+ids_best]

        df['pred_ensemble'] = get_ensemble_pred(df, ids_best_n)
        metrics_funs = {'MAE': np.mean, 'RMSE': _rmse, 'MdAE': np.median}
        df_metrics = pd.DataFrame({})

        df['abs. error'] = abs(df['pred_ensemble'] - df['target'])

        metrics_dict = {'n_best': n_best, 'dir': directory}

        for split in splits:
            errors = df[df['split']==split]['abs. error']
            for metric, fun in metrics_funs.items():
                name = metric + '_' + split
                metrics_dict[name] = fun(errors)
        metrics_best.append(metrics_dict)

    metrics_best = pd.DataFrame(metrics_best)
    return metrics_best


def main(_):
    df_curves = pd.DataFrame({})
    for directory in FLAGS.dirs:
        print("getting data from ", directory)
        df_curve = get_ensemble_curve_df(directory, FLAGS.n_best)
        df_curves = pd.concat([df_curves, df_curve], axis=0, ignore_index=True)

    fig, ax = plt.subplots()
    g = sns.lineplot(
        data=df_curves, x='n_best', y='MAE_test', hue='dir', ax=ax,
    )
    ax.set_xlabel('Size of ensemble', fontsize=FLAGS.font_size)
    ax.set_ylabel(f'MAE ({FLAGS.unit})', fontsize=FLAGS.font_size)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    plt.tight_layout()
    folders = [
        os.path.basename(os.path.normpath(directory)) for directory in FLAGS.dirs]
    dir_to_label_dict = {
        'egap_rand_search': 'Random search',
        'egap_pbj_ensemble': 'Reference ensemble',
        'egap_pbj_val_ensemble': 'Randomized train/val split reference ensemble',
        'ef_rand_search': 'Random search',
        'ef_pbj_ensemble': 'Reference ensemble',
    }
    new_labels = [dir_to_label_dict[folder] for folder in folders]
    sns.move_legend(ax, 'best', title=None, labels=new_labels)
    plt.show()
    fig.savefig(
        FLAGS.dirs[0]+'/ensemble_curve.png', bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    app.run(main)
