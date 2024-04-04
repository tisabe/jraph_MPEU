"""Uncertainty quantification using ensembles of models."""

import os
import json

import pandas as pd
from tqdm import tqdm
from absl import app, flags
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy import stats


FLAGS = flags.FLAGS
flags.DEFINE_string('directory', 'results/qm9/U0/uq_ensemble',
    'input directory name')


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

    for workdir in workdir_paths:
        with os.scandir(workdir) as workdir_items:
            for entry in workdir_items:
                if entry.name.endswith('result.csv') and entry.is_file():
                    workdirs_with_result.append(workdir)
                    continue

    for workdir in tqdm(workdirs_with_result[:]):
        model_dir = os.path.basename(os.path.normpath(workdir))
        csv_path = os.path.join(workdir, 'result.csv')
        df_model = pd.read_csv(csv_path, index_col='asedb_id')

        predictions = df_model['prediction'].rename('mu_'+model_dir)
        predictions_var = df_model['prediction_uq'].rename('sigma_'+model_dir)

        if label_str is None:
            # get the label string from config
            config_path = workdir + '/config.json'
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config_dict = json.load(config_file)
            label_str = config_dict['label_str']
            df['target'] = df_model[label_str]
            df['split'] = df_model['split']


        df = pd.concat([df, predictions, predictions_var], axis=1)

    return df


def plot_error_calibration(df):
    """Plot the error calibration curve for the estimated error 'prediction_std'.
    This is done by binning the prediction_std and calculating the """
    n_bins = 10
    var_bins = pd.qcut(df['total_sigma_recal'], q=n_bins, duplicates='drop')
    df['var_bin'] = var_bins
    df_grouped = df.groupby('var_bin')
    df_mean = df_grouped.mean(numeric_only=True)
    df_mean['RMV'] = df_mean['total_sigma_recal'].apply(np.sqrt)
    df_mean['RMSE'] = df_mean['squared_error'].apply(np.sqrt)
    # calculate expected normalized calibration error (ENCE)
    ence = np.mean(np.abs(df_mean['RMV']-df_mean['RMSE'])/df_mean['RMV'])
    print('ENCE: ', ence)
    fig, ax = plt.subplots()
    sns.scatterplot(
        ax=ax,
        x='RMV',
        y='RMSE',
        data=df_mean,
        #hue='split'
    )
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory + '/error_calibration.png', bbox_inches='tight', dpi=600)


def plot_quantile_calibration(df):
    """Plot the quantile calibration by plotting the predicted quantile 
    versus the empirical quantile."""
    quantile_pred = stats.percentileofscore(
        df['total_sigma_recal'], df['total_sigma_recal'])
    quantile_true = stats.percentileofscore(
        df['squared_error'], df['squared_error'])
    print(quantile_pred)
    print(quantile_true)
    fig, ax = plt.subplots()
    ax.scatter(quantile_pred, quantile_true)
    x_ref = np.linspace(*ax.get_xlim())
    ax.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()
    fig.savefig(FLAGS.directory + '/quantile_calibration.png', bbox_inches='tight', dpi=600)


def print_sharpness(df):
    """Print out sharpness metrics Root-mean-variation (RMV) and coefficient of variation
    (CV) of the predicted uncertainties in df."""
    rmv = np.sqrt(np.mean(df['total_sigma_recal']))
    print("RMV: ", rmv)
    cv = np.std(np.sqrt(df['total_sigma_recal']))/np.mean(np.sqrt(df['total_sigma_recal']))
    print("CV: ", cv)


def main(_):
    """Call functions defined in this module. And make pretty plots."""
    df = get_predictions_df(FLAGS.directory)
    cols = df.keys().to_list()
    cols_mu = [col for col in cols if 'mu_id' in col]
    cols_sigma = [col for col in cols if 'sigma_id' in col]
    df['mu_mean'] = df[cols_mu].mean(axis=1)

    # calculate uncertainty contributions and total uncertainty
    df['aleatoric'] = df[cols_sigma].mean(axis=1)
    df['epistemic'] = (df[cols_mu]**2).mean(axis=1) - df['mu_mean']**2
    df['total_sigma'] = df['aleatoric'] + df['epistemic']
    df['squared_error'] = (df['target'] - df['mu_mean'])**2

    # recalibrate uncertainties using isotonic regression model
    df_val = df.loc[lambda df_temp: df_temp['split'] == 'validation']
    model = IsotonicRegression(y_min=0)
    model.fit(df_val['total_sigma'], df_val['squared_error'])

    df['total_sigma_recal'] = model.transform(df['total_sigma'])
    """sns.scatterplot(
        data=df,
        x='squared_error',
        y='total_sigma_recal',
        hue='split'
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.show()"""

    df_test = df.loc[lambda df_temp: df_temp['split'] == 'test']
    print("MAE: ", np.mean(np.abs(df_test['target'] - df_test['mu_mean'])))
    print("RMSE: ", np.sqrt(np.mean(df_test['squared_error'])))

    plot_error_calibration(df_test)
    plot_quantile_calibration(df_test)
    print_sharpness(df_test)


if __name__ == "__main__":
    app.run(main)
