import argparse

import numpy as np
from numpy import genfromtxt
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import pickle


def plot_fit(predictions, splits, folder):
    fig, ax = plt.subplots(2)

    marker_size = 0.3
    
    for split in splits:
        df_post = predictions[split]

        # calculate fit metrics
        x = df_post['x'].to_numpy()
        y = df_post['y'].to_numpy()
        r2 = r2_score(x, y)
        print(f'R2 score on {split} data: {r2}')
        error = np.abs(x - y)
        mae = np.mean(error)
        rmse = np.sqrt(np.mean(np.square(error)))
        print(f'MAE on {split} data: {mae}'.format(mae))
        print(f'RMSE score on {split} data: {rmse}'.format(rmse))

        ax[0].scatter(df_post['x'].to_numpy(), df_post['y'].to_numpy(), s=marker_size, label=split)
        ax[1].scatter(df_post['x'].to_numpy(), error, s=marker_size, label=split)
        
    #ax.axline((0,0), slope=1, color='red', label='x=y')
    ax[0].set_title('Model regression performance')
    ax[0].set_ylabel('prediction')
    ax[1].set_xlabel('target')
    ax[1].set_ylabel('error')
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale('log')
    plt.show()
    plt.savefig(folder+'/fit.png')


def plot_curve(metrics_dict, splits, folder):
    fig, ax = plt.subplots()

    for split in splits:
        loss = np.array(metrics_dict[split])
        ax.plot(loss[:,0], loss[:,1], label=split)
        '''
        if split=='validation':
            # plot the validation loss offset by early stopping patience
            loss[:,0] = loss[:,0] + 1e+6
            ax.plot(loss[:,0], loss[:,1], label=split+' offset')
        '''        
    
    ax.legend()
    ax.set_xlabel('gradient step')
    ax.set_ylabel('MSE (eV^2), standardized')
    plt.yscale('log')
    plt.show()
    plt.savefig(folder+'/curve.png')


def main(args):
    folder = args.file

    splits = ['train', 'validation', 'test']

    ### plot predictions fit
    try:
        predictions_path = folder+'/predictions.pkl'
        with open(predictions_path, 'rb') as f:
            predictions = pickle.load(f)
        plot_fit(predictions, splits, folder)
    except FileNotFoundError:
        print(f'Did not find {predictions_path}, skipping fit plot.')
    
    ### plot learning curves
    try:
        metrics_path = folder+'/metrics.pkl'
        with open(metrics_path, 'rb') as f:
            metrics_dict = pickle.load(f)
        plot_curve(metrics_dict, splits, folder)
    except FileNotFoundError:
        print(f'Did not find {metrics_path}, skipping curve plot.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show regression plot and loss curve.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input directory name')
    args = parser.parse_args()
    main(args)

