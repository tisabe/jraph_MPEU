import argparse

import numpy as np
from numpy import genfromtxt
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score


def main(args):
    folder = args.file

    splits = ['train', 'validation', 'test']

    fig, ax = plt.subplots()

    marker_size = 0.3

    for split in splits:
        df_post = pandas.read_csv(folder+'/'+split+'_post.csv')

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

        ax.scatter(df_post['x'].to_numpy(), df_post['y'].to_numpy(), s=marker_size, label=split)
        
    #ax.axline((0,0), slope=1, color='red', label='x=y')
    ax.set_title('test post')
    ax.set_xlabel('target')
    ax.set_ylabel('prediction')
    ax.legend()

    plt.show()

    ### plot learning curves
    fig, ax = plt.subplots()

    for split in splits:
        loss = genfromtxt(folder+'/'+split+'_loss.csv', delimiter=',')
        ax.plot(loss[:,0], loss[:,1], label=split)
        if split=='validation':
            # plot the validation loss offset by early stopping patience
            loss[:,0] = loss[:,0] + 1e+6
            ax.plot(loss[:,0], loss[:,1], label=split+' offset')
    
    ax.legend()
    ax.set_xlabel('gradient step')
    ax.set_ylabel('loss (MAE), standardized')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show regression plot and loss curve.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input directory name')
    args = parser.parse_args()
    main(args)

