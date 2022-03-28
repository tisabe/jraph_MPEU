import argparse
import os

import numpy as np
from numpy import genfromtxt
import pandas
import matplotlib.pyplot as plt
import matplotlib


def main(args):
    # plot learning curves
    fig, ax = plt.subplots()
    for dirname in os.listdir(args.file):
        print(dirname)
        folder = os.path.join(args.file, dirname)
        try:
            loss = genfromtxt(folder+'/validation_loss.csv', delimiter=',')
            ax.plot(loss[:,0], loss[:,1], label=dirname)
        except OSError:
            print(f'{folder} not a valid path, path is skipped.')
    
    ax.legend()
    ax.set_xlabel('gradient step')
    ax.set_ylabel('MSE (eV^2), standardized')
    plt.yscale('log')
    plt.show()
    plt.savefig(args.file+'/curve.png')

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input super directory name')
    args = parser.parse_args()
    main(args)