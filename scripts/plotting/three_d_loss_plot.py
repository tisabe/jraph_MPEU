"""Three dimensional loss curve plot for the Loss curves."""

import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
from matplotlib import cbook, cm
from matplotlib.colors import LightSource


CSV_LOCATION = '/home/dts/Downloads/df_curves_egap.csv'
if __name__ == "__main__":
    input_df = pd.read_csv(CSV_LOCATION)
    # input_df['val_rmse'] = ast.literal_eval(input_df['validation_rmse_curve'].va)

    print(input_df.columns)

    for i in range(len(input_df['validation_rmse_curve'])):
        input_df['validation_rmse_curve'][i] = min(ast.literal_eval(input_df['validation_rmse_curve'][i]))
    
    df = input_df[['mp_steps', 'latent_size', 'validation_rmse_curve']]
    print(len(df['latent_size']))

    mp_step_list = []
    latent_size_list = []
    val_list = []

    for mp_step in [2, 3, 4, 5]:
        for latent_size in [128, 256, 512]:
            print(mp_step)
            print(latent_size)
            val_scores = df[(df['mp_steps'] == mp_step) & (df['latent_size'] == latent_size)]['validation_rmse_curve']
            print(val_scores)
            median_val_score = np.min(val_scores.to_numpy())
            print(f'ls: {latent_size}, mp: {mp_step} and val: {median_val_score}')
            mp_step_list.append(mp_step)
            latent_size_list.append(latent_size)
            val_list.append(median_val_score)


    x = mp_step_list
    y = latent_size_list
    z = np.array(val_list)
    # # z = np.array([[86.51636997377514], [171.84934697576574], [43.856065618121605], [86.18586512549301], [171.5167689712402], [43.51999269076107], [86.84880916468161], [172.18796731660453], [44.184118900059254], [87.18213956945152], [172.5187509122458], [44.5231263691537]])
    
    # x = np.reshape(x, (4, 3))
    # y = np.reshape(y, (4, 3))
    # z = np.reshape(z, (4, 3))

    # import matplotlib.tri as mtri

    # fig = plt.figure(figsize=(4, 3))

    # # Plot the surface.  The triangles in parameter space determine which x, y, z
    # # points are connected by an edge.
    # ax = fig.add_subplot(1, 1, 1, projection='3d')

    # print(x)
    # print(y)
    # print(z)

    # print(np.min(z))
        
    # ax.xaxis.pane.set_edgecolor('black')
    # ax.yaxis.pane.set_edgecolor('black')
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False


    # ax.set_xticks([2, 3, 4, 5], minor=False)
    # ax.set_yticks([128, 256, 384, 512], minor=False)

    # ax.set_zticks([0.47, 0.5, 0.53], minor=False)
    # ax.set_xlim([2, 5])
    # ax.set_ylim([128, 512])
    # ax.set_zlim([0.47, 0.53])

    # ax.yaxis.set_tick_params(labelsize=7)
    # ax.xaxis.set_tick_params(labelsize=7)
    # ax.zaxis.set_tick_params(labelsize=7)

    # eps_x= 0.0000000006
    # eps_y= 0.0000000006

    # eps_z= 0.0000000006

    # matplotlib.rc('font',size=28)
    # matplotlib.rc('font',family='serif')
    # matplotlib.rc('axes',labelsize=32)

    # ax.set_xlabel('MP Steps', fontsize=9)
    # ax.set_ylabel('Latent Size', fontsize=9)
    # ax.set_zlabel('RMSE (eV/atom)', fontsize=9)

    # ax.scatter3D(
    #     mp_step_list,
    #     latent_size_list,
    #     val_list, alpha=0.6, s=40, marker='o', color='tab:blue')
    # ax.plot_surface(x, y, z, alpha=0.3, linewidth=0.01, antialiased=False, edgecolors='tab:blue')

    # ax.grid(False)
    # fig.subplots_adjust(left=0, right=0.9, bottom=0.11, top=0.88, wspace=0.2, hspace=0.2)
    # ax.view_init(elev=10, azim=-78)
    # plt.tight_layout()

    # plt.savefig("/home/dts/Documents/hu/3d_scatter_plane_best_new.png", bbox_inches='tight', dpi=600)
    # plt.show()

    plt.figure()

    for latent_size in [128, 256, 512]:
        print(mp_step)
        print(latent_size)
        val_scores_df = df[df['latent_size'] == latent_size]
        val_scores_list = []
        for mp_step in [2, 3, 4, 5]:
            val_scores = val_scores_df[df['mp_steps'] == mp_step]['validation_rmse_curve']
            print(val_scores)
            median_val_score = np.min(val_scores.to_numpy())
            val_scores_list.append(median_val_score)
        print(len(val_scores_list))
        plt.plot(np.array([2,3,4,5]), val_scores_list, '--')
    plt.xlabel('MP Steps')
    plt.ylabel('Validation RMSE (eV/atom)')
    plt.legend(['128, 256, 512'])
    plt.show()

    # for mp_step in [2, 3, 4, 5]:
    #     print(mp_step)
    #     print(latent_size)
    #     val_scores_df = df[df['mp_steps'] == mp_step]
    #     val_scores_list = []
    #     for latent_size in [128, 256, 512]:
    #         val_scores = val_scores_df[df['latent_size'] == latent_size]['validation_rmse_curve']
    #         print(val_scores)
    #         median_val_score = np.min(val_scores.to_numpy())
    #         val_scores_list.append(median_val_score)
    #     print(len(val_scores_list))
    #     plt.plot(np.array([128, 256, 512]), val_scores_list, '--')
    # plt.xlabel('MP Steps')
    # plt.ylabel('Validation RMSE (eV/atom)')
    # plt.legend(['128, 256, 512'])
    # plt.show()
