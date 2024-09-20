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

    for i in range(len(input_df['validation_rmse_curve'])):
        input_df['validation_rmse_curve'][i] = min(ast.literal_eval(input_df['validation_rmse_curve'][i]))
    
    df = input_df[['mp_steps', 'latent_size', 'validation_rmse_curve']]

    # df_grouped = df.groupby(['mp_steps', 'latent_size']).median()
    # df.groupby('mp_steps').mean().reset_index()
    print(len(df['latent_size']))
    # print(len(df_grouped['latent_size']))

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

    # print('latent_size_list')
    # print(latent_size_list)
    # print('mp_step_list')
    # print(mp_step_list)
    # print('val_list')
    # print(val_list)

    

    # x, y = np.meshgrid(df['mp_steps'].unique().tolist(), df['latent_size'].unique().tolist())
    # print(x)
    # print(y)
    # print(df_grouped.head())
    # print(df_grouped.columns)
    # print('here')
    # print(df_grouped.validation_rmse_curve)
    # print(df_grouped.validation_rmse_curve)
    # plt.figure()
    # df_grouped.plot()
    # plt.show()

    # print(df_grouped.validation_rmse_curve.index[0])

    # fig = plt.figure()

    # # add axes
    # ax = fig.add_subplot(111, projection='3d')

    # # # plot the plane



    # ax.set_xlabel('Message Passing Steps')
    # ax.set_ylabel('Latent Space Size')
    # ax.set_zlabel('Mean Validation RMSE (eV/atom)')

    # # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # # ls = LightSource(270, 45)
    # # # To use a custom hillshading mode, override the built-in shading and pass
    # # # in the rgb colors of the shaded surface calculated from "shade".
    # # # rgb = ls.shade(val_list, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # # surf = ax.plot_surface(latent_size_list, mp_step_list, val_list, rstride=1, cstride=1,
    # #                     # facecolors=rgb,
    # #                     linewidth=0, antialiased=False, shade=False)

    x = mp_step_list
    y = latent_size_list
    z = np.array(val_list)
    # # z = np.array([[86.51636997377514], [171.84934697576574], [43.856065618121605], [86.18586512549301], [171.5167689712402], [43.51999269076107], [86.84880916468161], [172.18796731660453], [44.184118900059254], [87.18213956945152], [172.5187509122458], [44.5231263691537]])
    
    x = np.reshape(x, (4, 3))
    y = np.reshape(y, (4, 3))
    z = np.reshape(z, (4, 3))
    # # ax = plt.axes(projection='3d')
    # # ax.plot_surface(x, y, z)

    # # plt.show()

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
    #                     linewidth=0, antialiased=False, shade=False)

    # ax.plot_wireframe(x, y, z, rstride=1, cstride=1,cmap="autumn")

    import matplotlib.tri as mtri

    fig = plt.figure(figsize=(4, 3))
    # Make a mesh in the space of parameterisation variables u and v
    # u = np.linspace(64, 320, endpoint=True, num=10)
    # v = np.linspace(1, 6, endpoint=True, num=10)
    # tri = mtri.Triangulation(u, v)

    # Plot the surface.  The triangles in parameter space determine which x, y, z
    # points are connected by an edge.
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    # ax.set_zlim(-1, 1)


    # x = [128, 256, 512]
    # y = [2, 3, 4, 5]
    # x, y = np.meshgrid(x, y)
    # z = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
    print(x)
    print(y)
    print(z)
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, # facecolors=rgb,
    #                     linewidth=0, antialiased=False, shade=False)

    print(np.min(z))

    # ###patch start###
    # from mpl_toolkits.mplot3d.axis3d import Axis
    # if not hasattr(Axis, "_get_coord_info_old"):
    #     def _get_coord_info_new(self, renderer):
    #         mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
    #         mins += deltas / 4
    #         maxs -= deltas / 4
    #         return mins, maxs, centers, deltas, tc, highs
    #     Axis._get_coord_info_old = Axis._get_coord_info  
    #     Axis._get_coord_info = _get_coord_info_new
    # ###patch end###
        
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ax.set_xticks([2, 3, 4, 5], minor=False, rotation=70)
    # ax.set_yticks([128, 256, 384, 512], minor=False, rotation=70)
    # ax.set_zticks([0.47, 0.5, 0.53], minor=False)




    # plt.rcParams['grid.linewidth'] = 1   # change linwidth


    ax.set_xticks([2, 3, 4, 5], minor=False)
    ax.set_yticks([128, 256, 384, 512], minor=False)
    # ax.tick_params(axis='x', labelrotation=45)
    # ax.tick_params(axis='y', labelrotation=10)

    ax.set_zticks([0.47, 0.5, 0.53], minor=False)
    ax.set_xlim([2, 5])
    ax.set_ylim([128, 512])
    ax.set_zlim([0.47, 0.53])

    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)

    eps_x= 0.0000000006
    # eps_y= -0.005
    eps_y= 0.0000000006

    eps_z= 0.0000000006


    # ax.axes.set_xlim3d(left=2+10*eps_x, right=5.0-10*eps_x) 
    # ax.axes.set_ylim3d(bottom=128+eps_y, top=512-eps_y) 
    # ax.axes.set_zlim3d(bottom=0.47+eps_z/100, top=0.53-eps_z/100)     
    # ax.axes.set_xlim3d(left=2.02, right=4.98) 
    # ax.axes.set_ylim3d(bottom=128, top=512) 
    # ax.axes.set_zlim3d(bottom=0.47, top=0.53) 

    # plt.ylim([128, 512])
    # plt.xlim([2, 5])

    matplotlib.rc('font',size=28)
    matplotlib.rc('font',family='serif')
    matplotlib.rc('axes',labelsize=32)

    # ax.xaxis._axinfo['tick']['inward_factor'] = 0
    # ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    # ax.yaxis._axinfo['tick']['inward_factor'] = 0
    # ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    # ax.zaxis._axinfo['tick']['inward_factor'] = 0
    # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.set_xlabel('Message Passing$\n$ Steps', fontsize=9)
    ax.set_ylabel('Latent Size', fontsize=9)
    ax.set_zlabel('RMSE (eV/atom)', fontsize=9)

    # ax.yaxis._axinfo["grid"].update({"linewidth":0.05, "color" : "black"})
    # ax.xaxis._axinfo["grid"].update({"linewidth":0.05, "color" : "black"})
    # ax.zaxis._axinfo["grid"].update({"linewidth":0.05, "color" : "black"})

    # ax.yaxis._axinfo["grid"].update({"color" : "black"})
    # ax.xaxis._axinfo["grid"].update({"color" : "black"})
    # ax.zaxis._axinfo["grid"].update({"color" : "black"})

    ax.scatter3D(
        mp_step_list,
        latent_size_list,
        val_list, alpha=0.6, s=40, marker='o', color='tab:blue')
    ax.plot_surface(x, y, z, alpha=0.3, linewidth=0.001, antialiased=False, edgecolors='tab:blue')


    # az -57, ele 13
    # -65 16
    fig.subplots_adjust(left=0, right=0.9, bottom=0.11, top=0.88, wspace=0.2, hspace=0.2)
    ax.view_init(elev=12, azim=-65)
    plt.tight_layout()

    plt.savefig("/home/dts/Documents/hu/3d_scatter_plane_best_new.png", bbox_inches='tight', dpi=800)
    plt.show()

