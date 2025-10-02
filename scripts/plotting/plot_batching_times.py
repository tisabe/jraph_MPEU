""" Plot batching times.

Let's make one plot for the batching times for all options:

static
round_True
jnp
np

dynamic

And one plot for the update functions.

First plot, will be for GPU only and SchNett.

Second plot will be for CPU only and SchNett.

Third plot with GPU only and MPEU.

Fourth plot will be CPU only and MPEU.


Let's use the dataframes directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import argparse

BASE_DIR = '/home/dts/Documents/hu'
STATIC_NP_PROFILING_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_23_05__27_06_2024.csv'
PROFILING_JNP_CSV = 'parsed_profiling_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'
ROUND_PROFILING_JNP_CSV = 'parsed_profiling_round_to_multiple_qm9_aflow_schnet_100k_steps_23_54__8_03_2024.csv'



def create_four_panel_profiling_plot(computing_hardware='GPU'):
    """Here we want to feed into the correct dataframes to be able to easily make a four panel plot
    
    update_time
    """
    # x1, x2, x3, x4
    

def main(args):
    # plot learning curves
    df = pd.DataFrame({})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show ensemble of loss curves.')
    parser.add_argument('-f', '-F', type=str, dest='file', default='results_test',
                        help='input super directory name')
    args = parser.parse_args()
    main(args)






























# """ 
# AFLOW AFLOW GPU

# Dynamic
# 16	16.0000	NaN	NaN
# 32	32.0000	0.0009	0.0023
# 64	64.0000	0.0016	0.0030
# 128	128.0000	0.0030	0.0047

# Static (jnp)
# 16	16.0000	0.0029	0.0044
# 32	32.0000	0.0033	0.0049
# 64	64.0000	0.0040	0.0057
# 128	128.0000	0.0051	0.0074

# Static (np) round True
# batching: 0.0005, 0.0021
# update: 0.0043, 0.0108
# 16	16.0000	0.0005	0.0043
# 32	32.0000	NaN	NaN
# 64	64.0000	NaN	NaN
# 128	128.0000	0.0021	0.0108

# Static (np) round False
# batching: 0.0005, 0.0021
# update: 0.0020, 0.0044
# 16	16.0000	0.0005	0.0020
# 32	32.0000	NaN	NaN
# 64	64.0000	NaN	NaN
# 128	128.0000	0.0021	0.0044
# """

# import numpy as np
# import matplotlib.pyplot as plt
# marker_size = 11

# # Schnett - AFLOW - GPU
# dynamic_aflow_schnett_gpu_batch_size = [2, 3, 4]
# dynamic_aflow_schnett_gpu_batching_time = np.array([0.0009, 0.0016, 0.0030])*1000
# dynamic_aflow_schnett_gpu_batching_time_std = np.array([0.0009, 0.0016, 0.0030])*1000

# dynamic_aflow_schnett_gpu_update_time = np.array([0.0023, 0.0030, 0.0047])*1000

# # Static jnp
# static_jnp_aflow_schnett_gpu_batch_size = [1, 2, 3, 4]
# static_jnp_aflow_schnett_gpu_batching_time = np.array([0.0029, 0.0033, 0.0040, 0.0051])*1000
# static_jnp_aflow_schnett_gpu_batching_time_std = np.array([])*1000

# static_jnp_aflow_schnett_gpu_update_time = np.array([0.0023, 0.0030, 0.0047, 0.0074])*1000

# # Static np round True
# static_np_round_true_aflow_schnett_gpu_batch_size = [1, 4]
# static_np_round_true_aflow_schnett_gpu_batching_time = np.array([0.0005, 0.0021])*1000
# static_np_round_true_aflow_schnett_gpu_batching_time_std = np.array([0.0005, 0.0021])*1000

# static_np_aflow_round_true_schnett_gpu_update_time = np.array([0.0043, 0.0108])*1000

# # Static np round False
# static_np_round_false_aflow_schnett_gpu_batch_size = [1, 4]
# static_np_round_false_aflow_schnett_gpu_batching_time = np.array([0.0005, 0.0021])*1000
# static_np_round_false_aflow_schnett_gpu_batching_std = np.array([0.0000, 0.00002])*1000

# static_np_aflow_round_false_schnett_gpu_update_time = np.array([0.0020, 0.0044])*1000
# static_np_round_false_aflow_schnett_gpu_update_std = np.array([0.00003, 0.00002])*1000



# fig_gpu, ax_gpu = plt.subplots(2, 2, figsize=(7, 7)) #, gridspec_kw={'height_ratios': [1]})

# # First plot is batching: AFLOW - GPU - SchNett
# ax_gpu[0, 0].plot(dynamic_aflow_schnett_gpu_batch_size, dynamic_aflow_schnett_gpu_batching_time,
#     'x', markersize=marker_size, alpha=1, color='k')

# ax_gpu[0, 0].plot(static_jnp_aflow_schnett_gpu_batch_size, static_jnp_aflow_schnett_gpu_batching_time,
#     '^', markersize=marker_size, alpha=1, color='r')

# ax_gpu[0, 0].plot(
#     static_np_round_false_aflow_schnett_gpu_batch_size, static_np_round_false_aflow_schnett_gpu_batching_time,
#     '>', markersize=marker_size, alpha=1, color='b')

# ax_gpu[0, 0].plot(static_np_round_true_aflow_schnett_gpu_batch_size,
#                        static_np_round_true_aflow_schnett_gpu_batch_size,
#     'v', markersize=marker_size, alpha=1, color='y')

# ax_gpu[0, 0].legend(
#     ['dynamic', 'static jnp', 'static np', 'static np round to 64'], fontsize=12, loc='upper left')

# ax_gpu[0, 0].set_xlim(0, 5)
# ax_gpu[0, 0].set_ylim(0, 10)

# ax_gpu[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
# ax_gpu[0, 0].set_ylabel('Time (ms)', fontsize=12)


# # Gradient Update plot
# ax_gpu[1, 0].plot(dynamic_aflow_schnett_gpu_batch_size, dynamic_aflow_schnett_gpu_update_time,
#     '+', markersize=marker_size, alpha=1, color='k')
# ax_gpu[1, 0].plot(static_jnp_aflow_schnett_gpu_batch_size, static_jnp_aflow_schnett_gpu_update_time,
#     '^', markersize=marker_size, alpha=1, color='r')
# ax_gpu[1, 0].plot(static_np_round_false_aflow_schnett_gpu_batch_size,
#                      static_np_aflow_round_true_schnett_gpu_update_time,
#                      '>', markersize=marker_size, alpha=1, color='b')
# ax_gpu[1, 0].plot(static_np_round_true_aflow_schnett_gpu_batch_size,
#                      static_np_aflow_round_false_schnett_gpu_update_time,
#     'v', markersize=marker_size, alpha=1, color='y')

# ax_gpu[1, 0].set_ylim(0, 10)
# ax_gpu[1, 0].set_xlim(0, 5)
# ax_gpu[1, 0].set_ylabel('Time (ms)', fontsize=12)
# ax_gpu[1, 0].set_xlabel('Batch size', fontsize=12)


# """ Second Plot
# SchNet Qm9 GPU
# dynamic
# 0.0006, 0.0010, 0.0018, 0.0034
# 0.0019, 0.0024, 0.0034, 0.0056
# 16	16.0000	0.0006	0.0019
# 32	32.0000	0.0010	0.0024
# 64	64.0000	0.0018	0.0034
# 128	128.0000	0.0034	0.0056

# static - jnp
# 0.0032, 0.0036, 0.0043, 0.0055
# 0.0046, 0.0052, 0.0065, 0.0095
# 16	16.0000	0.0032	0.0046
# 32	32.0000	0.0036	0.0052
# 64	64.0000	0.0043	0.0065
# 128	128.0000	0.0055	0.0095


# static (np round False)
# batching: 0.0005, 0.0007, 0.0012, 0.0023
# update: 0.0019, 0.0023, 0.0035, 0.0062
# 16	16.0000	0.0005	0.0019
# 32	32.0000	0.0007	0.0023
# 64	64.0000	0.0012	0.0035
# 128	128.0000	0.0023	0.0062

# static np round True
# batching: 0.0005, 0.0008, 0.0012, 0.0022
# update: 0.0019, 0.0022, 0.0029, 0.0044
# 16	16.0000	0.0005	0.0019
# 32	32.0000	0.0008	0.0022
# 64	64.0000	0.0012	0.0029
# 128	128.0000	0.0022	0.0044

# """

# # Schnett - qm9 - GPU
# dynamic_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
# dynamic_qm9_schnett_gpu_batching_time = np.array([0.0006, 0.0010, 0.0018, 0.0034])*1000
# dynamic_qm9_schnett_gpu_batching_time_std = np.array([0.0009, 0.0016, 0.0030])*1000

# dynamic_qm9_schnett_gpu_update_time = np.array([0.0019, 0.0024, 0.0034, 0.0056])*1000

# # Static jnp
# static_jnp_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
# static_jnp_qm9_schnett_gpu_batching_time = np.array([0.0032, 0.0036, 0.0043, 0.0055])*1000
# static_jnp_qm9_schnett_gpu_batching_time_std = np.array([])*1000

# static_jnp_qm9_schnett_gpu_update_time = np.array([0.0046, 0.0052, 0.0065, 0.0095])*1000

# # Static np round True
# static_np_round_true_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
# static_np_round_true_qm9_schnett_gpu_batching_time = np.array([0.0005, 0.0008, 0.0012, 0.0022])*1000
# static_np_round_true_qm9_schnett_gpu_batching_time_std = np.array([0.0005, 0.0021])*1000

# static_np_qm9_round_true_schnett_gpu_update_time = np.array([0.0019, 0.0022, 0.0029, 0.0044])*1000

# # Static np round False
# static_np_round_false_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
# static_np_round_false_qm9_schnett_gpu_batching_time = np.array([0.0005, 0.0007, 0.0012, 0.0023])*1000
# static_np_round_false_qm9_schnett_gpu_batching_std = np.array([0.0000, 0.00002])*1000

# static_np_qm9_round_false_schnett_gpu_update_time = np.array([0.0019, 0.0023, 0.0035, 0.0062])*1000
# static_np_round_false_qm9_schnett_gpu_update_std = np.array([0.00003, 0.00002])*1000

# # Second plot is qm9 on GPU for SchNet.
# ax_gpu[0, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_batching_time,
#     'x', markersize=marker_size, alpha=1, color='k')

# ax_gpu[0, 1].plot(static_jnp_qm9_schnett_gpu_batch_size, static_jnp_qm9_schnett_gpu_batching_time,
#     '^', markersize=marker_size, alpha=1, color='r')

# ax_gpu[0, 1].plot(
#     static_np_round_false_qm9_schnett_gpu_batch_size, static_np_round_false_qm9_schnett_gpu_batching_time,
#     '>', markersize=marker_size, alpha=1, color='b')

# ax_gpu[0, 1].plot(static_np_round_true_qm9_schnett_gpu_batch_size,
#                        static_np_round_true_qm9_schnett_gpu_batch_size,
#     'v', markersize=marker_size, alpha=1, color='y')

# ax_gpu[0, 1].legend(
#     ['dynamic', 'static jnp', 'static np', 'static np round to 64'], fontsize=12, loc='upper left')

# ax_gpu[0, 1].set_xlim(0, 5)
# ax_gpu[0, 1].set_ylim(0, 10)

# ax_gpu[0, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
# ax_gpu[0, 1].set_ylabel('Time (ms)', fontsize=12)


# # Gradient Update plot
# ax_gpu[1, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_update_time,
#     '+', markersize=marker_size, alpha=1, color='k')
# ax_gpu[1, 1].plot(static_jnp_qm9_schnett_gpu_batch_size, static_jnp_qm9_schnett_gpu_update_time,
#     '^', markersize=marker_size, alpha=1, color='r')
# ax_gpu[1, 1].plot(static_np_round_false_qm9_schnett_gpu_batch_size,
#                      static_np_qm9_round_false_schnett_gpu_update_time,
#                      '>', markersize=marker_size, alpha=1, color='b')
# ax_gpu[1, 1].plot(static_np_round_true_qm9_schnett_gpu_batch_size,
#                      static_np_qm9_round_true_schnett_gpu_update_time,
#     'v', markersize=marker_size, alpha=1, color='y')

# plt.tight_layout()
# ax_gpu[1, 1].set_ylim(0, 10)
# ax_gpu[1, 1].set_xlim(0, 5)
# ax_gpu[1, 1].set_ylabel('Time (ms)', fontsize=12)
# ax_gpu[1, 1].set_xlabel('Batch size', fontsize=12)

# plt.tight_layout()
# ax_gpu[0, 1].set_ylim(0, 10)
# ax_gpu[0, 1].set_xlim(0, 5)
# # ax_gpu[1, 0].set_ylabel('Time (ms)', fontsize=12)
# ax_gpu[1, 1].set_xlabel('Batch size', fontsize=12)

# plt.show()


# ## CPU PLOTS - Schnett

# """
# SchNet AFLOW CPU
# dynamic
# 0.0011, 0.0020, 0.0036
# 0.0177, 0.0271, 0.0437
# 16	16.0000	NaN	NaN
# 32	32.0000	0.0011	0.0177
# 64	64.0000	0.0020	0.0271
# 128	128.0000	0.0036	0.0437

# static (jnp batch)
# 0.0011, 0.0016, 0.0022, 0.0034
# 0.0119, 0.0220, 0.0362, 0.0574
# 16	16.0000	0.0011	0.0119
# 32	32.0000	0.0016	0.0220
# 64	64.0000	0.0022	0.0362
# 128	128.0000	0.0034	0.0574

		
# static (np batch) round False

# 0.0006, 0.0026
# 0.0109, 0.0528
# 16	16.0000	0.0006	0.0109
# 32	32.0000	NaN	NaN
# 64	64.0000	NaN	NaN
# 128	128.0000	0.0026	0.0528

# static np round True
# 16	16.0000	NaN	NaN
# 32	32.0000	NaN	NaN
# 64	64.0000	NaN	NaN
# 128	128.0000	0.0025	0.0438
# """

# # Schnett - AFLOW - CPU
# dynamic_aflow_schnett_cpu_batch_size = [2, 3, 4]
# dynamic_aflow_schnett_cpu_batching_time = np.array([0.0011, 0.0020, 0.0036])*1000
# # dynamic_aflow_schnett_cpu_batching_time_std = np.array([0.0009, 0.0016, 0.0030])*1000

# dynamic_aflow_schnett_cpu_update_time = np.array([0.0177, 0.0271, 0.0437])*1000

# # Static jnp
# static_jnp_aflow_schnett_cpu_batch_size = [1, 2, 3, 4]
# static_jnp_aflow_schnett_cpu_batching_time = np.array([0.0011, 0.0016, 0.0022, 0.0034])*1000
# static_jnp_aflow_schnett_cpu_batching_time_std = np.array([])*1000

# static_jnp_aflow_schnett_cpu_update_time = np.array([0.0119, 0.0220, 0.0362, 0.0574])*1000

# # Static np round True
# static_np_round_true_aflow_schnett_cpu_batch_size = [4]
# static_np_round_true_aflow_schnett_cpu_batching_time = np.array([0.0025])*1000
# static_np_round_true_aflow_schnett_cpu_batching_time_std = np.array([])*1000

# static_np_aflow_round_true_schnett_cpu_update_time = np.array([0.0438])*1000

# # Static np round False
# static_np_round_false_aflow_schnett_cpu_batch_size = [1, 4]
# static_np_round_false_aflow_schnett_cpu_batching_time = np.array([0.0006, 0.0026])*1000
# static_np_round_false_aflow_schnett_cpu_batching_std = np.array([])*1000

# static_np_aflow_round_false_schnett_cpu_update_time = np.array([0.0109, 0.0528])*1000
# static_np_round_false_aflow_schnett_cpu_update_std = np.array([])*1000


# fig_cpu, ax_cpu = plt.subplots(2, 2, figsize=(7, 7)) #, gridspec_kw={'height_ratios': [1]})

# # First plot is batching: AFLOW - GPU - SchNett
# ax_cpu[0, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_batching_time,
#     'x', markersize=marker_size, alpha=1, color='k')

# ax_cpu[0, 0].plot(static_jnp_aflow_schnett_cpu_batch_size, static_jnp_aflow_schnett_cpu_batching_time,
#     '^', markersize=marker_size, alpha=1, color='r')

# ax_cpu[0, 0].plot(
#     static_np_round_false_aflow_schnett_cpu_batch_size, static_np_round_false_aflow_schnett_cpu_batching_time,
#     '>', markersize=marker_size, alpha=1, color='b')

# ax_cpu[0, 0].plot(static_np_round_true_aflow_schnett_cpu_batch_size,
#                        static_np_round_true_aflow_schnett_cpu_batch_size,
#     'v', markersize=marker_size, alpha=1, color='y')

# ax_cpu[0, 0].legend(
#     ['dynamic', 'static jnp', 'static np', 'static np round to 64'], fontsize=12, loc='upper left')

# ax_cpu[0, 0].set_xlim(0, 5)
# ax_cpu[0, 0].set_ylim(0, 10)

# ax_cpu[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
# ax_cpu[0, 0].set_ylabel('Time (ms)', fontsize=12)

# # Gradient Update plot
# ax_cpu[1, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_update_time,
#     '+', markersize=marker_size, alpha=1, color='k')
# ax_cpu[1, 0].plot(static_jnp_aflow_schnett_cpu_batch_size, static_jnp_aflow_schnett_cpu_update_time,
#     '^', markersize=marker_size, alpha=1, color='r')
# ax_cpu[1, 0].plot(static_np_round_false_aflow_schnett_cpu_batch_size,
#                      static_np_aflow_round_false_schnett_cpu_update_time,
#                      '>', markersize=marker_size, alpha=1, color='b')
# ax_cpu[1, 0].plot(static_np_round_true_aflow_schnett_cpu_batch_size,
#                      static_np_aflow_round_true_schnett_cpu_update_time,
#     'v', markersize=marker_size, alpha=1, color='y')

# ax_cpu[1, 0].set_ylim(0, 10)
# ax_cpu[1, 0].set_xlim(0, 5)
# ax_cpu[1, 0].set_ylabel('Time (ms)', fontsize=12)
# ax_cpu[1, 0].set_xlabel('Batch size', fontsize=12)

# plt.show()












# # """
# # Fourth Plot
# # SchNet qm9 CPU

# # 0.0007, 0.0012, 0.0024, 0.0042
# # 0.0132, 0.0178, 0.0423, 0.0693

# # dynamic
# # 16	16.0000	0.0007	0.0132
# # 32	32.0000	0.0012	0.0178
# # 64	64.0000	0.0024	0.0423
# # 128	128.0000	0.0042	0.0693

# # static (jnp)
# # 0.0012, 0.0018, 0.0026, 0.0039
# # 0.0164, 0.0370, 0.0582, 0.0978

# # 16	16.0000	0.0012	0.0164
# # 32	32.0000	0.0018	0.0370
# # 64	64.0000	0.0026	0.0582
# # 128	128.0000	0.0039	0.0978



# # static (np)
# # 16	16.0000	0.0006	0.0142
# # 32	32.0000	0.0011	0.0310
# # 64	64.0000	0.0018	0.0494
# # 128	128.0000 0.0030	0.0825
# # """
# # # Schnett - qm9 - CPU
# # dynamic_qm9_schnett_cpu_batch_size = [1, 2, 3, 4]
# # dynamic_qm9_schnett_cpu_batching_time = np.array([0.0007, 0.0012, 0.0024, 0.0042])*1000
# # dynamic_qm9_schnett_cpu_update_time = np.array([0.0132, 0.0178, 0.0423, 0.0693])*1000

# # static_qm9_schnett_cpu_batch_size = [1, 2, 3, 4]
# # static_qm9_schnett_cpu_batching_time = np.array([0.0012, 0.0018, 0.0026, 0.0039])*1000
# # static_qm9_schnett_cpu_update_time = np.array([0.0164, 0.0370, 0.0582, 0.0978])*1000




# # ax[0, 0].text(
# #     0.31, 0.62, 'SchNet - AFLOW - GPU', horizontalalignment='center',
# #     verticalalignment='center', transform=ax[0, 0].transAxes, fontsize=12)


# # # Second plot is AFLOW - GPU - SchNett
# # ax[0, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_batching_time,
# #     'x', markersize=marker_size, alpha=1, color='k')

# # ax[0, 1].plot(static_qm9_schnett_gpu_batch_size, static_qm9_schnett_gpu_batching_time,
# #     '^', markersize=marker_size, alpha=1, color='r')

# # ax[0, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_update_time,
# #     '+', markersize=marker_size, alpha=1, color='b')

# # ax[0, 1].plot(static_qm9_schnett_gpu_batch_size, static_qm9_schnett_gpu_update_time,
# #     'v', markersize=marker_size, alpha=1, color='y')
# # # ax[0, 1].legend(
# # #     ['dynamic batching', 'static batching', 'dynamic update', 'static update'], fontsize=12)

# # ax[0, 1].set_xlim(0, 5)
# # # ax[0, 1].set_ylim(0, 10)
# # ax[0, 1].set_yscale('log')
# # ax[0, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

# # ax[0, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)


# # # Third plot is AFLOW - SchNet - CPU
# # ax[1, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_batching_time,
# #     'x', markersize=marker_size, alpha=1, color='k')

# # ax[1, 0].plot(static_aflow_schnett_cpu_batch_size, static_aflow_schnett_cpu_batching_time,
# #     '^', markersize=marker_size, alpha=1, color='r')

# # ax[1, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_update_time,
# #     '+', markersize=marker_size, alpha=1, color='b')

# # ax[1, 0].plot(static_aflow_schnett_cpu_batch_size, static_aflow_schnett_cpu_update_time,
# #     'v', markersize=marker_size, alpha=1, color='y')
# # # ax[1, 0].legend(
# # #     ['dynamic batching', 'static batching', 'dynamic update', 'static update'],
# # #     loc='upper left', fontsize=12)



# # ax[1, 0].set_xlim(0, 5)

# # ax[1, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
# # ax[1, 0].set_xlabel('Batch size', fontsize=12)
# # ax[1, 0].set_ylabel('Time (ms)', fontsize=12)
# # ax[1, 0].set_yscale('log')
# # ax[1, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
# # ax[1, 0].set_ylim(1E-1, 1E3)

# # ### Fouth plot is qm9 - SchNet - CPU
# # ax[1, 1].plot(dynamic_qm9_schnett_cpu_batch_size, dynamic_qm9_schnett_cpu_batching_time,
# #     'x', markersize=marker_size, alpha=1, color='k')

# # ax[1, 1].plot(static_qm9_schnett_cpu_batch_size, static_qm9_schnett_cpu_batching_time,
# #     '^', markersize=marker_size, alpha=1, color='r')

# # ax[1, 1].plot(dynamic_qm9_schnett_cpu_batch_size, dynamic_qm9_schnett_cpu_update_time,
# #     '+', markersize=marker_size, alpha=1, color='b')

# # ax[1, 1].plot(static_qm9_schnett_cpu_batch_size, static_qm9_schnett_cpu_update_time,
# #     'v', markersize=marker_size, alpha=1, color='y')
# # # ax[1, 1].legend(
# # #     ['dynamic batching', 'static batching', 'dynamic update', 'static update'],
# # #     loc='upper left', fontsize=12)



# # ax[1, 1].set_xlim(0, 5)
# # # ax[1, 1].set_ylim(0, 100)
# # ax[1, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
# # ax[1, 1].set_xlabel('Batch size', fontsize=12)
# # ax[1, 1].set_yscale('log')
# # ax[1, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

# # ax[1, 1].set_ylim(1E-1, 1E3)

# # handles, labels = ax[1, 1].get_legend_handles_labels()
# # fig.legend(handles, labels, loc='upper center')


# # plt.tight_layout()
# # plt.savefig(
# #     '/home/dts/Documents/theory/batching_paper/mpeu_qm9_aflow_100k_dynamic_static.png',
# #     bbox_inches='tight', dpi=600)
# # plt.show()
