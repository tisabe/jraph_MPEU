# Plot batching times.
""" First Plot
AFLOW AFLOW GPU

Dynamic
16	16.0000	NaN	NaN
32	32.0000	0.0009	0.0023
64	64.0000	0.0016	0.0030
128	128.0000	0.0030	0.0047

Static (jnp)
16	16.0000	0.0029	0.0044
32	32.0000	0.0033	0.0049
64	64.0000	0.0040	0.0057
128	128.0000	0.0051	0.0074

Static (np)
16	16.0000	0.0005	0.0022
32	32.0000	NaN	NaN
64	64.0000	NaN	NaN
128	128.0000	0.0021	0.0076
"""

import numpy as np
import matplotlib.pyplot as plt
marker_size = 11

# Schnett - AFLOW - GPU
dynamic_aflow_schnett_gpu_batch_size = [2, 3, 4]
dynamic_aflow_schnett_gpu_batching_time = np.array([0.0009, 0.0016, 0.0030])*1000
dynamic_aflow_schnett_gpu_update_time = np.array([0.0023, 0.0030, 0.0047])*1000

static_aflow_schnett_gpu_batch_size = [1, 2, 3, 4]
static_aflow_schnett_gpu_batching_time = np.array([0.0029, 0.0033, 0.0040, 0.0051])*1000
static_aflow_schnett_gpu_update_time = np.array([0.0023, 0.0030, 0.0047, 0.0074])*1000


""" Second Plot
SchNet Qm9 GPU
dynamic
0.0006, 0.0010, 0.0018, 0.0034
0.0019, 0.0024, 0.0034, 0.0056

16	16.0000	0.0006	0.0019
32	32.0000	0.0010	0.0024
64	64.0000	0.0018	0.0034
128	128.0000	0.0034	0.0056

static
0.0032, 0.0036, 0.0043, 0.0055
0.0046, 0.0052, 0.0065, 0.0095

new_static (np not jnp batch)
batching: 0.0005, 0.0008, 0.0012, 0.0022
update: 0.0019, 0.0023, 0.0032, 0.0053

16	16.0000	0.0032	0.0046
32	32.0000	0.0036	0.0052
64	64.0000	0.0043	0.0065
128	128.0000	0.0055	0.0095

16	16.0000	0.0005	0.0019
32	32.0000	0.0008	0.0023
64	64.0000	0.0012	0.0032
128	128.0000	0.0022	0.0053

"""

# Schnett - Qm9 - GPU
dynamic_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
dynamic_qm9_schnett_gpu_batching_time = np.array([0.0006, 0.0010, 0.0018, 0.0034])*1000
dynamic_qm9_schnett_gpu_update_time = np.array([0.0019, 0.0024, 0.0034, 0.0056])*1000

static_qm9_schnett_gpu_batch_size = [1, 2, 3, 4]
static_qm9_schnett_gpu_batching_time = np.array([0.0032, 0.0036, 0.0043, 0.0055])*1000
static_qm9_schnett_gpu_update_time = np.array([0.0046, 0.0052, 0.0065, 0.0095])*1000


"""
Third Plot
SchNet AFLOW CPU

dynamic
16	16.0000	NaN	NaN
32	32.0000	0.0011	0.0177
64	64.0000	0.0020	0.0271
128	128.0000	0.0036	0.0437
static (jnp batch)
16	16.0000	0.0011	0.0119
32	32.0000	0.0016	0.0220
64	64.0000	0.0022	0.0362
128	128.0000	0.0034	0.0574

		
static (np batch)

16	16.0000	0.0006	0.0109
32	32.0000	NaN	NaN
64	64.0000	NaN	NaN
128	128.0000	0.0026	0.0528
"""
# Schnett - AFLOW - CPU
dynamic_aflow_schnett_cpu_batch_size = [2, 3, 4]
dynamic_aflow_schnett_cpu_batching_time = np.array([0.0011, 0.0020, 0.0036])*1000
dynamic_aflow_schnett_cpu_update_time = np.array([0.0177, 0.0271, 0.0437])*1000

static_aflow_schnett_cpu_batch_size = [1, 2, 3, 4]
static_aflow_schnett_cpu_batching_time = np.array([0.0011, 0.0016, 0.0022, 0.0034])*1000
static_aflow_schnett_cpu_update_time = np.array([0.0119, 0.0220, 0.0362, 0.0574])*1000

"""
Fourth Plot
SchNet qm9 CPU

0.0007, 0.0012, 0.0024, 0.0042
0.0132, 0.0178, 0.0423, 0.0693

dynamic
16	16.0000	0.0007	0.0132
32	32.0000	0.0012	0.0178
64	64.0000	0.0024	0.0423
128	128.0000	0.0042	0.0693

static (jnp)
0.0012, 0.0018, 0.0026, 0.0039
0.0164, 0.0370, 0.0582, 0.0978

16	16.0000	0.0012	0.0164
32	32.0000	0.0018	0.0370
64	64.0000	0.0026	0.0582
128	128.0000	0.0039	0.0978



static (np)
16	16.0000	0.0006	0.0142
32	32.0000	0.0011	0.0310
64	64.0000	0.0018	0.0494
128	128.0000 0.0030	0.0825
"""
# Schnett - qm9 - CPU
dynamic_qm9_schnett_cpu_batch_size = [1, 2, 3, 4]
dynamic_qm9_schnett_cpu_batching_time = np.array([0.0007, 0.0012, 0.0024, 0.0042])*1000
dynamic_qm9_schnett_cpu_update_time = np.array([0.0132, 0.0178, 0.0423, 0.0693])*1000

static_qm9_schnett_cpu_batch_size = [1, 2, 3, 4]
static_qm9_schnett_cpu_batching_time = np.array([0.0012, 0.0018, 0.0026, 0.0039])*1000
static_qm9_schnett_cpu_update_time = np.array([0.0164, 0.0370, 0.0582, 0.0978])*1000




# MPEU Plots
fig, ax = plt.subplots(2, 2, figsize=(7, 7)) #, gridspec_kw={'height_ratios': [1]})

# First plot is AFLOW - GPU - SchNett
ax[0, 0].plot(dynamic_aflow_schnett_gpu_batch_size, dynamic_aflow_schnett_gpu_batching_time,
    'x', markersize=marker_size, alpha=1, color='k')

ax[0, 0].plot(static_aflow_schnett_gpu_batch_size, static_aflow_schnett_gpu_batching_time,
    '^', markersize=marker_size, alpha=1, color='r')

ax[0, 0].plot(dynamic_aflow_schnett_gpu_batch_size, dynamic_aflow_schnett_gpu_update_time,
    '+', markersize=marker_size, alpha=1, color='b')

ax[0, 0].plot(static_aflow_schnett_gpu_batch_size, static_aflow_schnett_gpu_update_time,
    'v', markersize=marker_size, alpha=1, color='y')
ax[0, 0].legend(
    ['dynamic batching', 'static batching', 'dynamic update', 'static update'], fontsize=12, loc='upper left')
# ax[0, 0].text(
#     0.31, 0.62, 'SchNet - AFLOW - GPU', horizontalalignment='center',
#     verticalalignment='center', transform=ax[0, 0].transAxes, fontsize=12)

ax[0, 0].set_xlim(0, 5)
ax[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
ax[0, 0].set_ylabel('Time (ms)', fontsize=12)
ax[0, 0].set_yscale('log')
ax[0, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

# Second plot is AFLOW - GPU - SchNett
ax[0, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_batching_time,
    'x', markersize=marker_size, alpha=1, color='k')

ax[0, 1].plot(static_qm9_schnett_gpu_batch_size, static_qm9_schnett_gpu_batching_time,
    '^', markersize=marker_size, alpha=1, color='r')

ax[0, 1].plot(dynamic_qm9_schnett_gpu_batch_size, dynamic_qm9_schnett_gpu_update_time,
    '+', markersize=marker_size, alpha=1, color='b')

ax[0, 1].plot(static_qm9_schnett_gpu_batch_size, static_qm9_schnett_gpu_update_time,
    'v', markersize=marker_size, alpha=1, color='y')
# ax[0, 1].legend(
#     ['dynamic batching', 'static batching', 'dynamic update', 'static update'], fontsize=12)

ax[0, 1].set_xlim(0, 5)
# ax[0, 1].set_ylim(0, 10)
ax[0, 1].set_yscale('log')
ax[0, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

ax[0, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)


# Third plot is AFLOW - SchNet - CPU
ax[1, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_batching_time,
    'x', markersize=marker_size, alpha=1, color='k')

ax[1, 0].plot(static_aflow_schnett_cpu_batch_size, static_aflow_schnett_cpu_batching_time,
    '^', markersize=marker_size, alpha=1, color='r')

ax[1, 0].plot(dynamic_aflow_schnett_cpu_batch_size, dynamic_aflow_schnett_cpu_update_time,
    '+', markersize=marker_size, alpha=1, color='b')

ax[1, 0].plot(static_aflow_schnett_cpu_batch_size, static_aflow_schnett_cpu_update_time,
    'v', markersize=marker_size, alpha=1, color='y')
# ax[1, 0].legend(
#     ['dynamic batching', 'static batching', 'dynamic update', 'static update'],
#     loc='upper left', fontsize=12)



ax[1, 0].set_xlim(0, 5)

ax[1, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
ax[1, 0].set_xlabel('Batch size', fontsize=12)
ax[1, 0].set_ylabel('Time (ms)', fontsize=12)
ax[1, 0].set_yscale('log')
ax[1, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
ax[1, 0].set_ylim(1E-1, 1E3)

### Fouth plot is qm9 - SchNet - CPU
ax[1, 1].plot(dynamic_qm9_schnett_cpu_batch_size, dynamic_qm9_schnett_cpu_batching_time,
    'x', markersize=marker_size, alpha=1, color='k')

ax[1, 1].plot(static_qm9_schnett_cpu_batch_size, static_qm9_schnett_cpu_batching_time,
    '^', markersize=marker_size, alpha=1, color='r')

ax[1, 1].plot(dynamic_qm9_schnett_cpu_batch_size, dynamic_qm9_schnett_cpu_update_time,
    '+', markersize=marker_size, alpha=1, color='b')

ax[1, 1].plot(static_qm9_schnett_cpu_batch_size, static_qm9_schnett_cpu_update_time,
    'v', markersize=marker_size, alpha=1, color='y')
# ax[1, 1].legend(
#     ['dynamic batching', 'static batching', 'dynamic update', 'static update'],
#     loc='upper left', fontsize=12)



ax[1, 1].set_xlim(0, 5)
# ax[1, 1].set_ylim(0, 100)
ax[1, 1].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)
ax[1, 1].set_xlabel('Batch size', fontsize=12)
ax[1, 1].set_yscale('log')
ax[1, 1].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)

ax[1, 1].set_ylim(1E-1, 1E3)

handles, labels = ax[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')


plt.tight_layout()
plt.savefig(
    '/home/dts/Documents/theory/batching_paper/mpeu_qm9_aflow_100k_dynamic_static.png',
    bbox_inches='tight', dpi=600)
plt.show()
