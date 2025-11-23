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
from absl import app
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from matplotlib import rc, font_manager

# import matplotlib as mpl
# label_size = 12
# mpl.rcParams['xtick.labelsize'] = label_size 


BASE_DIR = '/home/dts/Documents/hu/jraph_MPEU/batch_data'
## The 2 million profiling steps data:
# Used for MPEU/SchNet
# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_19_01_2025.csv'
MPEU_SCHNET_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_19_01_2025.csv'

# Used for PaiNN
COMBINED_CSV = 'parsed_profiling_painn_batching_2_000_000_steps_15_05_2025.csv'
PAINN_CSV = 'parsed_profiling_painn_batching_2_000_000_steps_15_05_2025.csv'

# COMBINED_CSV = 'parsed_profiling_painn_cpu_qm9_aflow_schnet_mpeu_100k_steps_12_55__21_06_2025.csv'

# COMBINED_CSV = 'parsed_profiling_static_batching_seb_fix_qm9_aflow_schnet_mpeu_100k_steps_11_31__23_12_2024.csv'

# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_aflow_qm9_20_12_2024.csv'

BATCH_SIZE_DICT = {
    '16': 0,
    '32': 1,
    '64': 2,
    '128': 3,
}

BATCH_SIZE_LIST = [16, 32, 64, 128]

# MODEL_TYPE_LIST = ['schnet', 'mpeu']
MODEL_TYPE_LIST = ['schnet', 'MPEU']

# BATCH_METHOD_LIST = ['dynamic', 'static']
BATCH_METHOD_LIST = ['dynamic', 'static', 'static-64']
# BATCH_METHOD_LIST = ['static-2']

# COMPUTING_TYPE_LIST = ['gpu_a100', 'cpu']
COMPUTING_TYPE_LIST = ['gpu_a100']

DATASET_LIST = ['aflow']

FONTSIZE = 12
# FONT = 'Times'
# FONT = 'Times new roman'
FONT = 'serif'

fontProperties = {'family':'sans-serif','sans-serif':['Times'],
    'weight' : 'normal', 'size' : FONTSIZE+100}
ticks_font = font_manager.FontProperties(family='Times', style='normal',
    size=FONTSIZE, weight='normal', stretch='normal')
# rc('text', usetex=True)
rc('text')
rc('font',**fontProperties)


def plot_six_columns_models_datasets(df_list, models_order, datasets_order=['aflow', 'qm9'], compute_type='gpu_a100', mean_or_median='mean'):
    """
    Plots batching, update, and combined times for multiple models (SchNet, MPEU, PaiNN)
    across two datasets (AFLOW, QM9) in a 3x6 grid.
    """

    plt.rcParams['font.family'] = FONT
    plt.rcParams['font.size'] = FONTSIZE + 3
    plt.rcParams['axes.labelsize'] = FONTSIZE
    plt.rcParams['xtick.labelsize'] = FONTSIZE + 1
    plt.rcParams['ytick.labelsize'] = FONTSIZE + 4
    plt.rcParams['legend.fontsize'] = FONTSIZE +100
    plt.rcParams['axes.titlesize'] = FONTSIZE + 2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5

    plot_content_rows = ['batching', 'update', 'combined']

    n_models = len(models_order)
    n_datasets = len(datasets_order)
    total_plot_cols = n_models * n_datasets # This is 6 in your case

    # Define the width ratios for columns including explicit spacers
    # Each actual plot column gets a ratio of 1.
    # Spacer columns get a ratio of, for example, 0.2 or 0.5 depending on desired width.
    # In your case: Col1, Col2, Col3, Col4, SPACER_1, Col5, SPACER_2, Col6
    # So we have 6 plot columns + 2 spacer columns = 8 effective columns in GridSpec
    
    # Original columns: 0, 1, 2, 3, 4, 5
    # We want space between 3 and 4 (using 0-indexed) -> between (Col4) and (Col5)
    # And space between 4 and 5 -> between (Col5) and (Col6)
    
    # Let's map to grid spec indices:
    # 0 (Col1)
    # 1 (Col2)
    # 2 (Col3)
    # 3 (Col4)
    # 4 (Spacer 1)
    # 5 (Col5)
    # 6 (Spacer 2)
    # 7 (Col6)

    # width_ratios = [1, 1, 1, 1, 0.5, 1, 0.5, 1] # Example: 0.5 units for space
    # The problem asks for space between columns 4 and 5, and columns 5 and 6
    # If the user means 1-indexed, then:
    # Col 1, Col 2, Col 3, Col 4, [SPACE], Col 5, [SPACE], Col 6
    # This implies 3 spacers, not 2.
    # Let's re-read carefully: "vertical space between columns 4 and 5, and vertical space between columns 5 and 6"
    # This means the *visual gaps* should be there.

    # Let's assume you have 6 logical columns.
    # Col0 | Col1 | Col2 | Col3 | Col4 | Col5
    # You want space between Col3 and Col4 (your "4 and 5" if 1-indexed)
    # And space between Col4 and Col5 (your "5 and 6" if 1-indexed)

    # Grid arrangement:
    # Plot_Col_0, Plot_Col_1, Plot_Col_2, Plot_Col_3, Spacer_1, Plot_Col_4, Spacer_2, Plot_Col_5
    
    # Total GridSpec columns will be: 6 (actual plot columns) + 2 (spacers) = 8
    gs_cols = total_plot_cols + 2
    width_ratios = [1] * total_plot_cols # Start with all plot columns having equal width

    # Insert spacer ratios at the desired positions
    # Spacer 1: after column index 3 (which is the 4th logical column)
    width_ratios.insert(4, 0.2) # Insert a spacer after the 4th logical column (index 3)
    
    # Spacer 2: after column index 5 (which is the 5th logical column, after the first spacer)
    # After inserting the first spacer, the 5th logical column (original index 4) is now at index 5.
    width_ratios.insert(6, 0.2) # Insert a spacer after the 5th logical column (index 5 of the growing list)

    figsize_x = 14 if n_models > 1 else 5
    fig = plt.figure(figsize=(figsize_x, 8))
    
    # Create a GridSpec with the custom width ratios
    gs = fig.add_gridspec(len(plot_content_rows), gs_cols, width_ratios=width_ratios, wspace=0.0) # wspace to 0 as spacing is via ratios

    # Create an array of axes to mimic the original subplots behavior
    ax = np.empty((len(plot_content_rows), total_plot_cols), dtype=object)

    # Map logical column indices to GridSpec column indices
    gs_col_map = []
    current_gs_col = 0
    for i in range(total_plot_cols):
        gs_col_map.append(current_gs_col)
        current_gs_col += 1 # Move past the plot column
        if i == 3: # After logical column 3 (4th plot column, before 5th plot column)
            current_gs_col += 1 # Skip the first spacer
        if i == 4: # After logical column 4 (5th plot column, before 6th plot column)
            current_gs_col += 1 # Skip the second spacer

    # Now create the actual axes for plotting
    for row_idx in range(len(plot_content_rows)):
        for logical_col_idx in range(total_plot_cols):
            gs_idx = gs_col_map[logical_col_idx]
            ax[row_idx, logical_col_idx] = fig.add_subplot(gs[row_idx, gs_idx])

    # Ensure ax is always 2D for consistency as in original code
    if len(plot_content_rows) == 1 and total_plot_cols == 1:
        ax = np.array([[ax[0,0]]]) # Adjust for single plot creation
    elif len(plot_content_rows) == 1:
        ax = np.array([ax[0,:]])
    elif total_plot_cols == 1:
        ax = np.array([[a] for a in ax[:,0]])


    # Define common plotting elements
    color_list = ['#1f77b4', '#ff7f0e', '#9467bd'] # For different batch methods
    marker_list = ['x', '^', '.']

    xlim = [0, 140]

    ylim_settings = {
        ('schnet', 'batching'): 10, ('schnet', 'update'): 10, ('schnet', 'combined'): 10,
        ('MPEU', 'batching'): 10, ('MPEU', 'update'): 10, ('MPEU', 'combined'): 10,
        ('painn', 'batching'): 10, ('painn', 'update'): 40, ('painn', 'combined'): 40,
        'cpu_batching': 200, 'cpu_update': 200, 'cpu_combined': 400
    }

    ylabels_settings = {
        ('schnet', 'batching'): [0, 2, 4, 6, 8, 10],
        ('schnet', 'update'): [0, 2, 4, 6, 8, 10],
        ('schnet', 'combined'): [0, 2, 4, 6, 8, 10],
        ('MPEU', 'batching'): [0, 2, 4, 6, 8, 10],
        ('MPEU', 'update'): [0, 2, 4, 6, 8, 10],
        ('MPEU', 'combined'): [0, 2, 4, 6, 8, 10],
        ('painn', 'batching'): [0, 2, 4, 6, 8, 10],
        ('painn', 'update'): [0, 10, 20, 30, 40],
        ('painn', 'combined'): [0, 10, 20, 30, 40],
        'cpu_labels': [0, 50, 100, 150, 200]
    }

    font_props = font_manager.FontProperties(family=FONT, style='normal', size=FONTSIZE+5)
    legend_font_props = font_manager.FontProperties(family=FONT, style='normal', size=FONTSIZE+3)
    label_font_props = font_manager.FontProperties(family=FONT, style='normal', size=FONTSIZE+4)

    for model_idx, model in enumerate(models_order):
        if model in ['schnet', 'MPEU']:
            df = df_list[0]
        else:
            df = df_list[1]
        if model == 'painn':
            model_label = 'PaiNN'
        elif model == 'schnet':
            model_label = 'SchNet'
        else:
            model_label = model

        for dataset_idx, dataset in enumerate(datasets_order):
            current_global_col_idx = model_idx * n_datasets + dataset_idx

            current_ax_title = ax[0, current_global_col_idx]
            current_ax_title.set_title(f'{dataset.upper()}', font=font_props, fontsize=FONTSIZE+3)

            if dataset_idx == 0:
                # Need to calculate position carefully because of explicit spacers
                # Assuming models_order is 3 elements, so 3*2=6 plot columns
                # e.g., SchNet AFLOW (col 0), SchNet QM9 (col 1)
                # MPEU AFLOW (col 2), MPEU QM9 (col 3)
                # PaiNN AFLOW (col 4), PaiNN QM9 (col 5)

                # The x position for the super-title should span two logical columns.
                # The total width is sum(width_ratios). Calculate normalized center.
                
                # For SchNet (cols 0, 1):
                if model_idx == 0:
                    gs_col_start = gs_col_map[0]
                    gs_col_end = gs_col_map[1]
                    center_pos = (gs_col_start + gs_col_end + 1) / (gs_cols * 2) # approx center
                    fig.text(0.2, 0.93, # Estimate
                             f'{model_label}', ha='center', va='bottom',
                             fontproperties=font_manager.FontProperties(family=FONT, size=FONTSIZE+3, weight='bold'))
                # For MPEU (cols 2, 3):
                elif model_idx == 1:
                    gs_col_start = gs_col_map[2]
                    gs_col_end = gs_col_map[3]
                    fig.text(0.49, 0.93, # Estimate
                             f'{model_label}', ha='center', va='bottom',
                             fontproperties=font_manager.FontProperties(family=FONT, size=FONTSIZE+3, weight='bold'))
                # For PaiNN (cols 4, 5):
                elif model_idx == 2:
                    gs_col_start = gs_col_map[4]
                    gs_col_end = gs_col_map[5]
                    # fig.text((gs_col_map[4] + gs_col_map[5] + 1.25) / gs_cols, 0.95, # Estimate, add 2 for 2 spacers
                    fig.text(0.82, 0.93, # Estimate, add 2 for 2 spacers
                             f'{model_label}', ha='center', va='bottom',
                             fontproperties=font_manager.FontProperties(family=FONT, size=FONTSIZE+3, weight='bold'))


            for row_idx, profile_column in enumerate(plot_content_rows):
                current_ax = ax[row_idx, current_global_col_idx]

                if compute_type == 'cpu':
                    current_ylim = ylim_settings[f'cpu_{profile_column}']
                    current_ylabels = ylabels_settings['cpu_labels']
                else:
                    current_ylim = ylim_settings.get((model, profile_column), 10)
                    current_ylabels = ylabels_settings.get((model, profile_column), [0, 5, 10])


                for color_counter, batch_method_original in enumerate(BATCH_METHOD_LIST):
                    y_mean_list = []
                    y_std_list = []
                    label = batch_method_original
                    batching_round_to_64 = False

                    if batch_method_original == 'static-64':
                        batch_method_filter = 'static'
                        batching_round_to_64 = True
                        label = 'static-$64$'
                    elif batch_method_original == 'static':
                        batch_method_filter = 'static'
                        batching_round_to_64 = False
                        label = 'static-$2^N$'
                    else:
                        batch_method_filter = batch_method_original
                        batching_round_to_64 = False

                    for batch_size in BATCH_SIZE_LIST:
                        y_mean, y_std = get_avg_std_of_profile_column(
                            df, profile_column, model, batch_method_filter, compute_type,
                            batch_size, dataset, batching_round_to_64, mean_or_median,
                            combined=(profile_column=='combined')
                        )
                        y_mean_list.append(y_mean)
                        y_std_list.append(y_std)

                    y_mean_array = np.array(y_mean_list)
                    y_std_array = np.array(y_std_list)
                    valid_indices = ~np.isnan(y_mean_array) & ~np.isnan(y_std_array)

                    if np.any(valid_indices):
                        current_ax.errorbar(np.array(BATCH_SIZE_LIST)[valid_indices],
                                             np.multiply(y_mean_array[valid_indices], 1000),
                                             yerr=y_std_array[valid_indices],
                                             marker=marker_list[color_counter],
                                             markersize=11, alpha=0.9,
                                             color=color_list[color_counter],
                                             label=label, linestyle='')

                current_ax.set_ylim(0, current_ylim)
                current_ax.set_xlim(xlim[0], xlim[1])
                current_ax.set_xticks(BATCH_SIZE_LIST)

                current_ax.set_yticks(current_ylabels, font=label_font_props, fontsize=FONTSIZE+4) # Set tick locations
                if current_global_col_idx == 0:
                    current_ax.set_ylabel(f'{profile_column.capitalize()} time (ms)', font=font_props, fontsize=FONTSIZE+4)
                    # current_ax.set_yticks(current_ylabels, fontsize=FONTSIZE+2)
                    current_ax.tick_params(axis='y', length=5)
                elif model == 'painn' and profile_column != "batching": # Note: 'is not' for strings is tricky, better to use '!='
                    current_ax.set_yticklabels(current_ylabels, font=label_font_props, fontsize=FONTSIZE+4, rotation=0)
                    current_ax.set_ylabel('')
                    current_ax.tick_params(axis='y', length=4)
                else:
                    current_ax.set_yticklabels([])
                    current_ax.tick_params(axis='y', length=5)

                # if row_idx == len(plot_content_rows) - 1:
                #     current_ax.set_xticklabels(BATCH_SIZE_LIST, font=font_props, fontsize=FONTSIZE+2, rotation=0)
                #     current_ax.set_xlabel('Batch size', fontsize=FONTSIZE+3, font=font_props)
                #     current_ax.tick_params(axis='x', rotation=0)
                # else:
                #     current_ax.set_xticklabels([])
                if row_idx == len(plot_content_rows) - 1:
                    # Set the tick labels first
                    labels = current_ax.set_xticklabels(BATCH_SIZE_LIST, font=font_props, fontsize=FONTSIZE+4, rotation=0)
                    current_ax.set_xlabel('Batch size', fontsize=FONTSIZE+4, font=font_props)
                    current_ax.tick_params(axis='x', rotation=0)

                    # Find the indices for '16' and '32' in BATCH_SIZE_LIST
                    try:
                        idx_16 = BATCH_SIZE_LIST.index(16)
                        idx_32 = BATCH_SIZE_LIST.index(32)

                        # Adjust the horizontal alignment and position for '16'
                        # labels[idx_16] is the Text object for '16'
                        # You might need to experiment with the x-offset value (e.g., -0.2, -0.1)
                        # The `transform=current_ax.get_xaxis_transform()` is important
                        # for moving it relative to the axis data coordinates.
                        
                        # Get the current tick position for 16 and 32
                        tick_locs = current_ax.get_xticks()

                        # Shift '16' to the left
                        # labels[idx_16].set_ha('right') # Horizontal alignment to right
                        # Shift its actual position slightly. You'll need to experiment with the offset.
                        # The offset is in data coordinates.
                        labels[idx_16].set_x(tick_locs[idx_16] + 5) # Move 5 units to the left (adjust as needed)

                        # Shift '32' to the right
                        labels[idx_32].set_ha('left') # Horizontal alignment to left
                        labels[idx_32].set_x(tick_locs[idx_32] - 50) # Move 5 units to the right (adjust as needed)

                    except ValueError:
                        print("Batch sizes 16 or 32 not found in BATCH_SIZE_LIST. Cannot adjust their positions.")

                else:
                    current_ax.set_xticklabels([])



    handles, labels = ax[0, 0].get_legend_handles_labels()
    if handles:
        if n_models == 2:
            fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.06, 0.87),
                    prop=font_props, edgecolor="black", fancybox=False)
            print('here')
        else:
            fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.057, 0.885),
                    prop=legend_font_props, edgecolor="black", fancybox=False)
            print('not 2 models')

    try:
        plt.style.use(["science", "grid"])
    except ImportError:
        print("SciencePlots not installed. Plotting without it.")

    fig.align_labels()
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    filename = f'/home/dts/Documents/theory/batching_paper/figs/profiling_6col_{compute_type}_{mean_or_median}.png'
    plt.savefig(filename, dpi=600)
    plt.show()


# def plot_multiple_models_subplot(df_list, models_to_plot, compute_type, mean_or_median):
#     """
#     Plots batching, update, and combined times for multiple models (SchNet, MPEU, PaiNN)
#     in a 3x3 grid.
#     """
#     # Define plot content for rows
#     plot_content_rows = ['batching', 'update', 'combined'] # These correspond to rows 0, 1, 2

#     fig, ax = plt.subplots(len(plot_content_rows)*2, len(models_to_plot), figsize=(15, 12), sharex=True) # Shared X-axis

#     # Define common plotting elements
#     color_list = ['#1f77b4', '#ff7f0e', '#9467bd'] # For different batch methods
#     marker_list = ['x', '^', '.']

#     xlim = [0, 140] # Default for batch sizes
#     # Global y-limits and labels, these will be set per row (profile_type)
#     # The actual values will depend on your data, these are starting points
#     ylim_batching = {'schnet': 8, 'MPEU': 10, 'painn': 20, 'cpu': 200}
#     ylim_update = {'schnet': 8, 'MPEU': 10, 'painn': 30, 'cpu': 200}
#     ylim_combined = {'schnet': 15, 'MPEU': 20, 'painn': 50, 'cpu': 400} # Combined should be sum of batching and update

#     # Set up global font properties
#     font_props = font_manager.FontProperties(family=FONT, style='normal', size=FONTSIZE)

#     # Loop through each model to fill its column
#     for col_idx, model in enumerate(models_to_plot):
#         model_label = model # Default, can be refined below
#         df = df_list[0]
#         if model == 'schnet':
#             model_label = 'SchNet'
#         elif model == 'MPEU':
#             model_label = 'MPEU'
#         elif model == 'painn':
#             model_label = 'PaiNN'
#             df_list = df_list[1]
#         if col_idx % 2 == 0:
#             data_label = 'QM9'
#             dataset = 'qm9'
#         else:
#             data_label = 'AFLOW'
#             dataset = 'aflow'

#         ax[0, col_idx].set_title(data_label, font=font_props, fontsize=FONTSIZE)

#         # Loop through each profile type (row) for the current model
#         for row_idx, profile_column in enumerate(plot_content_rows):
#             current_ax = ax[row_idx, col_idx]
#             # dataset = 'qm9' # Assuming QM9 for this example, adjust if you have AFLOW as well
#             #                 # If you want both AFLOW and QM9, you'd need another dimension or separate plots.

#             # Determine appropriate y-limits for the current row/profile type
#             if profile_column == 'batching':
#                 if compute_type == 'cpu':
#                     current_ylim = ylim_batching['cpu']
#                     current_ylabels = [0, 50, 100, 150, 200]
#                 else:
#                     current_ylim = ylim_batching[model]
#                     current_ylabels = [0, 2, 4, 6, 8] if model == 'schnet' else ([0, 2, 4, 6, 8, 10] if model == 'MPEU' else [0, 5, 10, 15, 20])
#             elif profile_column == 'update':
#                 if compute_type == 'cpu':
#                     current_ylim = ylim_update['cpu']
#                     current_ylabels = [0, 50, 100, 150, 200]
#                 else:
#                     current_ylim = ylim_update[model]
#                     current_ylabels = [0, 2, 4, 6, 8] if model == 'schnet' else ([0, 2, 4, 6, 8, 10] if model == 'MPEU' else [0, 10, 20, 30])
#             else: # combined
#                 if compute_type == 'cpu':
#                     current_ylim = ylim_combined['cpu']
#                     current_ylabels = [0, 100, 200, 300, 400]
#                 else:
#                     current_ylim = ylim_combined[model]
#                     current_ylabels = [0, 5, 10, 15] if model == 'schnet' else ([0, 5, 10, 15, 20] if model == 'MPEU' else [0, 10, 20, 30, 40, 50])


#             # --- Plotting data for each batch method ---
#             for color_counter, batch_method_original in enumerate(BATCH_METHOD_LIST):
#                 y_mean_list = []
#                 y_std_list = []
#                 label = batch_method_original
#                 batching_round_to_64 = False # Default

#                 # Adjust batch_method and label for filtering
#                 if batch_method_original == 'static-64':
#                     batch_method_filter = 'static'
#                     batching_round_to_64 = True
#                     label = 'static-$64$'
#                 elif batch_method_original == 'static':
#                     batch_method_filter = 'static'
#                     batching_round_to_64 = False
#                     label = 'static-$2^N$'
#                 else: # e.g., 'dynamic'
#                     batch_method_filter = batch_method_original
#                     batching_round_to_64 = False

#                 for batch_size in BATCH_SIZE_LIST:
#                     y_mean, y_std = get_avg_std_of_profile_column(
#                         df, profile_column, model, batch_method_filter, compute_type,
#                         batch_size, dataset, batching_round_to_64, mean_or_median,
#                         combined=(profile_column=='combined') # Pass combined flag
#                     )
#                     y_mean_list.append(y_mean)
#                     y_std_list.append(y_std)

#                 # Convert to numpy arrays to handle NaNs gracefully
#                 y_mean_array = np.array(y_mean_list)
#                 y_std_array = np.array(y_std_list)

#                 # Filter out NaNs for plotting, but keep batch_size for x-axis alignment
#                 valid_indices = ~np.isnan(y_mean_array) & ~np.isnan(y_std_array)

#                 if np.any(valid_indices): # Only plot if there is valid data
#                     current_ax.errorbar(np.array(BATCH_SIZE_LIST)[valid_indices],
#                                          np.multiply(y_mean_array[valid_indices], 1000), # Convert to ms
#                                          yerr=y_std_array[valid_indices],
#                                          marker=marker_list[color_counter],
#                                          markersize=11, alpha=0.9,
#                                          color=color_list[color_counter],
#                                          label=label, linestyle='')
#                 else:
#                     # print(f"DEBUG: No valid data points to plot for model={model}, profile={profile_column}, method={batch_method_original}")
#                     pass # No data to plot for this method

#             # --- Set subplot specific properties ---
#             current_ax.set_ylim(0, current_ylim)
#             current_ax.set_xlim(xlim[0], xlim[1])
#             current_ax.set_xticks(BATCH_SIZE_LIST) # Use BATCH_SIZE_LIST for x-ticks

#             # Y-axis labels and ticks only for the leftmost column (SchNet)
#             if col_idx == 0:
#                 current_ax.set_ylabel(f'{profile_column.capitalize()} time (ms)', fontsize=FONTSIZE, font=font_props)
#                 current_ax.set_yticklabels(current_ylabels, font=font_props, fontsize=FONTSIZE, rotation=0)
#             else:
#                 current_ax.set_yticklabels([]) # Hide Y-axis labels for other columns

#             # X-axis labels only for the bottom row
#             if row_idx == len(plot_content_rows) - 1:
#                 current_ax.set_xticklabels(BATCH_SIZE_LIST, font=font_props, fontsize=FONTSIZE, rotation=0)
#                 current_ax.set_xlabel('Batch size', fontsize=FONTSIZE, font=font_props)
#             else:
#                 current_ax.set_xticklabels([]) # Hide X-axis labels for other rows


#     # --- General Figure Adjustments ---
#     # Add a single legend to the top-left subplot
#     handles, labels = ax[0, 0].get_legend_handles_labels()
#     if handles: # Only add legend if there are items to legend
#         fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
#                    prop=font_props, edgecolor="black", fancybox=False)


#     # Apply 'science' style (if available)
#     try:
#         plt.style.use(["science", "grid"])
#     except ImportError:
#         print("SciencePlots not installed. Plotting without it.")

#     fig.align_labels() # Align the axis labels
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for legend and overall title

#     # Save and show the plot
#     # You might want a more dynamic filename based on compute_type and mean_or_median
#     plt.savefig(
#         f'/home/dts/Documents/theory/batching_paper/figs/profiling_multiple_models_{compute_type}_{mean_or_median}.png',
#         dpi=600)
#     plt.show()


def plot_four_plots(df):
    """Plot all four 4x4 subplot plots.
    
    Each plot is a different ML model + hardware platform

    AFLOW in the left columns, qm9 in the right columns.
    Batching time in the top rows and update time in the bottom rows.

    This script runs through a for loop.
    """

    for model in MODEL_TYPE_LIST:
        for compute_type in COMPUTING_TYPE_LIST:
            plot_batching_update_subplot(df, model, compute_type)
            pass


def get_avg_std_of_profile_column(
        df, profile_column, model, batch_method, compute_type, batch_size,
        dataset, batching_round_to_64, mean_or_median='mean', combined=False):

    profile_column_dict = {
        'batching': f'step_2_000_000_batching_time_{mean_or_median}',
        'update': f'step_2_000_000_update_time_{mean_or_median}',
        'combined': f'step_2_000_000_update_time_{mean_or_median}',
        'recompilation': 'recompilation_counter',
    }

    gpu_profiling_df = df[
        (df['dataset'] == dataset) & (df['model'] == model) &
        (df['batching_type'] == batch_method) &
        (df['computing_type'] == compute_type) &
        (df['batch_size'] == batch_size) &
        (df['batching_round_to_64'] == batching_round_to_64)]

    gpu_profiling_df_col = gpu_profiling_df[profile_column_dict[profile_column]]

    if gpu_profiling_df.empty:
        print(
            f'WARNING: No data found for filters: dataset={dataset}, model={model}, batch_method={batch_method},'
            f'compute_type={compute_type}, batch_size={batch_size}')

    if mean_or_median == 'mean':
        mean_result, std_result = gpu_profiling_df_col.mean(), gpu_profiling_df_col.std()
        if profile_column == 'update':
            ## Then we need to subtract the mean batching time and add the batching time standard dev.
            gpu_profiling_df_batching = gpu_profiling_df[profile_column_dict['batching']]
            mean_batching, std_batching = gpu_profiling_df_batching.mean(), gpu_profiling_df_batching.std()
            mean_result = mean_result - mean_batching
            std_result = std_result + std_batching            
    elif mean_or_median == 'median':
        mean_result, std_result = gpu_profiling_df_col.median(), gpu_profiling_df_col.std()
        if profile_column == 'update':
            ## Then we need to subtract the mean batching time and add the batching time standard dev.
            gpu_profiling_df_batching = gpu_profiling_df[profile_column_dict['batching']]
            median_batching, std_batching = gpu_profiling_df_batching.median(), gpu_profiling_df_batching.std()
            mean_result = mean_result - median_batching
            std_result = std_result + std_batching
    else:
        raise(ValueError)
    return mean_result, std_result


def plot_batching_update_subplot(df, model, compute_type, mean_or_median):
    """Here we want to feed into the correct dataframes to be able to easily make a four panel plot
    
    AFLOW in the left colum, qm9 in the right column.
    Batching time in the top row, update time in the bottom row.
    """
    batching_aflow_list = []
    update_aflow_list = []

    batching_aflow_list = []
    update_aflow_list = []

    aflow_batch_axes = [0, 0]
    # previously was 7,7
    fig, ax = plt.subplots(3, 2, figsize=(5.1, 7.65)) #, gridspec_kw={'height_ratios': [1]})

    aflow_batching_axes = ax[0, 0]
    qm9_batching_axes = ax[0, 1]
    aflow_update_axes = ax[1, 0]
    qm9_update_axes = ax[1, 1]
    aflow_combined_axes = ax[2, 0]
    qm9_combined_axes = ax[2, 1]

    axes_list = [
        aflow_batching_axes,
        qm9_batching_axes,
        aflow_update_axes,
        qm9_update_axes,
        aflow_combined_axes,
        qm9_combined_axes]
    plot_content_list = [('aflow', 'batching'), ('qm9', 'batching'), ('aflow', 'update'),
                         ('qm9', 'update'), ('aflow', 'combined'), ('qm9', 'combined')]
    # color_list = ['k', 'r', 'b', 'y']
    color_list = ['#1f77b4', '#ff7f0e', '#9467bd']
    # color_list = ['k', 'r', 'b']
    marker_list = ['x', '^', '.']

    xlim = [0, 140]
    if model == 'schnet':
        ylim = 8
        ylabels = [0, 2, 4, 6, 8]

    elif model == 'MPEU':
        ylim = 10
        ylabels = [0, 2, 4, 6, 8, 10]

    elif model == 'painn':
        ylim = 50
        ylabels = [0, 10, 20, 30, 40]

    for plot_num in range(len(axes_list)):
        dataset = plot_content_list[plot_num][0]  # Should be either AFLOW or qm9
        profile_column = plot_content_list[plot_num][1]  # Either `batching` or `update`
        batching_round_to_64 = False
        for color_counter, batch_method in enumerate(BATCH_METHOD_LIST):
            y_mean_list = []
            y_std_list = []
            label = batch_method
            if batch_method == 'static-64':
                batching_round_to_64 = True
                batch_method = 'static'
                label = 'static-$64$'
            elif batch_method == 'static':
                batching_round_to_64 = False
                batch_method = 'static'
                label = 'static-$2^N$'

            # elif batch_method == 'static':
            #     batching_round_to_64 = True
            else:
                batching_round_to_64 = False  # Different than the 200k experiments.
                # sys.err(f'error wrong batch method {batch_method}')
            for batch_size in BATCH_SIZE_LIST:

                print(f'batch rond to 64 is set to {batching_round_to_64}')

                y_mean, y_std = get_avg_std_of_profile_column(
                    df, profile_column, model, batch_method, compute_type,
                    batch_size, dataset, batching_round_to_64, mean_or_median)
                y_mean_list.append(y_mean)
                y_std_list.append(y_std)
            

            print(f' the batch method is: {batch_method}')
            print(f' the y mean list is {y_mean_list}')
            # print(f' the profile col is: {profile_column}')
            print(f' the dataset is: {dataset}')
            print(f'The std is {y_std}\n')

            axes_list[plot_num].errorbar(BATCH_SIZE_LIST,
                                     np.multiply(y_mean_list, 1000), yerr=y_std,
                                     marker=marker_list[color_counter],
                                     markersize=11, alpha=0.9,
                                     color=color_list[color_counter],
                                     label=label, linestyle='')
        
    # ax[0, 0].set_xlim(0, 5)
    # ax[0, 0].set_xticklabels(['', '16', '32', '64', '128', ''], minor=False)

    ax[0, 0].set_title('AFLOW', font=FONT, fontsize=FONTSIZE)
    ax[0, 1].set_title('QM9', font=FONT, fontsize=FONTSIZE)

    if model == 'schnet':
        model_label = 'SchNet'
        offset = 0
    elif model == 'MPEU':
        model_label = model
        offset = 1.5
    elif model == 'painn':
        model_label = 'PaiNN'
        offset=0

    if compute_type == 'cpu':
        ylim = 200
        ylabels = [0, 50, 100, 150, 200]
        ax[0, 1].text(12, 6.5, 'CPU only', font=FONT, fontsize=FONTSIZE)
    else:
        # ax[0, 1].text(12, 6.5+offset, 'GPU+CPU', font=FONT, fontsize=FONTSIZE)
        if model == 'schnet':
            ylim = 8
            ylabels = [0, 2, 4, 6, 8]
        elif model == 'MPEU':
            ylim = 10
            ylabels = [0, 2, 4, 6, 8, 10]
        elif model == 'painn':
            ylim = 40
            ylabels = [0, 10, 20, 30, 40]

    ax[0, 1].text(12, 5.5+offset, mean_or_median, font=FONT, fontsize=FONTSIZE)


        
    ax[0, 1].text(12, 6.5+offset, model_label, font=FONT, fontsize=FONTSIZE)


    ax[0, 0].set_ylabel('Batching time (ms)', fontsize=FONTSIZE, font=FONT)
    # ax[0, 0].set_yscale('log')
    # ax[0, 0].set_yticks([1E-1, 1E-0, 1E1, 1E2, 1E3], minor=False)
    ax[0, 0].set_ylim(0, 8)
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_yticklabels([0, 2, 4, 6, 8], font=FONT, fontsize=FONTSIZE, rotation=0)


    # ax[0, 1].set_ylim(0, ylim)
    ax[0, 1].set_ylim(0, 8)

    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 0].set_xticks([16, 32, 64, 128])
    ax[0, 1].set_xticks([16, 32, 64, 128])
    ax[0, 0].set_xlim(xlim[0], xlim[1])
    ax[0, 1].set_xlim(xlim[0], xlim[1]) 
    ax[1, 0].set_ylabel('Update time (ms)', fontsize=FONTSIZE, font=FONT)


    ax[1, 0].set_ylim(0, ylim)

    ax[1, 0].set_xticklabels([])
    ax[1, 1].set_ylim(0, ylim)
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xticklabels([])
    ax[1, 0].set_xticks([16, 32, 64, 128])
    ax[1, 1].set_xticks([16, 32, 64, 128])
    ax[1, 0].set_xlim(xlim[0], xlim[1])
    ax[1, 0].set_yticklabels(ylabels, font=FONT, fontsize=FONTSIZE, rotation=0)

    ax[1, 1].set_xlim(xlim[0], xlim[1]) 


    ax[2, 0].set_ylim(0, ylim)
    ax[2, 1].set_ylim(0, ylim)
    ax[2, 1].set_yticklabels([])
    ax[2, 0].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)

    ax[2, 1].set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax[2, 0].set_ylabel('Combined time (ms)', fontsize=FONTSIZE, font=FONT)
    ax[2, 0].set_xlim(xlim[0], xlim[1])
    ax[2, 1].set_xlim(xlim[0], xlim[1]) 
    ax[2, 0].set_yticklabels(ylabels, font=FONT, fontsize=FONTSIZE, rotation=0)
  
    ax[2, 0].set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[2, 1].set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=0)
    ax[2, 0].set_xticks([16, 32, 64, 128])
    ax[2, 1].set_xticks([16, 32, 64, 128])

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    ax[0, 0].legend(loc='upper left', prop=font, edgecolor="black", fancybox=False)


    # ax[1, 1].legend(loc='lower right')
    # for i in range(3):
    #     for j in range(2):
    #         ax[i, j].tick_params(axis='both', which='minor', labelsize=FONTSIZE-2)
    #         ax[i, j].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    #         plt.setp(ax[i,j].get_xticklabels(), fontsize=FONTSIZE, font=FONT) 
    #                 # horizontalalignment="left")
    #         plt.setp(ax[i,j].get_yticklabels(), fontsize=FONTSIZE, font=FONT)
    #                 # horizontalalignment="left")


    plt.style.use(["science", "grid"])
    fig.align_labels()

    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/profiling_2mill_{model}_gpu_aflow_left_qm9_right_{mean_or_median}.png',
        dpi=600)
    plt.show()







def plot_recompilation_bar_plot(df):
    """Create bar plot of # of recompilations.
    
    X-axis is the batch size.
    Y-axis is the number of recompilations.
    """
    profile_column = 'recompilation'
    computing_type = 'gpu_a100'
    model = 'painn'
    dataset = 'qm9'
    # color_list = ['#1f77b4', '#ff7f0e', '#9467bd']
    color_list = ['skyblue', '#ff7f0e', 'mediumvioletred']

    # Create a new batching method, batch-64 based on rounding.
    df.loc[
        (df.batching_type == 'static') & (df.batching_round_to_64 == True),'batching_type'] ='static-64'
    # Get data only for gpu and AFLOW and MPEU
    df = df[df['model'] == model]
    df = df[df['computing_type'] == computing_type]
    df = df[df['dataset'] == dataset]
    df['recompilation_counter'] = df['recompilation_counter'] - 1  # Don't count first compile.
    

    # Now take the mean over the different iterations.
    df = df[['batch_size', 'batching_type', 'recompilation_counter']]
    print(df.columns)
    df = df.groupby(['batch_size', 'batching_type']).mean()
    #[['static', 'static-64', 'dynamic']]
    #[['age','height','weight']]
    print(df.columns)
    # There is no stdev since the data is alwasy the same shuffle.
    # df_std = df.groupby(['batch_size', 'batching_type']).std()
    # print(df_std)
    print(df.unstack())
    print(df.unstack()['recompilation_counter'][['static', 'static-64', 'dynamic']])
    df = df.unstack()['recompilation_counter'][['static', 'static-64', 'dynamic']]
    ax = df.plot.bar(figsize=(5.1, 4), color=color_list, width=0.9)
    # bars = ax.patches
    # patterns = [ "" , "o" , "-"]
    # hatches = ''.join(h*len(df) for h in patterns)

    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch) 
    # import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)

    plt.legend(
        ["dynamic", "static-$2^N$", "static-$64$"], fontsize=FONTSIZE,
        prop=font, edgecolor="black", fancybox=False)
    ax.set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax.set_ylabel('Number of recompilations', fontsize=FONTSIZE, font=FONT)

    ax.set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=45)

    if dataset == 'aflow':
        ax.set_yticks([0, 100, 200, 300, 400, 500], font=FONT, fontsize=FONTSIZE)

        ax.set_yticklabels([0, 100, 200, 300, 400, 500], font=FONT, fontsize=FONTSIZE)

    else:
        ax.set_yticklabels([0, 50, 100, 150, 200, 250], font=FONT, fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/recompilation_count_2_million_dataset_{dataset}_model_{model}.png',
        dpi=600)
    plt.show()


def hard_code_recompilation_plot():
    """Give the raw numbers from the above function.
    
    This is a workaround since I cannot seem to get the correct ordering of for the columns
    in the pandas bar plot.
    """
    color_list = ['#1f77b4', '#ff7f0e', '#9467bd']
    __, ax = plt.subplots(figsize=(5.1, 4))
    bar_width = 0.3
    x = np.arange(len([16, 32, 64, 128]))
    static_2n_data = [2, 3, 2, 0.0]
    static_64_data = [68.0, 102.0, 157.0, 234.0]
    dynamic_data = [0, 0, 0, 0]
    plt.bar(x - 0.3, dynamic_data, bar_width, label='dynamic', color="skyblue")
    plt.bar(x + 0.3, static_2n_data, bar_width, label='static-$2^N$', color=color_list[1])
    plt.bar(x, static_64_data, bar_width, label='static-64', color='mediumvioletred')

    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)


    ax.set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax.set_ylabel('Number of recompilations', fontsize=FONTSIZE, font=FONT)
    ax.set_xticks([0, 1, 2, 3], font=FONT, fontsize=FONTSIZE, rotation=45)

    ax.set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=45)
    ax.set_yticks([0, 50, 100, 150, 200, 250], font=FONT, fontsize=FONTSIZE)
    # ax.set_yticklabels([0, 10, 20, 30, 40, 50], font=FONT, fontsize=FONTSIZE)
    # else:
    #     ax.set_yticklabels([0, 50, 100, 150, 200, 250], font=FONT, fontsize=FONTSIZE)
    plt.legend(
        ["dynamic", "static-$2^N$", "static-$64$"], fontsize=FONTSIZE,
        prop=font, edgecolor="black", fancybox=False, loc='upper left')
    offset=0
    ax.text(2.0, 230, 'QM9', font=FONT, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/recompilation_count_2_million.png',
        dpi=600)
    plt.show()

def main(argv):
    # plot learning curves
    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))
    # Ok now let's plot the batching times. Let's plot 4 graphs.
    # AFLOW / SchNet (GPU / CPU)
    print(df['step_2_000_000_batching_time_mean'].head())
    print(df['computing_type'])
    # plot_batching_update_subplot(df, model='painn',
    #                              compute_type='gpu_a100',
    #                              mean_or_median='mean')

    df_list = [pd.read_csv(os.path.join(BASE_DIR, MPEU_SCHNET_CSV)), pd.read_csv(os.path.join(BASE_DIR, PAINN_CSV))]
    # plot_batching_update_subplot(df, model_list='painn',
    #                              compute_type='gpu_a100',
    #                              mean_or_median='mean')
    models_to_plot = ['schnet', 'MPEU', 'painn']
    # models_to_plot = ['schnet', 'MPEU']
    # models_to_plot = ['painn']

    # plot_multiple_models_subplot(df_list, models_to_plot, compute_type='gpu_a100', mean_or_median='mean')

    plot_six_columns_models_datasets(df_list, models_to_plot, compute_type='gpu_a100', mean_or_median='mean')

if __name__ == '__main__':
    app.run(main)
