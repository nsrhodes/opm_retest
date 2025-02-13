#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:24:46 2024

@author: julian
"""
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps import transforms
from surfplot import Plot, utils
import nibabel as nib
import matplotlib as mpl
from nilearn import image
import pingouin as pg
import pandas as pd
from scipy.io import loadmat
#import surfplot_atlas_plots


from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
tasks_data = loadmat(r'Y:\projects\OPM-Analysis\OPM_retest\code\noICA_IS_power_20241126.mat').get(f'task_data_array')

# %%

# Labels for frequencies
file_label = ['theta', 'alpha', 'beta', 'lowGamma', 'highGamma']
freq_lab = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']

# Define AAL indices and labels
aal_idx = [7, 14, 31, 37, 44, 55, 60, 77, 89]
aal_lab =['MFG', 'IFG', 'ACG','HIP', 'CAL', 'FFG', 'SPG', 'THA', 'ITG']
region_arr = [None]*90
for x in range(len(aal_idx)):
    region_arr[aal_idx[x]] = aal_lab[x]

#%%
# Iterate over each frequency band
for i in range(1):  # Change this to range(5) to loop over all bands
    data = tasks_data[:, i, :]

    # Create figure and heatmap
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(data.T, cmap="Reds", xticklabels=False, square = True, yticklabels=region_arr, cbar=True)
    
    # Configure colorbar tick labels
    colbar = ax.collections[0].colorbar
    colbar.ax.tick_params(labelsize=60)
    #for tick_label in colbar.ax.get_yticklabels():
        #tick_label.set_fontweight('bold')

    # Set y-axis tick labels
    plt.yticks( fontsize=52, rotation =0)
    
    # Define the run and subject labels
    #runs = [f"{i+1}" for i in range(5)] * 5  # Repeats '1, 2, 3, 4, 5' five times
    subjects = [f"Sub {i+1}" for i in range(5) for _ in range(5)]
    
    # Set x-ticks for run numbers
    #plt.xticks(ticks=np.arange(25) + 0.5, labels=runs, rotation=0, ha='center', fontsize=34)
    
    # Add "Sub" labels below the run numbers
    ax = plt.gca()
    for j in range(5):
        # Position each "Sub" label below the group of runs (1-5, 6-10, etc.)
        ax.text(2.5 + 5 * j, 94, f"Sub {j+1}", ha='center', va='bottom', fontsize=40)
    
    # Labels and Title
    plt.xlabel("Runs", labelpad=20, fontsize=60)
    plt.ylabel("AAL Regions", fontsize=60)
    #plt.title(f"{freq_lab[i]} oscillations in AAL Regions by Runs", fontweight='bold', fontsize=26)
    
    # Adjust layout to ensure clear visibility
    plt.tight_layout()
    plt.show()
    plt.savefig(fr'Y:\projects\OPM-Analysis\OPM_retest\code\plot_dump_IS\heatmaps\new_col_big{file_label[i]}.png')

# %%

# Iterate over each frequency band
for i in range(5):  # Change this to range(5) to loop over all bands
    data = tasks_data[:, i, :]*100

    # Calculate aspect ratio of the data (25 columns by 90 rows)
    aspect_ratio = data.shape[1] / data.shape[0]  # 25/90
    
    # Adjust figure size to ensure a square overall plot
    fig_width = 20  # Set desired figure width
    fig_height = fig_width / aspect_ratio
    fig = plt.figure(figsize=(15, 30))
    
    ax = sns.heatmap(
        data.T, 
        cmap="Reds", 
        xticklabels=True, 
        yticklabels=region_arr, 
        cbar=True, 
        #cbar_kws = {'label':'%'},
        vmin=data.min(),  # Set global vmin
        vmax=data.max()   # Set global vmax
    )
    
    ax.set_xticklabels([''] * data.shape[0])
    
    # Configure colorbar 
    colbar = ax.collections[0].colorbar
    tick_labels = colbar.get_ticks()  # Get the default tick positions
    colbar.set_ticks([data.min(), data.max()])  # Set them explicitly to prevent dynamic adjustment
    colbar.set_label('%', size = 70)
    colbar.ax.tick_params(labelsize=60) # Set tick size
    #for tick_label in colbar.ax.get_yticklabels():
        #tick_label.set_fontweight('bold')  # Ensure all tick labels are bold
        
    from matplotlib.ticker import FormatStrFormatter
    colbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


    # Set y-axis tick labels
    plt.yticks(fontsize=40, rotation=0)
    
    # Define the run and subject labels
    #runs = [f"{i+1}" for i in range(5)] * 5  # Repeats '1, 2, 3, 4, 5' five times
    subjects = [f"Sub {i+1}" for i in range(5) for _ in range(5)]
    
    # Set x-ticks for run numbers
    #plt.xticks(ticks=np.arange(25) + 0.5, labels=runs, rotation=0, ha='center', fontsize=34)
    
    # Add "Sub" labels below the run numbers
    ax = plt.gca()
    for j in range(5):
        # Position each "Sub" label below the group of runs (1-5, 6-10, etc.)
        ax.text(2.5 + 5 * j, 107, f"sub-00{j+1}", ha='center', va='bottom', fontsize=44, rotation = 20)
    
    # Labels and Title
    plt.xlabel("Runs", labelpad=85, fontsize=70)
    plt.ylabel("Regions", fontsize=70)
    # plt.title(f"{freq_lab[i]} oscillations in AAL Regions by Runs", fontweight='bold', fontsize=26)
    
    # Adjust layout to ensure clear visibility
    plt.tight_layout()
    plt.show()
    plt.savefig(fr'Y:\projects\OPM-Analysis\OPM_retest\code\plot_dump_IS\heatmaps\new_col_big{file_label[i]}.png')


# %% test# Define global vmin and vmax based on the overall data range
global_vmin = tasks_data.min()
global_vmax = tasks_data.max()

# Iterate over each frequency band
for i in range(5):  # Loop over all bands
    data = tasks_data[:, i, :]

    # Calculate aspect ratio of the data (25 columns by 90 rows)
    aspect_ratio = data.shape[1] / data.shape[0]  # 25/90
    
    # Adjust figure size to ensure a square overall plot
    fig_width = 20  # Set desired figure width
    fig_height = fig_width / aspect_ratio
    fig = plt.figure(figsize=(15, 30))
    
    # Create heatmap with explicit vmin and vmax
    ax = sns.heatmap(
        data.T, 
        cmap="rainbow", 
        xticklabels=True, 
        yticklabels=region_arr, 
        cbar=True, 
        #cbar_kws = {'label':'%'},
        vmin=global_vmin,  # Set global vmin
        vmax=global_vmax   # Set global vmax
    )
    
    # Configure colorbar 
    colbar = ax.collections[0].colorbar
    colbar.set_ticks([round(global_vmin,2), round(global_vmax,2)])  # Set ticks explicitly
    colbar.set_label('%', size = 50)
    colbar.ax.tick_params(labelsize=40)  # Set tick size

    # Set y-axis tick labels
    plt.yticks(fontsize=42, rotation=0)
    
    ax.set_xticklabels([''] * data.shape[0])
    
    # Define the run and subject labels
    subjects = [f"Sub {i+1}" for i in range(5) for _ in range(5)]
    
    # Add "Sub" labels below the run numbers
    ax = plt.gca()
    for j in range(5):
        ax.text(2.5 + 5 * j, 95, f"sub-00{j+1}", ha='center', va='bottom', fontsize=30)
    
    # Labels and Title
    plt.xlabel("Runs", labelpad=30, fontsize=50)
    plt.ylabel("Regions", fontsize=50)
    
    # Adjust layout to ensure clear visibility
    plt.tight_layout()
    plt.savefig(fr'Y:\projects\OPM-Analysis\OPM_retest\code\plot_dump_IS\heatmaps\{file_label[i]}.png')
    plt.show()
