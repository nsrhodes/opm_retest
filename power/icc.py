#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps import transforms
from surfplot import Plot, utils
import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib as mpl
from nilearn import image
import pingouin as pg
import pandas as pd
import numpy as np
from scipy.io import loadmat
import surfplot_atlas_plots
from surfplot_atlas_plots import atlas_surface_plotter, construct_mpl_surf_image, plottingFunc

task_id = 'IS' #rest/IS

# Assuming your data is in a .25 (runs) x 5 (frequency bands) x 90 (regions) numpy array
tasks_data = loadmat(fr'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/noICA_IS_power_20241126.mat').get(f'task_data_array')
data = tasks_data


#%% Plot Average Power per Frequency

file_label = ['theta', 'alpha', 'beta', 'lowGamma', 'highGamma']
freq_lab = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
col_lab = ['Reds', 'Purples', 'BuPu', 'Oranges', 'Reds']

for i in range(1):#len(tasks_data[0])):
    val_arr = np.mean(tasks_data[:,i,:], axis = 0)
    label = file_label[i]
    freq = f' Average {freq_lab[i]} activity'
    
    plottingFunc(val_arr, freq, cmapval = col_lab[i])
    plt.savefig(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/analysis/inscapes_plots/avg/avg_{label}.png')
        

#%% Plot Individual Activity per frequency

file_label = ['theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
freq_lab2 = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
col_lab = ['Blues', 'Purples', 'BuPu', 'Oranges', 'Reds']


for k in range(len(freq_lab2)):
    freq = freq_lab2[k]
    file = file_label[k]
    for j in range(len(tasks_data)):
        val_arr = tasks_data[j,k,:]
        
        
        
        # Calculate RP and run indices
        rp_num = (j // 5) + 1  # RP01, RP02, etc. (0-4 is RP01, 5-9 is RP02, etc.)
        run_num = (j % 5) + 1  # Run 1-5 within each RP
        
        label = f'{freq} Activity RP0{rp_num}_0{run_num}'
        
        plottingFunc(val_arr, label, cmapval=col_lab[k])
        plt.savefig(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_{task_id}/individual_{file}/RP0{rp_num}_0{run_num}_{file}.png')
    

#%%

# Reshape the data to a long format (this is a typical format required for ICC)
# Example for one region and one frequency band
region = 0  # select region
freq_band = 1  # select frequency band
data_region_band = data[:, freq_band, region]

# You need to create a dataframe with 'Participants', 'Sessions', and 'Power'
# For example, assuming 5 participants and 5 sessions:
participants = np.repeat(np.arange(1, 6), 5)  # 5 participants, 5 sessions each
sessions = np.tile(np.arange(1, 6), 5)  # 5 sessions for each participant

df = pd.DataFrame({
    'Participant': participants,
    'Session': sessions,
    'Power': data_region_band
})

# Compute ICC for this specific region and frequency band (Intra-class Correlation)
icc = pg.intraclass_corr(data=df, targets='Participant', raters='Session', ratings='Power')


#%%



# Create storage for ICC values over the 90 regions
icc_values = np.zeros((5,90,2)) #(freq_bands, regions, (icc1, icc1k)) 
avg_icc = np.zeros((5,3))
p_ci = np.zeros((5,90,3))


#Iterate over all region
for i in range(len(icc_values)):
    freq_band = i
    for j in range(len(icc_values[1])):
        
        region = j
        data_region_band = data[:,freq_band, region]
        
        participants = np.repeat(np.arange(1,6),5)
        sessions = np.tile(np.arange(1,6),5)
        
        df = pd.DataFrame({
            'Participant':participants,
            'Session':sessions,
            'Power':data_region_band
            })
        
        icc = pg.intraclass_corr(data=df, targets = 'Participant', raters='Session', ratings = 'Power')
        icc_values[i,j,0] = icc[icc['Type'] == 'ICC1']['ICC'].values[0]
        icc_values[i,j,1] = icc[icc['Type'] == 'ICC1k']['ICC'].values[0]
        
        p_ci[i,j,0] =  icc[icc['Type'] == 'ICC1']['pval'].values[0]
        p_ci[i,j,1] =  icc[icc['Type'] == 'ICC1']['CI95%'].values[0][0]
        p_ci[i,j,2] =  icc[icc['Type'] == 'ICC1']['CI95%'].values[0][1]
        
    avg_icc[i,0] = np.nanmean(icc_values[i,:,0])
    avg_icc[i,1] = np.nanmean(icc_values[i,:,1])
    avg_icc[i,2] = np.std(icc_values[i,:,0],ddof=1)

#%%

#matplotlib qt

icc1 = icc_values[4,:,1]

fig = plt.figure()

plottingFunc(icc1, 'highgamma ICC1k', cmapval='Reds')

#%%

file_label = ['theta', 'alpha', 'beta', 'lowGamma', 'highGamma']
freq_lab = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
col_lab = ['Blues', 'Purples', 'Greens', 'Oranges', 'Reds']
icclab = ['ICC', 'ICC1k']

for x in range(1):
    for y in range(2):
        label = f'ICC {freq_lab[x]} activity\n{icclab[y]} = {avg_icc[x,y]:.5f}'
        plottingFunc(icc_values[x,:,y], label, cmapval=col_lab[x])
        filename = f'avg_{file_label[x]}_{icclab[y]}.png'
        plt.savefig(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_IS/{filename}')
        
# %% Meta (?)
import numpy as np
import matplotlib.pyplot as plt

# Assume meta_array is your (5 x 90 x 3) array.
# meta_array[band, region, 0] = p-value
# meta_array[band, region, 1] = CI_lower
# meta_array[band, region, 2] = CI_upper

frequency_names = ['Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
meta_array = p_ci

for band in range(5):
    pvals = meta_array[band, :, 0]
    ci_lower = meta_array[band, :, 1]
    ci_upper = meta_array[band, :, 2]
    
    print(f"--- {frequency_names[band]} ---")
    print("P-values: mean =", np.mean(pvals),
          "median =", np.median(pvals),
          "std =", np.std(pvals))
    print("CI Lower: mean =", np.mean(ci_lower),
          "median =", np.median(ci_lower))
    print("CI Upper: mean =", np.mean(ci_upper),
          "median =", np.median(ci_upper))
    
    # Optional: plot histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.hist(pvals, bins=15, color='skyblue', edgecolor='black')
    plt.title(f'{frequency_names[band]} p-values')
    
    plt.subplot(1, 3, 2)
    plt.hist(ci_lower, bins=15, color='lightgreen', edgecolor='black')
    plt.title(f'{frequency_names[band]} CI Lower')
    
    plt.subplot(1, 3, 3)
    plt.hist(ci_upper, bins=15, color='salmon', edgecolor='black')
    plt.title(f'{frequency_names[band]} CI Upper')
    
    plt.tight_layout()
    plt.show()
