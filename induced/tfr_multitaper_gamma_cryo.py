# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:10:05 2025

@author: Natalie Rhodes
"""

import mne
import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_1samp, pearsonr
from mne.stats import permutation_cluster_1samp_test, fdr_correction
import matplotlib as mpl
from matplotlib import pyplot as plt

# Define parameters
n_subjects =3
n_runs = 1  
sfreq = 600 
freqs = np.arange(4, 30, 1.0)  # Frequency range from 30 to 100 Hz  
n_cycles = freqs / 2  # Different number of cycles per frequency  
time_bandwidth = 4  # Time bandwidth  
baseline = (-0.9, -0.1)
time_window = (0.1, 0.9)  # Analysis time range

# Set font to Arial
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 24  # Larger font size for axes labels
mpl.rcParams['xtick.labelsize'] = 20  # Larger font size for x-axis ticks
mpl.rcParams['ytick.labelsize'] = 20  # Larger font size for y-axis ticks
# Define color palette (MATLAB's cols 1-5: blue, red, yellow, purple, green)
matlab_colors = [
    '#0072BD',  # Blue
    '#D95319',  # Red
    '#EDB120',  # Yellow
    '#7E2F8E',  # Purple
    '#77AC30',  # Green
]


# Store peak gamma frequencies
peak_gamma_frequencies = []
peak_widths = []
sub_mask = []
sess_masks = []
tfr_data_arrays = []
# Loop over subjects  
subs = [1,4,5]    
tfr_list = []  # Store TFRs for averaging later  
tfr_data_list = [] # Store TFR raw data for second-level stats
for ss in subs:  


    # Load data  
    file_path = f"/d/mjt/9/projects/OPM-Analysis/OPM_retest/data/derivatives/cryo_gamma_analysis_20250130_v7.0.0/sub-RP0{ss}/ses-001/cryo/sub-RP0{ss}_ses-001_ts_pca_5mm.mat"  
    
    data = loadmat(file_path)  
    
    ts_data = data['ts']  
    # Reshape data  
    
    ts_data = np.transpose(ts_data, (1, 0))  # (epochs, time)
    
    ts_data = ts_data[:, np.newaxis, :]  # Add channel dim -> (epochs, 1, time)
    
    # Create MNE info and epochs  
    
    n_channels = ts_data.shape[1]  # Get correct number of channels  
    info = mne.create_info(n_channels, sfreq=sfreq)  
    epochs = mne.EpochsArray(ts_data, info, tmin=-1)  

    # Compute Time-Frequency Representation (TFR)- UNAVERAGED
    power = mne.time_frequency.tfr_multitaper(
    epochs, picks="all", freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, average=False
    )  
    
    # Get TFR data shape: (n_channels, n_freqs, n_times, n_epochs)
    tfr_data = power.data
    tfr_data = np.squeeze(tfr_data)
    
    # === Convert TFR data to percent change relative to baseline ===
    baseline_idx = np.where((power.times >= baseline[0]) & (power.times <= baseline[1]))[0]  # Get baseline indices
    baseline_power = np.mean(tfr_data[:, :, baseline_idx], axis=2, keepdims=True)  # Mean across baseline window
    tfr_data = ((tfr_data - baseline_power) / baseline_power) * 100  # Compute percent change
    
    tfr_data_list.append(np.mean(tfr_data,axis=0))
     
    tfr_list.append(power)  

    
        
#%% 


data_avg = tfr_data_list
time_idx = np.where((power.times >= time_window[0]) & (power.times <= time_window[1]))[0]
f_inds = (4,30)
freq_idx = np.where((freqs >= f_inds[0]) & (freqs <= f_inds[1]))[0]
mean_power = [np.mean(inst[:,time_idx],1) for inst in data_avg]
mean_envelope = [np.mean(inst,0) for inst in data_avg]

peak_freq = []
peak_amp = []
env_amp = []
for ss in range(3):
    # fig,ax = plt.subplots(figsize=(6,5))
    max_amp = np.max(mean_power[ss][freq_idx])
    max_freq = freqs[np.argmax(mean_power[ss][freq_idx])]
    peak_freq.append(max_freq)
    peak_amp.append(max_amp)
    # plt.plot(freqs,mean_power[ss], color=matlab_colors[ss], linewidth=3)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude (% Change)")
    # plt.tight_layout()
    # #plt.ylim(-10,65)
    # plt.xlim(30,99)
    # plt.show()
    
    fig,ax = plt.subplots(figsize=(6,5))
    template = power.copy().average()
    template.data = np.expand_dims(data_avg[ss], 0)

    template.plot(picks=0, axes=ax, tmin=-0.5, tmax=1.5)
    plt.tight_layout()
    
    # fig,ax = plt.subplots(figsize=(6,5))
    # plt.plot(template.times, mean_envelope[ss], color=matlab_colors[ss])
    mean_amp = np.mean(mean_envelope[ss][time_idx])
    env_amp.append(mean_amp)
    # plt.plot(template.times, mean_envelope[ss], color=matlab_colors[ss], linewidth=3)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude (% Change)")l
    # plt.tight_layout()
    # #plt.ylim(-10,30)
    # plt.xlim(-0.5,1.5)
    # plt.show()    
        
    

