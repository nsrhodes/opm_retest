# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:46:13 2025

@author: Julian Bandhan
"""

from scipy.io import loadmat


data = loadmat(f'Y:\projects\OPM-Analysis\OPM_retest\code\power_inscapes.mat').get('pow_inscapes')

po = data[:,0] #Array of size 25, with 601xN values (601 psd by N channels)
post_po = data[:,1] #Array of size 25, with 601xN values (601 psd by N channels)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
%matplotlib qt

# Load data
data = loadmat(r'Y:\projects\OPM-Analysis\OPM_retest\code\power_inscapes.mat').get('pow_inscapes')

# Extract pre-processing and post-processing data (601xN matrices for 25 subjects)
po = [subj for subj in data[:,0]]  # List of pre-processing matrices
post_po = [subj for subj in data[:,1]]  # List of post-processing matrices

# Determine max number of channels to standardize shapes
max_channels = max(subj.shape[1] for subj in po)

# Stack and compute mean across all subjects and channels
all_pre = np.hstack([np.pad(subj, ((0,0), (0, max_channels - subj.shape[1])), mode='constant') for subj in po])
all_post = np.hstack([np.pad(subj, ((0,0), (0, max_channels - subj.shape[1])), mode='constant') for subj in post_po])

# Compute mean & standard deviation across subjects and channels
mean_pre = np.mean(all_pre, axis=1)
mean_post = np.mean(all_post, axis=1)
std_pre = np.std(all_pre, axis=1)
std_post = np.std(all_post, axis=1)

# X-axis for frequency bins
x = np.arange(601)

# --- Seaborn Aesthetic Settings ---
sns.set_theme(style="whitegrid")  # Clean white background with subtle grid

# Create the plot
plt.figure(figsize=(10, 5))

# Plot mean pre/post-processing
sns.lineplot(x=x, y=mean_pre, color='red', label='Pre Notch Filter', linewidth=2)
sns.lineplot(x=x, y=mean_post, color='blue', label=' Notch Filter', linewidth=2)

# Optional: Add shaded region for standard deviation
plt.fill_between(x, mean_pre - std_pre, mean_pre + std_pre, color='red', alpha=0.2)
plt.fill_between(x, mean_post - std_post, mean_post + std_post, color='blue', alpha=0.2)

# Labels & Aesthetics
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power ($fT^2$/$\sqrt{Hz}$)', fontsize=12)
plt.title('Mean Power Spectrum Before & After Processing', fontsize=14)

# X-axis limits (0 to 100)
plt.xlim([0, 100])
plt.ylim([10e-15, 10e-12])
plt.yscale('log')
#plt.xticks(np.linspace(0, 100, 6))  # Fewer, evenly spaced ticks
plt.yticks(fontsize=10)

# Improve grid aesthetics
plt.grid(True, linestyle='--', alpha=0.6)

# Legend
plt.legend(fontsize=12)

plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load data
data = loadmat(r'Y:\projects\OPM-Analysis\OPM_retest\code\power_inscapes.mat').get('pow_inscapes')

# Extract pre-processing and post-processing data (601xN matrices for 25 subjects)
po = [subj for subj in data[:,0]]  # List of pre-processing matrices
post_po = [subj for subj in data[:,1]]  # List of post-processing matrices

# Determine max number of channels to standardize shapes
max_channels = max(subj.shape[1] for subj in po)

# Stack and compute mean across all subjects and channels
all_pre = np.hstack([np.pad(subj, ((0,0), (0, max_channels - subj.shape[1])), mode='constant') for subj in po])
all_post = np.hstack([np.pad(subj, ((0,0), (0, max_channels - subj.shape[1])), mode='constant') for subj in post_po])

# Compute mean & standard deviation across subjects and channels
mean_pre = np.mean(all_pre, axis=1)
mean_post = np.mean(all_post, axis=1)
std_pre = np.std(all_pre, axis=1)
std_post = np.std(all_post, axis=1)

# X-axis for frequency bins
x = np.arange(601)

# --- Plot Settings ---
plt.figure(figsize=(10, 5))  # Set figure size
plt.plot(x, mean_pre, 'r', label='Pre Notch Filter', linewidth=2)
plt.plot(x, mean_post, 'b', label='Post Notch Filter', linewidth=2)

# Optional: Add shaded region for standard deviation
plt.fill_between(x, mean_pre - std_pre, mean_pre + std_pre, color='r', alpha=0.2)
plt.fill_between(x, mean_post - std_post, mean_post + std_post, color='b', alpha=0.2)

# Labels & Aesthetics
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power (fT/$\sqrt{Hz}$)', fontsize=12)
plt.title('Mean Power Spectrum Before & After Notch Filter', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Subtle grid lines

# Set limits and ticks to improve readability
plt.xlim([0, 100])  # Ensure full x-axis is shown
plt.yscale('log')
plt.ylim([10e-15, 10e-12])
plt.yticks(fontsize=10)  # Increase readability

plt.show()
