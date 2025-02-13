# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:10:21 2025

@author: Julian Bandhan
"""

import pandas as pd
import numpy as np
import mat73
import fooof
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'  # Set global font to Arial

import seaborn as sns
import pingouin as pg
#%matplotlib qt

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# %%

# Load in the subject data
subject_data = pd.read_excel(r'Y:\projects\OPM-Analysis\OPM_retest\demographics\OPM_retest_inscapes.xlsx')
subject_data = subject_data.loc[subject_data.including == 1]
subject_data = subject_data.reset_index()

# Set the age bins
sub_bins = ['RP01','RP02','RP03','RP04','RP05']


# Set the info to load the PSD data
derivatives_dir = r'Y:\projects\OPM-Analysis\OPM_retest\data\derivatives\inscapes_no_ica_2024_12_16_v7.0.0'
task = 'inscapes'
psd_suf = '_9-fooof_desc-whole_brainVE_coordinates_AAL90_1_150Hz_psd.mat'
psd_regional_suf = '_9-fooof_desc-VE_coordinates_AAL90_1_150Hz_psd.mat'

# Load in the AAL region names
aal_reg = pd.read_csv(r'Y:/projects/OPM/opm_pipeline_templates/Adult/AAL90.txt', header=None, delimiter='\t')[1].tolist()


# %% Functions

# Get spectrums for all (?)
def getSpectrums(df):

    # Initialize
    spectrums = []
    freqs = []

    # Iterate over the subjects
    for ind, row in df.iterrows():
        data = mat73.loadmat(derivatives_dir + '/sub-' + row['subject'] + '/ses-00' + str(row['session']) + '/opm/sub-' + row['subject'] + '_ses-00' + str(row['session']) + '_task-' + task + '_run-00' + str(row['run']) + psd_suf)
        spectrums.append(np.array(data['psd']))
        freqs.append(np.array(data['f']))

    return spectrums, freqs

# Get regioinal spectrums for a particular range
def getRegionalSpectrums(df):

    # Initialize
    spectrums = []
    freqs = []

    # Iterate over the subjects
    for ind, row in df.iterrows():

        data = mat73.loadmat(derivatives_dir + '/sub-' + row['subject'] + '/ses-00' + str(row['session']) + '/opm/sub-' + row['subject'] + '_ses-00' + str(row['session']) + '_task-' + task + '_run-00' + str(row['run']) + psd_regional_suf)
        spectrums.append(np.array(data['psd']))
        freqs.append(np.array(data['f']))

    return spectrums, freqs

# Get FOOOF fits
def getFits(spectrums, freqs, freq_range, peak_width_limits, max_n_peaks, peak_threshold, min_peak_height):
    
    # Initialize
    fm = []

    # Iterate over the subjects
    for sub in range(0, len(spectrums)):

        # Fit the original FOOOF
        fm_temp = fooof.FOOOF(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, peak_threshold=peak_threshold, min_peak_height=min_peak_height)
        fm_temp.fit(freqs[sub], spectrums[sub], freq_range)
        fm.append(fm_temp)
        
    return fm

# Get regional FOOOF fits
def getRegionalFits(spectrums, freqs, freq_range, peak_width_limits, max_n_peaks, peak_threshold, min_peak_height):
    
    # Initialize
    fm = []

    # Iterate over the subjects
    for sub in range(0, len(spectrums)):

        # Initialize
        fm_sub = []

        # Iterate over the regions
        for ind, reg in enumerate(spectrums[sub]):

            # Fit the original FOOOF
            fm_temp = fooof.FOOOF(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, peak_threshold=peak_threshold, min_peak_height=min_peak_height)
            fm_temp.fit(freqs[sub], reg, freq_range)
            fm_sub.append(fm_temp)

        # Append
        fm.append(fm_sub)
        
    return fm

# %%

# Get the spectrums
spectrums, freqs = getSpectrums(subject_data)

# Set the FOOOF parameters
freq_range = [2, 40] 
peak_width_limits = [1.2, np.Inf]
max_n_peaks = np.Inf
peak_threshold = 0
min_peak_height = 0

# Define each frequency band and width
#bands = {'alpha': [6, 12.99], 'lbeta': [13, 20.99], 'hbeta': [21, 25.99]}
bands = {'theta': [1,8], 'alpha':[8,13], 'beta':[13,30],'lgamma':[30,50],'hgamma':[50,80]}

# Fit
fm = getFits(spectrums, freqs, freq_range=freq_range, peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, peak_threshold=peak_threshold, min_peak_height=min_peak_height)


# %%

# Extract the fooofed, periodic, and aperiodic data
fooofed_spectrum = np.stack([getattr(i, 'fooofed_spectrum_') for i in fm], axis=0)
spectrum = np.stack([getattr(i, 'power_spectrum') for i in fm], axis=0)
ap_plot = np.stack([getattr(i, '_ap_fit') for i in fm], axis=0)
paramts = np.stack([getattr(i,'aperiodic_params_') for i in fm],axis = 0)
p_plot = spectrum - ap_plot
rmse = (spectrum - fooofed_spectrum)**2
freq = np.array(fm[0].freqs)


# Assign to a dataframe
df = subject_data[['subject']]
df = pd.concat((df, pd.DataFrame(spectrum, columns=freq)), axis=1)
df = pd.melt(df, id_vars=['subject'], value_vars=freq, var_name='freq', value_name='spectrum')

df2 = subject_data[['subject']]
df2 = pd.concat((df2, pd.DataFrame(ap_plot, columns=freq)), axis=1)
df2 = pd.melt(df2, id_vars=['subject'], value_vars=freq, var_name='freq', value_name='ap_plot')

df3 = subject_data[['subject']]
df3 = pd.concat((df3, pd.DataFrame(p_plot, columns=freq)), axis=1)
df3 = pd.melt(df3, id_vars=['subject'], value_vars=freq, var_name='freq', value_name='p_plot')

df4 = subject_data[['subject']]
df4 = pd.concat((df4, pd.DataFrame(fooofed_spectrum, columns=freq)), axis=1)
df4 = pd.melt(df4, id_vars=['subject'], value_vars=freq, var_name='freq', value_name='fooofed_spectrum')
df = pd.concat((df, df2['ap_plot'], df3['p_plot'], df4['fooofed_spectrum']), axis=1)


# Fitted
sns.set_context('poster')
sns.set_style('white')
fig = plt.figure(figsize=(15,10))
axs = sns.lineplot(data=df, x='freq', y='spectrum', hue='subject', errorbar='sd', linewidth=5)
axs.set(xlabel='Frequency (Hz)', ylabel='Absolute power (log $fT^2$/Hz)', xlim=[1, 40], ylim=[-3.2, -0.5])
axs.set_ylabel('Absolute power (log $fT^2$/Hz)',fontsize=60)
axs.get_legend()
plt.savefig(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\foof\absolute_power.png')

base_array = np.tile(np.arange(1, 6), 5)  
sessions = np.tile(base_array, 65)

df['session'] = sessions

subject_labels = [f"sub-{i:03d}" for i in range(1, 6)]  # ['sub-001', ..., 'sub-005']
subjects = np.repeat(subject_labels, 5)  # Each subject repeated 5 times in order
subjects = np.tile(subjects, len(sessions) // len(subjects))  # Tile to match session length

df['subs'] = subjects

# Define a custom color palette for subjects RP01 to RP05
subject_colors = {
    'sub-001': 'royalblue',
    'sub-002': 'orangered',
    'sub-003': 'gold',
    'sub-004': 'darkviolet',
    'sub-005': 'yellowgreen'
}
# %% Spectrum Plot
sns.set_context('poster', rc={
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})
sns.set_style('white')

# Create figure
fig = plt.figure(figsize=(10, 15))

# 1️⃣ Plot **individual session lines** for each subject/session pair
for subj in df['subs'].unique():
    for session in df[df['subs'] == subj]['session'].unique():
        subset = df[(df['subs'] == subj) & (df['session'] == session)]
        plt.plot(
            subset['freq'], subset['spectrum'], 
            color=subject_colors.get(subj, 'gray'),  # Use gray if subject is outside defined range
            alpha=0.2, linewidth=2
        )

# 2️⃣ Plot **subject-averaged lines** with error bars (thicker)
sns.lineplot(
    data=df, x='freq', y='spectrum', hue='subs', 
    errorbar=None, linewidth=3, palette=subject_colors, legend=True
)

# Set axis labels and limits
plt.xlabel('Frequency (Hz)', labelpad=15, fontsize=30)
plt.ylabel('Absolute power \n (log $fT^2$/Hz)', labelpad=15, fontsize=30)
plt.xlim([1, 40])
plt.legend(title=None)
#plt.ylim([-3.2, -0.5])
plt.tight_layout()

# Save figure
plt.savefig(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\foof\absolute_power.png')

plt.show()
# %% Aperiodic Power Plot

# Set Seaborn styles
sns.set_context('poster', rc={
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})
sns.set_style('white')

# Create figure
fig = plt.figure(figsize=(10, 15))

# 1️⃣ Plot **individual session lines** for each subject/session pair
for subj in df['subs'].unique():
    for session in df[df['subs'] == subj]['session'].unique():
        subset = df[(df['subs'] == subj) & (df['session'] == session)]
        plt.plot(
            subset['freq'], subset['ap_plot'], 
            color=subject_colors.get(subj, 'gray'),  # Use gray if subject is outside defined range
            alpha=0.2, linewidth=2
        )

# 2️⃣ Plot **subject-averaged lines** with error bars (thicker)
sns.lineplot(
    data=df, x='freq', y='ap_plot', hue='subs', 
    errorbar=None, linewidth=3, palette=subject_colors, legend=True
)

# Set axis labels and limits
plt.xlabel('Frequency (Hz)', labelpad=15, fontsize=30)
plt.ylabel('Aperiodic power\n (log $fT^2$/Hz)', labelpad=15, fontsize=30)
plt.xlim([1, 40])
plt.legend(title=None)
#plt.ylim([-3.2, -0.5])
plt.tight_layout()

# Save figure
plt.savefig(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\foof\aperiodic_power.png')

plt.show()

#%% Periodic Power Plot

# Set Seaborn styles
sns.set_context('poster', rc={
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})
sns.set_style('white')

# Create figure
fig = plt.figure(figsize=(10, 15))

# 1️⃣ Plot **individual session lines** for each subject/session pair
for subj in df['subs'].unique():
    for session in df[df['subs'] == subj]['session'].unique():
        subset = df[(df['subs'] == subj) & (df['session'] == session)]
        plt.plot(
            subset['freq'], subset['p_plot'], 
            color=subject_colors.get(subj, 'gray'),  # Use gray if subject is outside defined range
            alpha=0.2, linewidth=2
        )

# 2️⃣ Plot **subject-averaged lines** with error bars (thicker)
sns.lineplot(
    data=df, x='freq', y='p_plot', hue='subs', 
    errorbar=None, linewidth=3, palette=subject_colors, legend=True
)

# Set axis labels and limits
plt.xlabel('Frequency (Hz)', labelpad=15, fontsize=30)
plt.ylabel('Periodic power\n (log $fT^2$/Hz)', labelpad=15, fontsize=30)
plt.xlim([1, 40])
plt.legend(title=None)
#plt.ylim([-3.2, -0.5])
plt.tight_layout()

# Save figure
plt.savefig(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\foof\periodic_power.png')


# Fitted - add bins
ylim = axs.get_ylim()
for band, width in bands.items():
    plt.vlines(width, ylim[0], ylim[1], colors='grey', linestyles='dashed')
axs.set(ylim=ylim)
plt.savefig(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\foof\periodic_power_with_widths.png')

# %% ICC stuffs
import mne
import numpy as np
from matplotlib import pyplot as plt
import os.path as op


import pandas as pd
import pingouin as pg



participants = np.repeat(np.arange(1,6),65)
base_array = np.tile(np.arange(1, 6), 5)  
sessions = np.tile(base_array, 65)

df['session'] = sessions


icc = pg.intraclass_corr(data=df, targets = 'subject', raters='session', ratings = 'p_plot')

 # %%
 
 
participants = np.repeat(np.arange(1,6),5)
sessions = np.tile(np.arange(1,6),5)

df_fooof = pd.DataFrame({
    'Participant':participants,
    'Session':sessions,
    'Power':paramts[:,1]
    })

icc = pg.intraclass_corr(data=df_fooof, targets = 'Participant', raters='Session', ratings = 'Power')

icc[icc['Type'] == 'ICC1']['CI95%'].values[0]



