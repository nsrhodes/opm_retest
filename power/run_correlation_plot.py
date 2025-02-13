#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:28:22 2024

@author: julian
"""
#%matplotlib qt

from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

task = 'IS'
tasks_data = loadmat(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/noICA_{task}_power_20241126.mat').get(f'task_data_array')



from scipy.stats import linregress, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# %%
run_1 = [0, 5, 10, 15, 20]
run_2x = [1, 6, 11, 16, 21]
run_2 = [4,9,14,19,24]
freq_lab = ['Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
freq_name = ['teta', 'alpha', 'beta', 'lowgamma', 'highgamma']
cols = ['mediumblue', 'orangered', 'gold', 'darkviolet','olivedrab']

# %%

def format_ticks(value, _):
    """Formats tick labels to display up to 2 decimal places."""
    return f'{value:.0f}'

for l in range(1):
    plt.figure(figsize=(18, 18))
    
    for i in range(5):
        sub = i
        
        run_x = run_1[i]
        run_y = run_2[i]
        freq = l
        run1 = tasks_data[run_x, freq, :]*100
        run2 = tasks_data[run_y, freq, :]*100
        
        # Calculate min and max for consistent scaling
        min_val = min(min(run1), min(run2))
        max_val = max(max(run1), max(run2))
        padding = (max_val - min_val) * 0.25  # Add 10% margin for better spacing
        y = np.linspace(min_val - padding, max_val + padding, len(run1))
        
        # Calculate p-value using Pearson correlation between run1 and run2
        corr_coeff, p_value = pearsonr(run1, run2)
        
        # Create scatter plot
        plt.scatter(run1, run2, s=150, c = cols[i], label=f'sub-00{sub + 1}')
        
        # Plot the y=x line
        #plt.plot(y, y, linewidth=5, color='k', linestyle='dashed')
        
    # Ensure both axes have the same scale
    plt.axis('square')  # Forces equal aspect ratio (square plot)
    lims = [10,40]
    plt.xlim(lims)
    plt.ylim(lims)
    
    # Format ticks to show only up to 2 decimal places
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    # Customize ticks and axis labels
    plt.xticks(rotation = 45, fontsize=60)
    plt.yticks(fontsize=60)
    
    # Add labels
    plt.xlabel(f'Run {1}(%)', fontsize=80)
    plt.ylabel(f'Run {5}(%)', fontsize=80)
    
    # Add legend
    plt.legend(fontsize=50,loc = 2)
    
    # Use tight layout to avoid clipping axis labels
    plt.tight_layout()
    
    #Save the plot without clipping axis labels
    plt.savefig(
        f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_IS/run_corr/bigfirstlast{freq_name[l]}_run_correlation.png',
        dpi=300,
        bbox_inches='tight'
    )
    
    # Display plot
    plt.show()

# %% Check Corr over individual subjects

#run indices for wach participants
run_1 = [0, 5, 10, 15, 20]
run_2x = [1, 6, 11, 16, 21]
run_2 = [4,9,14,19,24]

for l in range(5):
    sub = l
    print(f'sub-00{sub+1}:')
    for m in range(5):
        freq = m #idx for freq band
        run1 = tasks_data[run_1[l], freq,:]*100
        run2 = tasks_data[run_2[l], freq,:]*100
        corr_coeff, p_value = pearsonr(run1,run2)
        
        print(f'{freq_lab[m]}: ')
        print(f'[{corr_coeff**2:.3f}, {p_value:.2e}] \n')
    print('--------------------')



# %% Check Corr over all subjects
    
for i in range(5):
    freq = i
    all_run1 = []
    all_run2 = []
    print(f'{freq_lab[i]}')
    
    for j in range(5):
        all_run1.extend(tasks_data[run_1[j],freq,:]*100)
        all_run2.extend(tasks_data[run_2[j],freq,:]*100)
        
    corr_coeff, p_value = pearsonr(all_run1, all_run2)
    
    print(f'R, p-value = [{corr_coeff**2:.3f}, {p_value:.2e}] \n')
        

    

        
    
    
    
       



# %% For individual Plotting

lim = [[10,40],[5.5,50],[5.5,40],[1,19],[1,27]]


for l in range(5):
    plt.figure(figsize=(18, 18))
    for i in range(5):
        sub = i
        
        run_x = run_1[i]
        run_y = run_2[i]
        freq = l
        run1 = tasks_data[run_x, freq, :]*100
        run2 = tasks_data[run_y, freq, :]*100
        
        # Calculate min and max for consistent scaling
        min_val = min(min(run1), min(run2))
        max_val = max(max(run1), max(run2))
        padding = (max_val - min_val) * 0.25  # Add 10% margin for better spacing
        y = np.linspace(min_val - padding, max_val + padding, len(run1))
        
        # Calculate p-value using Pearson correlation between run1 and run2
        corr_coeff, p_value = pearsonr(run1, run2)
        
        # Create scatter plot
        plt.scatter(run1, run2, s=350, c = cols[i], label=f'sub-00{sub + 1}')
        
        # Plot the y=x line
        #plt.plot(y, y, linewidth=5, color='k', linestyle='dashed')
        
    # Ensure both axes have the same scale
    plt.axis('square')  # Forces equal aspect ratio (square plot)
    lims = lim[l]
    plt.xlim(lims)
    plt.ylim(lims)
    
    # Format ticks to show only up to 2 decimal places
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    # Customize ticks and axis labels
    plt.xticks(rotation = 45, fontsize=80)
    plt.yticks(fontsize=80)
    
    # Add labels
    plt.xlabel(f'Run {1}(%)', fontsize=100)
    plt.ylabel(f'Run {5}(%)', fontsize=100)
    
    # Add legend
    plt.legend(fontsize=45, loc= 2)
    
    # Use tight layout to avoid clipping axis labels
    #plt.tight_layout()
    
    #Save the plot without clipping axis labels
    plt.savefig(
        f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_IS/run_corr/bigfirstlast{freq_name[l]}_run_correlation.png',
        dpi=300,
        bbox_inches='tight'
    )
    
    print(f'firstlast{freq_name[l]}_run_correlation')
    
    # Display plot
    plt.show()