#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fancy surface plots using surfplot

@author: sebastiancoleman
"""

from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps import transforms
from surfplot import Plot, utils
import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from nilearn import image
from scipy.io import loadmat
#%matplotlib qt

#%%  Atlas Surface Construction and Plotter
def atlas_surface_plotter(atlas_file, values, threshold=0, cmap='YlOrRd_r', symmetric_cbar=False):
    
    # make volume image of strength  
    atlas_img = image.load_img(atlas_file)
    atlas_data = atlas_img.get_fdata()

    # different procedure for 3D vs 4D parcellation files
    if len(atlas_img.shape)==4:
        
        # empty image
        single_vol = image.index_img(atlas_img, 0)
        atlas_new = np.zeros(np.shape(atlas_data))
        
        # place values in each parcel region
        for reg in range(len(values)):
            atlas_new[:, :, :, reg] = atlas_data[:, :, :, reg] * values[reg]
        
        # mean over fourth dimension
        atlas_new = np.mean(atlas_new, 3) * len(values)
        
        # make image from new atlas data
        new_img = nib.Nifti1Image(atlas_new, single_vol.affine, single_vol.header)

    elif len(atlas_img.shape)==3:

        # empty image
        atlas_new = np.zeros(np.shape(atlas_data))

        # place values in each parcel region
        indices = np.unique(atlas_data[atlas_data>0])
        for reg in range(len(values)):
            reg_mask = atlas_data == indices[reg]
            atlas_new[reg_mask] = values[reg]

        # make image from new atlas data
        new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)

    # save temporary image
    nib.save(new_img, '/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/fig.nii.gz')

    # transform to fsaverage surface
    fslr = transforms.mni152_to_fsaverage('/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/fig.nii.gz')
    lh_data, rh_data = fslr[0].agg_data(), fslr[1].agg_data()
    lh_data = utils.threshold(lh_data, threshold, two_sided=symmetric_cbar)
    rh_data = utils.threshold(rh_data, threshold, two_sided=symmetric_cbar)

    # plot
    surfaces = fetch_fsaverage()
    lh, rh = surfaces['pial']#'_left'], surfaces['pial_right']
    p = Plot(surf_lh=lh, surf_rh=rh, brightness=0.6)
    if symmetric_cbar:
        vmax = np.max(np.abs(values))
        p.add_layer({'left': lh_data, 'right': rh_data}, color_range=[-vmax, vmax], cmap=cmap, cbar=False)
    else:
        vmax = np.max(values)
        vmin = np.min(values)
        p.add_layer({'left': lh_data, 'right': rh_data}, color_range=[vmin, vmax], cmap=cmap, cbar=False)
    fig = p.build()
    fig.show()
    
    return fig

def construct_mpl_surf_image(surf_fig, values, figsize=(5,4), cmap='YlOrRd_r', symmetric=False, cbar_label='Brain Activity'):

    # extract image data
    canvas = surf_fig.canvas
    canvas.draw()
    imflat = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    imdata = imflat.reshape(*reversed(canvas.get_width_height()), 4)
    imdata = imdata[:, :, :3]
    plt.close()
    
    # remove white space
    nonwhite_pix = (imdata != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_row[::4] = True
    nonwhite_col = nonwhite_pix.any(0)
    nonwhite_col[::4] = True
    imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
    
    # construct plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
    cax = fig.add_axes([0.25, 0.15, 0.5, 0.03])  # [left, bottom, width, height]
    
    # brain
    ax.imshow(imdata_cropped)
    ax.axis("off")
    
    # colorbar
    if symmetric:
        vmax = np.max(np.abs(values))
        vmin = -vmax
    else:
        vmax = np.max(values)
        vmin = np.min(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    
    cax.tick_params(labelsize=5)
    
    # Set the number of ticks on the colorbar
    #ticks = np.linspace(vmin, vmax, 6)  # Create the desired number of ticks
    #cbar.set_ticks(ticks)
    
    return fig, ax, cax



#%%  Plotting Function
def plottingFunc(val_array, cbarLabel, thresholdval = 0.001, cmapval = "Purples"):
    atlas_file = '/d/mjt/s4/toolboxes/brainnetviewer/2019-10-31/Data/ExampleFiles/AAL90/aal.nii' #'/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
    #atlas_file = r'X:\toolboxes\brainnetviewer\2019-10-31\Data\ExampleFiles\AAL90\aal.nii'
    values= val_array #alpha_values #BE_data[0][1]
    
    fig1 = atlas_surface_plotter(atlas_file, values, threshold=thresholdval, cmap=cmapval)
    fig, ax, cax = construct_mpl_surf_image(fig1, values, figsize=(6,5), 
                                            cmap=cmapval, symmetric=False, 
                                            cbar_label=cbarLabel)
    return fig1, ax, cax


# #%%

# task_id = 'IS' #BE/IS
# tasks_data = loadmat(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/{task_id}_rest_power.mat').get(f'{task_id}_data')
# #alpha_values = np.mean(BE_data[:, 4, :], axis = 0)

# #%% Plot Average Power per Frequency

# file_label = ['theta', 'alpha', 'beta', 'lowGamma', 'highGamma']
# freq_lab = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
# col_lab = ['Blues', 'Purples', 'Greens', 'Oranges', 'Reds']

# for i in range(len(tasks_data[0])):
#     val_arr = np.mean(tasks_data[:,i,:], axis = 0)
#     label = file_label[i]
#     freq = f' Average {freq_lab[i]} activity'
    
#     plottingFunc(val_arr, freq, cmapval = col_lab[i])
#     plt.savefig(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_{task_id}/avg_{label}.png')
        

# #%%

# file_label = ['theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
# freq_lab2 = [ 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
# col_lab = ['Blues', 'Purples', 'BuPu', 'Oranges', 'Reds']
# for k in range(len(freq_lab2)):
#     freq = freq_lab2[k]
#     file = file_label[k]
#     for j in range(len(tasks_data)):
#         val_arr = tasks_data[j,k,:]
        
        
        
#         # Calculate RP and run indices
#         rp_num = (j // 5) + 1  # RP01, RP02, etc. (0-4 is RP01, 5-9 is RP02, etc.)
#         run_num = (j % 5) + 1  # Run 1-5 within each RP
        
#         label = f'{freq} Activity RP0{rp_num}-0{run_num}'
        
#         plottingFunc(val_arr, label, cmapval=col_lab[k])
#         plt.savefig(f'/d/mjt/9/projects/OPM-Analysis/OPM_retest/code/plot_dump_{task_id}/individual_{file}/RP0{rp_num}_0{run_num}_{file}.png')
    
