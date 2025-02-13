# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:56:00 2024

@author: Julian Bandhan
"""

import mne
import numpy as np
from matplotlib import pyplot as plt
import os.path as op


import pandas as pd
import pingouin as pg


        
participants = np.repeat(np.arange(2,5),5)
sessions = np.tile(np.arange(1,6),3)

data = pd.read_csv(r'Y:\projects\OPM-Analysis\OPM_retest\code\pyscripts\icc_temp.txt', header = None)[0]

df = pd.DataFrame({
    'Participant':participants,
    'Session':sessions,
    'Power':data
    })

icc = pg.intraclass_corr(data=df, targets = 'Participant', raters='Session', ratings = 'Power')
icc_value = icc[icc['Type'] == 'ICC1']['ICC'].values[0]
icc_value_1k = icc[icc['Type'] == 'ICC1k']['ICC'].values[0]

print(f'{icc_value:.3f}')
print(f'{icc_value_1k:.3f}')