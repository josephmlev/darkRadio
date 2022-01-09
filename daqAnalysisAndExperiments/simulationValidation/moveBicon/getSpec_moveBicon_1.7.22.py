#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:53:07 2021

@author: joseph
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d
import pyvisa as visa
from datetime import datetime
import time as time

plt.close('all')


#Note: freqArr is calculated to have same points as spectrum analizer


plt.close('all')


ADDRESS = 'TCPIP0::169.254.174.204::INSTR'
RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)



startFreq = 5 * 10**6 #Hz
stopFreq = 300 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq/1e6, stopFreq/1e6, nPoints)

if 1:
    INST.write(':SWE:POIN ' + str(nPoints))
    INST.write('FREQ:STAR ' + str(startFreq))
    INST.write('FREQ:STOP ' + str(stopFreq))

x = 0 # -0.698, 0.002, 0.701
y = 0 # -0.312, 0.014, 0.174
z = 0 # -1.204, 0.015 
filename = 'random_hanging'


specStr = (INST.query('TRAC:DATA? TRACE1'))
powerArr = np.array([float(i) for i in specStr.split(',')])


df = pd.DataFrame()
df['Frequency (MHz)'] = freqArr
df['x'] = x
df['y'] = y
df['z'] = z
df['Raw Power (dBm)'] = powerArr

if 1:
    timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')[:19]
    
    df.to_pickle(timestamp + '_' + filename + '.pkl')
    print(timestamp + '_' + filename + '.pkl')
    










