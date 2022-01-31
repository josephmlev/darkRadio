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


ADDRESS = 'TCPIP0::169.254.245.69::INSTR'
RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)



startFreq = 1 * 10**6 #Hz
stopFreq = 315 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq/1e6, stopFreq/1e6, nPoints)

if 0:
    INST.write(':SWE:POIN ' + str(nPoints))
    INST.write('FREQ:STAR ' + str(startFreq))
    INST.write('FREQ:STOP ' + str(stopFreq))

input('scan NS off and hit enter')
specStr = (INST.query('TRAC:DATA? TRACE1'))
powerArr = np.array([float(i) for i in specStr.split(',')])


df = pd.DataFrame()
df['Frequency (MHz)'] = freqArr
df['P_cold'] = powerArr

input('scan NS on and hit enter')
specStr = (INST.query('TRAC:DATA? TRACE1'))
powerArr = np.array([float(i) for i in specStr.split(',')])
df['P_on'] = powerArr

input('scan short and hit enter')
specStr = (INST.query('TRAC:DATA? TRACE1'))
powerArr = np.array([float(i) for i in specStr.split(',')])
df['P_background'] = powerArr

LG_ENR = (10**(df['P_hot']/10) - 10**(df['P_background']/10)) / (10**(df['P_cold']/10) - 10**(df['P_background']/10)) 

df['ENR'] = LG_ENR
df['log ENR'] = 10*np.log10(LG_ENR)

#plt.plot(freqArr,df['log ENR'].rolling(1000).median())

if 0:
    timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')[:19]
    
    df.to_pickle(timestamp + '_' + filename + '.pkl')
    print(timestamp + '_' + filename + '.pkl')
    










