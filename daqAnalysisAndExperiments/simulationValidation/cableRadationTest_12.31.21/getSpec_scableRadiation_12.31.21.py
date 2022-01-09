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


ADDRESS = 'TCPIP0::169.254.118.80::INSTR'
RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)



startFreq = 30 * 10**6 #Hz
stopFreq = 300 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq/1e6, stopFreq/1e6, nPoints)

if 0:
    INST.write(':SWE:POIN ' + str(nPoints))
    INST.write('FREQ:STAR ' + str(startFreq))
    INST.write('FREQ:STOP ' + str(stopFreq))



specStr = (INST.query('TRAC:DATA? TRACE1'))
powerArr = np.array([float(i) for i in specStr.split(',')])



df = pd.DataFrame()
df['Frequency (MHz)'] = freqArr
df['Raw Power (dBm)'] = powerArr
if 1:
    timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')[:19]
    filename = 'roomCableTo93OhmTermTG_biconSA_v2'
    df.to_pickle(timestamp + '_' + filename + '.pkl')
    print(timestamp + '_' + filename + '.pkl')



'''
Z0 = 50

freqRigExpert, R, X = np.loadtxt('rawCsv/Z_30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2.csv', delimiter=",", unpack = True)
rInterp = interp1d(freqRigExpert,R)(freqArr)
xInterp = interp1d(freqRigExpert,X)(freqArr)
zInterp = rInterp + 1j* xInterp
absZInterp = abs(zInterp)

gamma = np.sqrt((R - Z0)**2 + X**2) / np.sqrt((R + Z0)**2 + X**2)
gammaInterp = interp1d(freqRigExpert,gamma)(freqArr)



df = pd.DataFrame()

df['Frequency (MHz)'] = freqArr
df['Raw Antenna Power (dBm)'] = powerArr
df['Complex Z'] = zInterp
df['Magnitude Z'] = absZInterp
df['Reflection Coefficent'] = gammaInterp
df['VSWR'] = (1 + gammaInterp) / (1 - gammaInterp)
df['Reflection Coefficent'] = -20 * np.log10(gammaInterp)
df['Matching Loss'] = -10 * np.log10(1 - gammaInterp**2)
df['Normilized Antenna Power (dBm)'] = df['Raw Antenna Power (dBm)'] + df['Matching Loss']


if 0:
    filename = '30cmDipClip_noChokes_60N60E57V_20dBatt_rbw10k_DANLn100pm3dB_avg4_v2.pkl'
    df.to_pickle(filename)
    '''









