# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:48:54 2021

@author: phys-simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

specDf = pd.read_pickle('testData_19.5MHz_10.17.21.pkl')

startFreq, stopFreq, nPoints = 10000000, 30000000, 10000
freqArr = np.linspace(startFreq, stopFreq, nPoints)
background = specDF.iloc[:, 0:20].mean(axis = 'columns')

noSig = specDf.loc[:, '2021-10-17 22:13:25.176859']
sig = specDf.loc[:, '2021-10-17 22:13:39.361080']


noSigDiff = noSig - background
sigDiff =  sig - background

noSigZ = stats.zscore(noSigDiff)
sigZ = stats.zscore(sigDiff)





plt.close('all')


plt.figure()
plt.title('Z score of scan 1 vs Freq')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Z score')
plt.plot(freqArr, noSigZ)

plt.figure()
plt.title('Z score of scan 2 vs Freq')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Z score')
plt.scatter(freqArr, sigZ)

plt.figure()
plt.title('Histogram of scan 1 Zscore')
plt.hist(noSigZ, bins = 500)
plt.xlabel('Sigma')
plt.ylabel('Frequency')

plt.figure()
plt.title('Histogram of scan 2 Zscore')
plt.hist(sigZ, bins = 500)
plt.xlabel('Sigma')
plt.ylabel('Frequency')


plt.figure()
plt.title('Background')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dBm)')
plt.plot(freqArr, background, label = 'Background')
plt.legend()

plt.figure()
plt.title('Scan 1 (2021-10-17 22:13:25.17)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dBm)')
plt.plot(freqArr, noSig)

plt.figure()
plt.title('Scan 2 (2021-10-17 22:13:39.36)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dBm)')
plt.plot(freqArr, sig)

fig1 = plt.figure()
plt.title('Scan 1 and difference from background (2021-10-17 22:13:25.17)')
frame1=fig1.add_axes((.1,.3,.8,.57))
plt.plot(freqArr, noSig, label = "Scan")
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dBm]')
frame2=fig1.add_axes((.1,.1,.8,.3))    
plt.ylabel('Difference in power')    
plt.plot(freqArr, noSigDiff, 'r', label = 'scan - background')
plt.legend()

fig1 = plt.figure()
plt.title('Scan 2 and difference from background (2021-10-17 22:13:39.36)')
frame1=fig1.add_axes((.1,.3,.8,.57))
plt.plot(freqArr, sig, label = "Scan")
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dBm]')
frame2=fig1.add_axes((.1,.1,.8,.3))    
plt.ylabel('Difference in power')    
plt.plot(freqArr, sigDiff, 'r', label = 'Scan - Background')
plt.legend()