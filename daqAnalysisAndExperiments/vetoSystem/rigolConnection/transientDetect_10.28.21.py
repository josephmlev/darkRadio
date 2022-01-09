# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:11:55 2021

@author: phys-simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.close('all')

numberScans = 4000
print('number of scans =', numberScans)
specDf =  pd.read_pickle('12hrSpec_10.30.21.pkl').iloc[:, 0:numberScans]
specDf24 =  pd.read_pickle('24hrSpec_10.26.21.pkl').iloc[:, :]
startFreq = 50000000 #Hz
stopFreq = 300000000 #Hz
nPoints = 8000 
rbw = 30000 #Hz
vbw = rbw
sweepTime = .3 #sec. This is auto set. Can be overwritten but wonky results
experimentPeriod = 10 #sec


backgroundTime = 10 #sec
flagThreshold = 30 #dB

freqArr = np.linspace(startFreq, stopFreq, nPoints)/1e6
nBackgroundScans = int(backgroundTime // experimentPeriod)



#main loop


#method 1 (for loop)
if 0:
    flagDf = pd.DataFrame()
    count = nBackgroundScans#init count where we left off from background
    start = datetime.now()
    for i, scanTime in enumerate(list(specDf)[nBackgroundScans :]):
        background = specDf.iloc[:,i - nBackgroundScans : i].mean(axis=1)
        flagDf[scanTime] = specDf[scanTime] - background > flagThreshold
    print(datetime.now() - start)


if 0:
    flagDf = pd.DataFrame()
    backgroundDf = pd.DataFrame()
    count = nBackgroundScans#init count where we left off from background
    start = datetime.now()
    for i, scanTime in enumerate(list(specDf)[nBackgroundScans :]):
        backgroundDf[scanTime] = specDf.iloc[:,i - nBackgroundScans : i].mean(axis=1)
        flagDf[scanTime] = backgroundDf.iloc[:,i] - backgroundDf.iloc[:, i-1] > flagThreshold
    print(datetime.now() - start)

#method 2
if 1:
 
    backgroundDf2 = pd.DataFrame()
    flagDf = pd.DataFrame()
    start = datetime.now()
    #backgroundDf2 = specDf.rolling(window = nBackgroundScans, axis = 1).mean()
    #flagDf2 = backgroundDf2.diff(axis = 1) > flagThreshold
    flagDf = specDf24.rolling(window = nBackgroundScans, axis = 1).mean().diff(axis = 1) > flagThreshold
    print( datetime.now() - start)

nDetections = flagDf.sum(axis = 1)
totalDetections = nDetections.sum() 

plt.figure()
plt.xlabel('Frequency (MHz)')
plt.ylabel("Number of Detections")
plt.title('Number of Detections vs Frequency. Threshold = %i dB' % flagThreshold)
plt.plot(freqArr, nDetections)
plt.legend()

plt.figure()
plt.xlabel('Frequency (MHz)')
plt.ylabel("Number of Detections")
plt.title('Normalized Number of Detections vs Frequency. Threshold = %i dB' % flagThreshold)
plt.plot(freqArr, nDetections/totalDetections)
plt.legend()


#pdf at problem frequency
    
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


suspectFreq = 126.6
suspectIdx = find_nearest(freqArr, suspectFreq)
thresholdArr = np.linspace(5,30,50)
flagThresholdDf = pd.DataFrame(index = thresholdArr)
suspectDf = specDf.iloc[suspectIdx, :]
for flagThreshold in thresholdArr:
    flagThresholdDf.loc[flagThreshold, 0] = ((suspectDf.rolling(window = nBackgroundScans).mean().diff() > flagThreshold).sum())

plt.figure()
plt.plot(flagThresholdDf)
plt.title('Total Number of detections at %f MHz' % suspectFreq)
plt.xlabel('Threshold (dB)')
plt.ylabel('Number of Detections')
 

 

plt.figure()
plt.xlabel('Frequency (MHz)')
plt.ylabel("Number of Detections")
plt.title('Number of Detections vs Frequency. Threshold = %i dB' % flagThreshold)
plt.plot(freqArr, nDetections)
plt.legend()

#total detections as a function of threshold

if 0:
 
    backgroundDf2 = pd.DataFrame()
    flagDf = pd.DataFrame()
    start = datetime.now()
    thresholdArr = np.linspace(5,30,2)
    flagThresholdAllFreqDf = pd.DataFrame(index = thresholdArr)
    for flagThreshold in thresholdArr:
        flagThresholdAllFreqDf.loc[flagThreshold, 0] = (specDf.rolling(window = nBackgroundScans, axis = 1).mean().diff() > flagThreshold).sum()
    print( datetime.now() - start)

if 0:
 
    backgroundDf2 = pd.DataFrame()
    totalDetectionsAllFreqDf = pd.DataFrame()
    flagDf = pd.DataFrame()
    start = datetime.now()
    backgroundDf2 = specDf.rolling(window = nBackgroundScans, axis = 1).mean()
    thresholdArr = np.linspace(5,30,2)
    for flagThreshold in thresholdArr:
         totalDetectionsAllFreq = (  specDf.rolling(window = nBackgroundScans, axis = 1).mean().diff(axis = 1) > flagThreshold).sum(axis = 1).sum()
         #totalDetectionsAllFreqDf.loc[flagThreshold, 0]
    print( datetime.now() - start)
    nDetections = flagDf.sum(axis = 1)
    totalDetections = nDetections.sum()






averageSpec = specDf.iloc[:, 0:3999 ].mean(axis=1)
averageSpec24 = specDf24.iloc[:, 0:3999 ].mean(axis=1)

plt.figure()
plt.plot(freqArr, averageSpec)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBm)')



plt.figure()
plt.title('Background Spectrum Close and Far From Rigol' )
plt.plot(freqArr, averageSpec, label = "In Dan's office")
plt.plot(freqArr, averageSpec24, label = 'Close to Rigol')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBm)')
plt.legend()

