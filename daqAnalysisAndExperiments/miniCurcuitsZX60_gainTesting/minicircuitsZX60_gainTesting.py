# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:11:55 2021

@author: phys-simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pyvisa as visa

plt.close('all')

ADDRESS = 'TCPIP0::169.254.206.61::INSTR'
startFreq = 3800 * 10**6 #Hz
stopFreq = 4200 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq, stopFreq, nPoints)

if 0:
    RM = visa.ResourceManager()
    INST = RM.open_resource(ADDRESS)
    INST.write(':TRAC:AVER:COUN 250')
    
    
    INST.write(':SWE:POIN ' + str(nPoints))
    INST.write('FREQ:STAR ' + str(startFreq))
    INST.write('FREQ:STOP ' + str(stopFreq))
    
    
    INST.write(':INIT:CONT OFF')
    print('scanning...')
    time.sleep(30)
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    normSpecArr = np.array([float(i) for i in specStr.split(',')])
    print('got it')
    print()
    np.save('normilize_TGn40dBm_23dBatt', normSpecArr)
    np.save('freqArr', freqArr)
    
    input('press enter')
    INST.write(':INIT:CONT OFF')
    print('scanning...')
    time.sleep(30)
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    ampSpecArr = np.array([float(i) for i in specStr.split(',')])
    print('got it')
    print()
    np.save('bothAmps_n40dBm_30dBatt', ampSpecArr)
    ampGainSpec = ampSpecArr - normSpecArr
    np.save('ampGain', ampGainSpec)




normSpecArr = np.load('normilize_TGn40dBm_23dBatt.npy')
ampSpecArr = np.load('bothAmps_n40dBm_30dBatt.npy')
ampGainSpec = ampSpecArr - normSpecArr

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

w = 100
normSpecArrAvg = moving_average(normSpecArr, w)
ampSpecArrAvg = moving_average(ampSpecArr, w)
ampGainSpecAvg = ampSpecArrAvg - normSpecArrAvg

    
plt.figure()
plt.plot(freqArr/1e6, ampGainSpec)
plt.plot(freqArr[(w-1):]/1e6, ampGainSpecAvg, label= 'Moving Average')
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Gain of Both Minicircuits ZX60 amps')
plt.legend()

plt.figure()
plt.plot(freqArr/1e6, normSpecArr)
plt.plot(freqArr[(w-1):]/1e6, normSpecArrAvg, label = 'Moving Average')
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Tracking gen into 30dB attenuaton')
plt.legend()

plt.figure()
plt.plot(freqArr/1e6, ampSpecArr)
plt.plot(freqArr[(w-1):]/1e6, ampSpecArrAvg, label = 'Moving Average')
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Unnormilized Track Gen to 30dB Attenuation to Both Amps')
plt.legend()

