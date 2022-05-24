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

ADDRESS = 'TCPIP0::169.254.158.45::INSTR'
startFreq = 800 * 10**6 #Hz
stopFreq = 1200 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq, stopFreq, nPoints)/1e6

RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)
INST.write(':TRAC:AVER:COUN 100')


INST.write(':SWE:POIN ' + str(nPoints))
INST.write('FREQ:STAR ' + str(startFreq))
INST.write('FREQ:STOP ' + str(stopFreq))

if 0:
    INST.write(':INIT:CONT OFF')
    print('scanning...')
    time.sleep(30)
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    specArr = np.array([float(i) for i in specStr.split(',')])
    print('got it')
    print()

    np.save('50OhmTermCenterOfRoom_67dBGain_800-1200MHz_100Avg_100kRBW_rfPreOn', specArr)


littleDipole = np.load('littleDipoleCenterOfRoom_67dBGain_800-1200MHz_100Avg_100kRBW_rfPreOn.npy')
yagi = np.load('yagiCenterOfRoom_67dBGain_800-1200MHz_100Avg_100kRBW_rfPreOn.npy')
vivaldi = np.load('littleVivaldiCenterOfRoom_67dBGain_800-1200MHz_100Avg_100kRBW_rfPreOn.npy')
term = np.load('50OhmTermCenterOfRoom_67dBGain_800-1200MHz_100Avg_100kRBW_rfPreOn.npy')

yagiMedian = [np.median(x) for x in np.reshape(yagi, (-1,10))] 
dipoleMedian = [np.median(x) for x in np.reshape(littleDipole, (-1,10))] 
vivaldiMedian = [np.median(x) for x in np.reshape(vivaldi, (-1,10))] 
termMedian = [np.median(x) for x in np.reshape(term, (-1,10))] 


plt.figure()
plt.plot(freqArr, yagi, 'b',  label = 'Yagi')
plt.plot(freqArr, littleDipole, 'g', label = 'Dipole', alpha = 1)
plt.plot(freqArr, vivaldi, 'r', label = 'Vivaldi', alpha = .6)
plt.plot(freqArr, term, 'c', label = 'Term', alpha = .5)
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('67dB amp chain.')
plt.legend()

plt.figure()
plt.plot(freqArr[::10], yagiMedian, 'b', label = 'Yagi')
plt.plot(freqArr[::10], dipoleMedian, 'g', label = 'Dipole', alpha = .7)
plt.plot(freqArr[::10], vivaldiMedian, 'r', label = 'Vivaldi', alpha = .6)
plt.plot(freqArr[::10], termMedian, 'y', label = 'Term', alpha = .5)
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('67dB amp chain. Median')
plt.legend()