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

ADDRESS = 'TCPIP0::169.254.113.164::INSTR'
startFreq = 2.4799991 * 10**9 #Hz
stopFreq =  2.4799993 * 10**9 #Hz
nPoints = 10000 
rbw = 1 #Hz
freqArr = np.linspace(startFreq, stopFreq, nPoints)

RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)
INST.write(':TRAC:AVER:COUN 10')


INST.write(':SWE:POIN ' + str(nPoints))
INST.write('FREQ:STAR ' + str(startFreq))
INST.write('FREQ:STOP ' + str(stopFreq))
INST.write('SENS:BAND:RES ' + str(rbw))


INST.write(':INIT:CONT OFF')
print('scanning...')
time.sleep(5)
specStr = (INST.query('TRAC:DATA? TRACE1'))
specArr = np.array([float(i) for i in specStr.split(',')])
print('got it')
print()


plt.figure()
plt.plot(freqArr, specArr)
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Yagi, RF pre off, 67dB amp chain.')
