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

ADDRESS = 'TCPIP0::169.254.199.115::INSTR'
#ADDRESS = 'TCPIP0::0.0.0.0::INSTR'
startFreq = 30 * 10**6 #Hz
stopFreq =  300 * 10**6 #Hz
nPoints = 10000 
rbw = 100 #Hz
freqArr = np.linspace(startFreq, stopFreq, nPoints)

RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)
#INST.write(':TRAC:AVER:COUN 10')


#INST.write(':SWE:POIN ' + str(nPoints))
#INST.write('FREQ:STAR ' + str(startFreq))
#INST.write('FREQ:STOP ' + str(stopFreq))
#INST.write('SENS:BAND:RES ' + str(rbw))


#INST.write(':INIT:CONT OFF')
print('scanning...')
#time.sleep(3)
specStr = (INST.query('TRAC:DATA? TRACE1'))
specArr = np.array([float(i) for i in specStr.split(',')])
print('got it')
print()


plt.figure()
plt.plot(freqArr/1e6, specArr)
plt.xlabel('Freq. (MHz)')
plt.ylabel('Power (dBm)')
plt.title('Bicon in Room')