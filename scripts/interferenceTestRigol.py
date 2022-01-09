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

ADDRESS = 'TCPIP0::169.254.132.248::INSTR'
startFreq = 50 * 10**6 #Hz
stopFreq = 1000 * 10**6 #Hz
nPoints = 10000 
freqArr = np.linspace(startFreq, stopFreq, nPoints)

RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)
INST.write(':TRAC:AVER:COUN 30')


INST.write(':SWE:POIN ' + str(nPoints))
INST.write('FREQ:STAR ' + str(startFreq))
INST.write('FREQ:STOP ' + str(stopFreq))

cont = 1
while cont == 1:
    print('turn off interference source under test. enter anything when ready to scan')
    inp = input()
    INST.write(':INIT:CONT OFF')
    print('scanning...')
    time.sleep(10)
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    specArrOff = np.array([float(i) for i in specStr.split(',')])
    print('got it')
    print()
    
    print('turn on interference source under test. enter anything when ready to scan')
    inp = input()
    INST.write(':INIT:CONT OFF')
    print('scanning...')
    time.sleep(10)
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    specArrOn = np.array([float(i) for i in specStr.split(',')])
    print('got it')
    print()
    
    print('Enter y to continue')
    inp = input()
    if inp == 'y':
        print('here we go again!')
        print()
        cont = 1
    else:
        cont = 0
    INST.write(':INIT:CONT OFF')

plt.figure()
plt.plot(freqArr, specArrOn - specArrOff)