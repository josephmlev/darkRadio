# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:07:44 2021

@author: phys-simulation
"""

import numpy as np
import time as t
import csv
import os
import configparser
import subprocess
import pyvisa as visa
import serial as pyser
import sys, smtplib
from stat import S_IREAD, S_IWUSR
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

'''
start = datetime.now()
print(datetime.now() - start)
'''
  

ADDRESS = 'TCPIP0::169.254.132.248::INSTR'


startFreq = 50000000 #Hz
stopFreq = 300000000 #Hz
nPoints = 8000 
rbw = 30000 #Hz
vbw = rbw
sweepTime = .3 #sec. This is auto set. Can be overwritten but wonky results
experimentPeriod = 10 #sec
nScans = int(round(experimentPeriod/sweepTime - .25 * experimentPeriod/sweepTime, 0)) #take 25% less because sweeps take longer

backgroundTime = 100 #sec

  
RM = visa.ResourceManager()
INST = RM.open_resource(ADDRESS)
RM.list_resources()

# Open the instrument (note that this is not static and needs to be reset every time you
# restart the script
RM = visa.ResourceManager('@py')
INST = RM.open_resource(ADDRESS)

# Get the number of sweep points (2-161)
#SWEEP_POINTS  = int(INST.query(':SWE:POIN?'))

if 0: #
    # Set the Rigol into real time spectrum analyzer mode (2-89)
    INST.write(':INST:SEL SA') 
    
    # Rigol suggests sleeping for 8 seconds after issuing the previous command
    time.sleep(8)

#set sweep points (2-162)

INST.write(':SWE:POIN ' + str(nPoints))

#set span (2-143)
INST.write('FREQ:STAR ' + str(startFreq))
INST.write('FREQ:STOP ' + str(stopFreq))

#set rbw (2-121)
INST.write(':SENSe:BANDwidth:RESolution 30000')

# Set the detector into sample detect mode (2-134) POS, SAMP AVER
INST.write('SENS:DET:FUNC POS')

# Set the external gain (2-130)
GAIN = 0 
INST.write(':CORR:SA:GAIN ' + str(GAIN))

# Set the reference level for the y scale (2-65)
INST.write(':DISP:WIND:TRAC:Y:SCAL:SPAC LOG')

# Set the y-scale unit to dbm (2-215)
INST.write(':UNIT:POW DBM')

# Set into continuous mode (2-87)
INST.write(':INIT:CONT ON')

# Set the attenuation of the RF front-end attenuator to 0 dB (2-152)
INST.write(':SENS:POW:RF:ATT 0')

# Set the y-scale reference level (2-65)
RLEV = '-40DBM'
INST.write(':DISP:WIND:TRAC:Y:SCAL:RLEV ' + str(RLEV))

# Set the y-scale per division level to 5 dBm (2-64)
INST.write(':DISP:WIND:TRAC:Y:PDIV ' + str(10))

# Set the window to be rectangular (2-123)
INST.write(':BAND:SHAP KAIS ')

INST.write(':FORM:TRAC:DATA ASC')

#set trace detect. RMS takes rms over time. See (2-27) in normal manual (2-134)
INST.write(':DET:TRAC RMS')

# Set the trace to clear write (WRIT) or average (AVER) (2-198)
INST.write(':TRAC:MODE AVER')
INST.write(':TRAC:TYPE AVER')

#set number of averages 2-118
INST.write(':TRAC:AVER:COUN ' + str(nScans))

print('Done!')




freqArr = np.linspace(startFreq, stopFreq, nPoints)

specStr = (INST.query('TRAC:DATA? TRACE1'))

#b = [float(i) for i in a.split(',')] #takes 15ms for 10,000 pts

nBackgroundScans = int(backgroundTime // experimentPeriod)



#init DF and fill with initial background data
specDf = pd.DataFrame(index = freqArr)

for i in range(nBackgroundScans):
   specStr = (INST.query('TRAC:DATA? TRACE1'))
   specArr = [float(i) for i in specStr.split(',')]
   specDf[datetime.now()] = specArr 
   t.sleep(10)
   print(i)
   


#main loop
count = nBackgroundScans#init count where we left off from background
flagDf = pd.DataFrame(index = freqArr)

while(count < 4000):
    time = datetime.now()
    specStr = (INST.query('TRAC:DATA? TRACE1'))
    specDf[time] = [float(i) for i in specStr.split(',')]
    background = specDf.iloc[:,count - nBackgroundScans :count].mean(axis=1)#maybe flip this with line under
    flagDf[time] = specDf[time] - background > 15
    t.sleep(10)
    print(count)
    count += 1

    
'''   
#init background
background = np.empty((nPoints, 2))
for i, amp in enumerate(spec.split(',')):
    background[i,0] = float(amp)  


for i in range(2):
    a = (INST.query('TRAC:DATA? TRACE1'))
    for i, amp in enumerate(a.split(',')):
        background[i,1] = float(amp)
    background[:,0] = background.mean(1)
    time.sleep(.2)



specDf = pd.DataFrame(index = freqArr)

for i in range(1):
    a = (INST.query('TRAC:DATA? TRACE1'))
    b = [float(i) for i in a.split(',')]
    specDf[datetime.now()] = b 
    time.sleep(0)
    print(i)
'''
plt.close('all')
plt.scatter(freqArr, specArr)