# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:43:13 2021

@author: phys-simulation
"""

import numpy as np
import time
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
This is my first attempt at this code. 10/17/21. 

The idea is to set the rigol into constant scan mode and ask it for a new spectrum 
every 2ish s after 10 averages. I'll turn on some rf at 20MHz and see if we can see it 
above a baseline computing by averaging together several scans (that were preaveraged on 
rigol)

To load files:
dfName = pd.read_pickle('testData_19.5MHz_10.17.21.pkl')
'''

freqArr = np.linspace(startFreq, stopFreq, nPoints)


a = (INST.query('TRAC:DATA? TRACE1'))

#b = [float(i) for i in a.split(',')] #takes 15ms for 10,000 pts

#init background
background = np.empty((nPoints, 2))
for i, amp in enumerate(a.split(',')):
    background[i,0] = float(amp)  

start = datetime.now()
for i in range(2):
    a = (INST.query('TRAC:DATA? TRACE1'))
    for i, amp in enumerate(a.split(',')):
        background[i,1] = float(amp)
    background[:,0] = background.mean(1)
    time.sleep(.2)
print(datetime.now() - start)


specDf = pd.DataFrame(index = freqArr)

for i in range(1):
    a = (INST.query('TRAC:DATA? TRACE1'))
    b = [float(i) for i in a.split(',')]
    specDf[datetime.now()] = b 
    time.sleep(0)
    print(i)

plt.close('all')
plt.scatter(freqArr, b)


#background = np.empty((nPoints, 2))
#background[(:,0)] = b

'''
for i in range(4):
    start = datetime.now()
    a = (INST.query('TRAC:DATA? TRACE1'))
    print(datetime.now() - start)
    time.sleep(0)
    b = [float(i) for i in a.split(',')] #takes 3 ~ 15ms for 10,000 pts (I think it depends what cpu is doing)
    print(b[5000])
'''