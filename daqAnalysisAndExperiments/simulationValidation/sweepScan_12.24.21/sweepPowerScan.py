# -*- coding: utf-8 -*-
"""
Spyder Editor

This script is meant to use the rigol RSA5065 to measure the power 
of a CW signal that is stepped through a list of unknown frequencies
but held for a known time (EX: 10 sec/step)
"""



import numpy as np
import pandas as pd
import time
import csv
import os
import configparser
import subprocess
import pyvisa as visa
import sys, smtplib
from stat import S_IREAD, S_IWUSR
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

'''
start = datetime.now()
print(datetime.now() - start)
'''


def startScan(span, centerFrequency, numAverages):
    
    # Set the span (page 2-141)
    INST.write(':SENS:FREQ:SPAN ' + str(span))
    # Set the center frequency (2-139)
    INST.write(':SENS:FREQ:CENT ' + str(centerFrequency))
    # Set the number of averages (2-118)
    INST.write(':SENS:AVER:COUN ' + str(numAverages))
    # Set the power unit to DBM (2-215) - note that linear units do not work
    INST.write(':UNIT:POW DBM')
    
def configure(fileName):
    configParser = configparser.RawConfigParser()  
    configParser.read(fileName)
    return configParser
    
configVals = configure('CONFIG.txt')

# Set the acqusition time
ACQ_TIME = float(configVals.get('Configuration file', 'ACQ TIME'))*0.001

# Set the IP address. Get it from rigol; system/interface/lan/ip. NOT from ipconfig
ADDRESS = configVals.get('Configuration file', 'ADDRESS')

# Set the maximum frequency
END_FREQUENCY = float(configVals.get('Configuration file', 'END FREQUENCY'))*10**6

# Set the preamplifier gain
GAIN = float(configVals.get('Configuration file', 'GAIN'))

# Set the number of averages done on the Rigol
NUM_AVERAGES = 0

# Set the prefix on the saved files
PREFIX = configVals.get('Configuration file', 'PREFIX')

# Set the resolution over span ratio
RATIO = float(configVals.get('Configuration file', 'RATIO'))

# Set the resolution
RESOLUTION = float(configVals.get('Configuration file', 'RESOLUTION'))

# Set the relative level for the display
RLEV = configVals.get('Configuration file', 'RLEV')

# Set the name of status file
SAVE_FILE = configVals.get('Configuration file', 'SAVEFILE')


SAVE_CENT_FILE = 'centFreq.txt'

SAVE_DIR = 'Data'

SAVE_TEMP = 'tempData.txt'

# Set the total amount of time to stay at each window (in minutes)
TOTAL_TIME = float(configVals.get('Configuration file', 'TOTAL TIME'))*60

    
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
nPoints = 8000
INST.write(':SWE:POIN ' + str(nPoints))

#set span (2-143)
startFreq = 40e6
stopFreq = 301e6
INST.write('FREQ:STAR ' + str(startFreq))
INST.write('FREQ:STOP ' + str(stopFreq))

#set rbw (2-121)
INST.write(':SENSe:BANDwidth:RESolution 100000')

# Set the detector into sample detect mode (2-134) POS, SAMP AVER
INST.write('SENS:DET:FUNC POS')

# Set the external gain (2-130)
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
INST.write(':DISP:WIND:TRAC:Y:SCAL:RLEV ' + str(RLEV))

# Set the y-scale per division level to 5 dBm (2-64)
INST.write(':DISP:WIND:TRAC:Y:PDIV ' + str(10))

# Set the window to be rectangular (2-123)
INST.write(':BAND:SHAP KAIS ')

INST.write(':FORM:TRAC:DATA ASC')

#set auto VBW (2-124)
INST.write(':BAND|BWID:VID:AUTO 1')




# Set the trace to clear write (WRIT) or average (AVER) (2-198)
INST.write(':TRAC:MODE WRIT')
INST.write(':TRAC:TYPE WRIT')

#set number of averages 2-118
INST.write(':TRAC:AVER:COUN 10')

#Sets to cont peak search 2-20
INST.write(':CALC:MARK1:CPS:STAT 1')

INST.write(':CALC:MARK:PEAK:TABL 1')

#only get peaks greater than display line(Thanks Ben)
INST.write(':CALC:MARK:PEAK:TABL:READ GTDL')

INST.write(':CALC:MARK:PEAK:THR = -98')

powerDf = pd.DataFrame(columns = ['Frequency (MHz)', 'Power (dBm)'])

print('sleeping for 3 seconds')
time.sleep(3)
print('scanning')


for i in range(12000):
    if i % 10 == 0:
        print(i)
    a = (INST.query(':TRAC:MATH:PEAK?'))
        
    listData = [i for i in a[:-1].split(',')]
    listDataMax = []
    try:
        listDataMax.append(float(listData[0]))
        listDataMax.append(float(listData[1]))
    except:
        print('error')
        time.sleep(.02)
        continue
    powerDf.loc[len(powerDf.index)] = listDataMax
  
    #print(INST.query(':TRAC:MATH:PEAK?'))
    time.sleep(.02)


 
 
powerDf['Frequency (MHz)'] = round(powerDf['Frequency (MHz)'] / 1e6, 1)
groupDf = powerDf.groupby(['Frequency (MHz)']).max()
#groupDfMedian.to_pickle('./')

'''
#check for missing data points
init = 10
freqArr = []
for i in groupDfMedian.index:
    freqArr.append(i)

diffArr=[]
for i in range(len(freqArr)):
    diffArr.append( freqArr[i+1] - freqArr[i])
'''




print('Done!')


    